import torch
import torch.nn as nn
from common.utils.human_models import mano
from common.nets.module import BackboneNet,Extra_hand_feature,RotationNet,Exact_Prior_feature
from common.nets.loss import CoordLossL1, ParamLossL1,CoordLossL2,ParamLossL2,CoordBCELoss,ParamBCELoss
import copy
from thop import profile
from torchinfo import summary

class Model(nn.Module):
    def __init__(self, backbone_net, hand_feature_extra,prior_feat,rotationNet):
        super(Model, self).__init__()

        # modules
        self.backbone_net = backbone_net
        self.hand_feature_extra = hand_feature_extra
        self.prior_feat=prior_feat
        self.rotation_net=rotationNet

        self.mano_layer_right = copy.deepcopy(mano.layer['right']).cuda()
        self.mano_layer_left = copy.deepcopy(mano.layer['left']).cuda()

        # loss functions
        self.coord_loss = CoordLossL1()
        self.param_loss = ParamLossL1()

    def get_coord(self, root_pose, hand_pose, shape, cam_param, hand_type):
        batch_size = root_pose.shape[0]
        if hand_type == 'right':
            output = self.mano_layer_right(betas=shape, hand_pose=hand_pose, global_orient=root_pose)
        else:
            output = self.mano_layer_left(betas=shape, hand_pose=hand_pose, global_orient=root_pose)

        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        joint_cam = torch.bmm(torch.from_numpy(mano.sh_joint_regressor).cuda()[None, :, :].repeat(batch_size, 1, 1),
                              mesh_cam)

        # project 3D coordinates to 2D space
        x = joint_cam[:, :, 0] * cam_param[:, None, 0] + cam_param[:, None, 1]
        y = joint_cam[:, :, 1] * cam_param[:, None, 0] + cam_param[:, None, 2]
        joint_proj = torch.stack((x, y), 2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:, mano.sh_root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam

        return joint_proj, joint_cam, mesh_cam, root_cam

    def forward(self, inputs, targets, meta_info, mode):
        input_img = inputs['img']
        img_feat = self.backbone_net(input_img)

        hand_feature_l,hand_feature_r,ljoint_img,rjoint_img= self.hand_feature_extra(img_feat)

        l_proir,r_proir=self.prior_feat(hand_feature_l,hand_feature_r)

        rroot_pose, rhand_pose, rshape, rcam_param, lroot_pose, lhand_pose, lshape, lcam_param, rel_trans = \
            self.rotation_net(hand_feature_l,hand_feature_r,ljoint_img.detach(),rjoint_img.detach(),l_proir,r_proir)

        # get outputs
        ljoint_proj, ljoint_cam, lmesh_cam, lroot_cam = self.get_coord(lroot_pose, lhand_pose, lshape, lcam_param,
                                                                       'left')
        rjoint_proj, rjoint_cam, rmesh_cam, rroot_cam = self.get_coord(rroot_pose, rhand_pose, rshape, rcam_param,
                                                                       'right')

        # combine outputs for the loss calculation (follow mano.th_joints_name)
        mano_pose = torch.cat((rroot_pose, rhand_pose, lroot_pose, lhand_pose), 1)
        mano_shape = torch.cat((rshape, lshape), 1)
        joint_cam = torch.cat((rjoint_cam, ljoint_cam), 1)
        joint_img = torch.cat((rjoint_img, ljoint_img), 1)
        joint_proj = torch.cat((rjoint_proj, ljoint_proj), 1)

        if mode == 'train':
            loss = {}
            loss['rel_trans'] = self.coord_loss(rel_trans[:, None, :], targets['rel_trans'][:, None, :],
                                                meta_info['rel_trans_valid'][:, None, :])
            loss['mano_pose'] = self.param_loss(mano_pose, targets['mano_pose'], meta_info['mano_param_valid'])
            loss['mano_shape'] = self.param_loss(mano_shape, targets['mano_shape'], meta_info['mano_shape_valid'])
            loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'],
                                                meta_info['joint_valid'] * meta_info['is_3D'][:, None, None]) * 10
            loss['mano_joint_cam'] = self.coord_loss(joint_cam, targets['mano_joint_cam'],
                                                     meta_info['mano_joint_valid']) * 10
            loss['joint_img'] = self.coord_loss(joint_img, targets['joint_img'], meta_info['joint_trunc'],
                                                meta_info['is_3D'])
            loss['mano_joint_img'] = self.coord_loss(joint_img, targets['mano_joint_img'],
                                                     meta_info['mano_joint_trunc'])
            loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:, :, :2], meta_info['joint_valid'])

            return loss
        else:
            # test output
            out = {}
            out['img'] = inputs['img']
            out['rel_trans'] = rel_trans
            out['lmano_mesh_cam'] = lmesh_cam
            out['rmano_mesh_cam'] = rmesh_cam
            out['lmano_root_cam'] = lroot_cam
            out['rmano_root_cam'] = rroot_cam
            out['lmano_joint_cam'] = ljoint_cam
            out['rmano_joint_cam'] = rjoint_cam
            out['lmano_root_pose'] = lroot_pose
            out['rmano_root_pose'] = rroot_pose
            out['lmano_hand_pose'] = lhand_pose
            out['rmano_hand_pose'] = rhand_pose
            out['lmano_shape'] = lshape
            out['rmano_shape'] = rshape
            out['lmano_joint'] = ljoint_proj
            out['rmano_joint'] = rjoint_proj
            if 'mano_joint_img' in targets:
                out['mano_joint_img'] = targets['mano_joint_img']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'mano_mesh_cam' in targets:
                out['mano_mesh_cam_target'] = targets['mano_mesh_cam']
            if 'do_flip' in meta_info:
                out['do_flip'] = meta_info['do_flip']
            return out

def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            nn.init.constant_(m.bias,0)
    except AttributeError:
        pass

def get_model(mode):
    backbone_net = BackboneNet()
    hand_feature_extra=Extra_hand_feature()
    prior_feat=Exact_Prior_feature()
    rotationNet=RotationNet()


    if mode == 'train':
        backbone_net.init_weights()
        hand_feature_extra.apply(init_weights)
        prior_feat.apply(init_weights)
        rotationNet.apply(init_weights)

    model = Model(backbone_net, hand_feature_extra,prior_feat, rotationNet)

    return model


def calculate_flops_and_params(model, input_size, modules=None):
    if modules is None:
        modules = [
            ('backbone_net', model.backbone_net),
            ('hand_feature_extra', model.hand_feature_extra),
            ('prior_feat', model.prior_feat),
            ('rotation_net', model.rotation_net)
        ]
    input_img = torch.randn(input_size).cuda()
    flops_list = []
    params_list = []

    for name, module in modules:
        if name == 'backbone_net':
            flops, params = profile(module, inputs=(input_img,), verbose=False)
            summary(module, input_data=(input_img,))
        elif name == 'hand_feature_extra':
            feat = model.backbone_net(input_img)
            flops, params = profile(module, inputs=(feat,), verbose=False)
            summary(module, input_data=(feat,))
        elif name == 'prior_feat':
            left_feat, right_feat, ljoint_img, rjoint_img = model.hand_feature_extra(feat)
            flops, params = profile(module, inputs=(left_feat, right_feat,), verbose=False)
            summary(module, input_data=(left_feat, right_feat,))
        elif name == 'rotation_net':
            l_proir, r_proir = model.prior_feat(left_feat, right_feat)
            flops, params = profile(module, inputs=(left_feat, right_feat, ljoint_img, rjoint_img, l_proir, r_proir,), verbose=False)
            summary(module, input_data=(left_feat, right_feat, ljoint_img, rjoint_img, l_proir, r_proir,))

        flops_list.append(flops)
        params_list.append(params)
        print(f'{name.upper()} FLOPs: {flops / 1e9} G')
        print(f'{name.upper()} Params: {params / 1e6} M')

    total_flops = sum(flops_list)
    total_params = sum(params_list)

    print(f'TOTAL FLOPs: {total_flops / 1e9} G')
    print(f'TOTAL Params: {total_params / 1e6} M')

    return total_flops, total_params

if __name__ == '__main__':
    model = get_model(mode='eval')
    model.cuda()
    model.eval()
    input_size = (1, 3, 256, 256)
    calculate_flops_and_params(model,input_size)
