import os
import os.path as osp
import sys


class Config:
    ## dataset
    vis = False
    debug = False
    ## dataset
    trainset_3d = ['InterHand26M']
    trainset_2d = []
    testset = 'InterHand26M'

    ## input, output
    input_img_shape = (256, 256)
    input_hm_shape = (64, 64, 64)
    output_hm_shape = (8, 8, 8)
    sigma = 2.5
    bbox_3d_size = 400  # depth axis
    bbox_3d_size_root = 400  # depth axis
    output_root_hm_shape = 64  # depth axis

    ## model
    resnet_type = 50  # 18, 34, 50, 101, 152

    ## training

    # lr_dec_epoch = [17, 19, 27, 29] # if dataset == 'InterHand2.6M' else [45, 47]
    # end_epoch = 30 # if dataset == 'InterHand2.6M' else 50
    # lr_dec_epoch = [21, 24]  # if dataset == 'InterHand2.6M' else [45, 47]
    # end_epoch = 25  # if dataset == 'InterHand2.6M' else 50
    lr_dec_epoch = [15, 17]  # if dataset == 'InterHand2.6M' else [45, 47]
    end_epoch = 30  # if dataset == 'InterHand2.6M' else 50
    lr = 1e-4
    lr_dec_factor = 10
    train_batch_size = 80
    train_smple = 36635
    #train:366358

    ## testing
    test_batch_size = 80
    # test_batch_size = 80
    test_smple = 26149
    # test:261494
    test_default_epoch='SA_29'
    trans_test = 'rootnet'  # gt, rootnet

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    # model_dir = osp.join(output_dir, 'model_dump')
    model_dir = osp.join(output_dir)
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    tb_dir = osp.join(log_dir, 'tb_log')
    train_log_dir=osp.join(output_dir,'train_log')
    result_dir = osp.join(output_dir, 'result')
    valdbs_tb_log = osp.join(log_dir, 'valdbs_tb_log')

    ## others
    num_thread = 40
    gpu_ids = '1'
    num_gpus = 1
    continue_train = False
    human_model_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))


cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from common.utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))

for i in range(len(cfg.trainset_3d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d[i]))
for i in range(len(cfg.trainset_2d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)

