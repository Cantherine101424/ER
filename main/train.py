import os
import argparse
from main.config import cfg
from torch.utils.tensorboard import SummaryWriter
from common.base import Trainer,Tester
from common.nets.module import EarlyStopping
import torch.backends.cudnn as cudnn
import numpy as np
from main.test import test
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids',default='1')
    parser.add_argument('--continue', dest='continue_train', action='store_true',default=False)
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train)
    cudnn.benchmark = True

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    writer = SummaryWriter(cfg.tb_dir)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    best_val_loss = float('inf')

    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        epoch_loss = 0.0

        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()

            # from torchinfo import summary
            # from common.nets.module import BackboneNet, Extra_hand_feature, RotationNet, Exact_Prior_feature
            # from main.model import Model
            # moo = Model(BackboneNet, Extra_hand_feature, RotationNet, Exact_Prior_feature)
            # summary(moo, input_size=(inputs, targets, meta_info, 'train'))

            loss = trainer.model(inputs, targets, meta_info, 'train')
            loss = {k: loss[k].mean() for k in loss}

            # backward
            total_loss = sum(loss.values())
            total_loss.backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()

            epoch_loss += total_loss.item()

            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
            ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k, v in loss.items()]
            screen.append('total_loss: %.4f' % total_loss.detach())
            trainer.logger.info(' '.join(screen))

            for k, v in loss.items():
                writer.add_scalar(f'loss/{k}', v.detach(), epoch * trainer.itr_per_epoch + itr)
            writer.add_scalar('loss/total_loss', total_loss.detach(), epoch * trainer.itr_per_epoch + itr)

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        epoch_loss /= trainer.itr_per_epoch

        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)

        eval_result = test(epoch)
        val_loss = np.mean(eval_result['mpjpe_sh'] + eval_result['mpjpe_ih'])
        print('MPVPE for all hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'] + eval_result['mpvpe_ih'])))
        print('MPVPE for single hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'])))
        print('MPVPE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_ih'])))
        print('MPJPE for all hand sequences: %.2f mm' % (np.mean(eval_result['mpjpe_sh'] + eval_result['mpjpe_ih'])))
        print('MPJPE for single hand sequences: %.2f mm' % (np.mean(eval_result['mpjpe_sh'])))
        print('MPJPE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['mpjpe_ih'])))

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            trainer.save_best_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, is_best=is_best)

        prev_model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % (epoch - 1))
        if os.path.exists(prev_model_path):
            os.remove(prev_model_path)

        # Early stopping check
        if early_stopping.check_early_stop(epoch_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    writer.close()

if __name__ == "__main__":
    main()
