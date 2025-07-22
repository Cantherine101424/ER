import os
import os.path as osp
import math
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms

from common.timer import Timer
from common.logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from main.config import cfg
import re
from main.model import get_model
from data.dataset import MultipleDatasets

# dynamic dataset import
for i in range(len(cfg.trainset_3d)):
    exec('from ' + cfg.trainset_3d[i] + ' import ' + cfg.trainset_3d[i])
for i in range(len(cfg.trainset_2d)):
    exec('from ' + cfg.trainset_2d[i] + ' import ' + cfg.trainset_2d[i])
exec('from ' + cfg.testset + ' import ' + cfg.testset)


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name='train_logs.txt')

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        return optimizer

    def set_lr(self, epoch):
        if len(cfg.lr_dec_epoch) == 0:
            return cfg.lr

        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr


    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")
        trainset3d_loader = []
        for i in range(len(cfg.trainset_3d)):
            trainset3d_loader.append(eval(cfg.trainset_3d[i])(transforms.ToTensor(), "train"))
        trainset2d_loader = []
        for i in range(len(cfg.trainset_2d)):
            trainset2d_loader.append(eval(cfg.trainset_2d[i])(transforms.ToTensor(), "train"))

        valid_loader_num = 0
        if len(trainset3d_loader) > 0:
            trainset3d_loader = [MultipleDatasets(trainset3d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset3d_loader = []

        if len(trainset2d_loader) > 0:
            trainset2d_loader = [MultipleDatasets(trainset2d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset2d_loader = []

        if valid_loader_num > 1:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=True)
        else:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=False)

        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus * cfg.train_batch_size,
                                          shuffle=True, num_workers=cfg.num_thread, pin_memory=True, drop_last=True)


    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model('train')
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

    def save_best_model(self, state,  is_best=False):
        if is_best:
            best_file_path = osp.join(cfg.model_dir, 'best_model.pth.tar')
            torch.save(state, best_file_path)
            self.logger.info("Write best snapshot into {}".format(best_file_path))
        else:
            self.logger.info("save_best_model is wrong")

    def save_model(self, state, epoch, is_best=False):
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

        if is_best:
            best_file_path = osp.join(cfg.model_dir, 'best_model.pth.tar')
            torch.save(state, best_file_path)
            self.logger.info("Write best snapshot into {}".format(best_file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pth.tar'))

        # filter files like snapshot_.pth.tar
        snapshot_files = [file_name for file_name in model_file_list if re.search(r'snapshot_(\d+).pth.tar', file_name)]

        # extract epoch num
        if not snapshot_files:
            raise ValueError("No snapshot files found in the directory.")

        cur_epoch = max([int(re.search(r'snapshot_(\d+).pth.tar', file_name).group(1)) for file_name in snapshot_files])

        model_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        start_epoch = ckpt['epoch'] + 1

        model.load_state_dict(ckpt['network'])
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except Exception as e:
            self.logger.warning(f"Failed to load optimizer state: {e}")

        return start_epoch, model, optimizer


class Tester(Base):
    def __init__(self, test_epoch):
        self.test_epoch = test_epoch
        super(Tester, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(cfg.testset)(transforms.ToTensor(), "test")
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                     shuffle=False, num_workers=cfg.num_thread, pin_memory=True)

        self.testset = testset_loader
        self.batch_generator = batch_generator

    def _make_model(self):
        if self.test_epoch == 'best_model':
            model_path = os.path.join(cfg.model_dir, 'best_model.pth.tar')
        else:
            model_path = os.path.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(self.test_epoch))

        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model('test')
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()
        self.model = model

    def _evaluate(self, preds, cur_sample_idx):
        eval_result=self.testset.evaluate(preds,cur_sample_idx)
        return eval_result

    def _sub_evaluate(self, preds):
        value=self.testset.evaluate(preds)
        return value

    def _print_eval_result(self, eval_result):
        eval_result=self.testset.print_eval_result(eval_result)
        return eval_result


