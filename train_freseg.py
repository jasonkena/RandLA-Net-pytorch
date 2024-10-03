# Common
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from frenet import get_dataloader

import logging
import warnings
import argparse
import numpy as np
from tqdm import tqdm
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# my module
from util.config import ConfigFreseg
from util.metric import compute_acc, IoUCalculator
from network.RandLANet import Network
from network.loss_func import compute_loss
from dataset.semkitti_trainset import weird_collate

from functools import partial

def nested_to_device(nested, device):
    if isinstance(nested, dict):
        return {k: nested_to_device(v, device) for k, v in nested.items()}
    elif isinstance(nested, list):
        return [nested_to_device(v, device) for v in nested]
    else:
        assert isinstance(nested, torch.Tensor)
        return nested.to(device)

torch.backends.cudnn.enabled = False

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=5, help='Batch Size during training [default: 5]')
parser.add_argument('--val_batch_size', type=int, default=30, help='Batch Size during training [default: 30]')
parser.add_argument("--npoint", type=int, default=2048, metavar="N")
parser.add_argument("--path_length", type=int, help="path length")
parser.add_argument("--fold", type=int, help="fold")
parser.add_argument(
    "--frenet", action="store_true", help="whether to use Frenet transformation"
)
parser.add_argument("--num_workers", type=int, default=16, metavar="N")
FLAGS = parser.parse_args()
FLAGS.log_dir = os.path.join('log', f'{FLAGS.fold}_{FLAGS.path_length}_{FLAGS.npoint}_{FLAGS.frenet}')

cfg = ConfigFreseg(FLAGS.npoint)

class FakeDataset:
    def __init__(self, num_classes, ignored_labels):
        self.num_classes = num_classes
        self.ignored_labels = ignored_labels

class Trainer:
    def __init__(self):
        # Init Logging
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        self.log_dir = FLAGS.log_dir
        log_fname = os.path.join(FLAGS.log_dir, 'log_train.txt')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = logging.getLogger("Trainer")
        # tensorboard writer
        self.tf_writer = SummaryWriter(self.log_dir)
        # get_dataset & dataloader

        self.train_loader, _ = get_dataloader(
            species="seg_den",
            path_length=FLAGS.path_length,
            num_points=FLAGS.npoint,
            fold=FLAGS.fold,
            is_train=True,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers,
            frenet=FLAGS.frenet,
            collate_fn=partial(weird_collate, my_cfg=cfg),
        )
        self.val_loader, _ = get_dataloader(
            species="seg_den",
            path_length=FLAGS.path_length,
            num_points=FLAGS.npoint,
            fold=FLAGS.fold,
            is_train=False,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers,
            frenet=FLAGS.frenet,
            collate_fn=partial(weird_collate, my_cfg=cfg),
        )

        # Network & Optimizer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Network(cfg)
        self.net.to(device)

        # Load the Adam optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)

        # Load module
        self.highest_val_iou = 0
        self.start_epoch = 0
        CHECKPOINT_PATH = FLAGS.checkpoint_path
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']

        # Loss Function
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        # Multiple GPU Training
        if torch.cuda.device_count() > 1:
            self.logger.info("Let's use %d GPUs!" % (torch.cuda.device_count()))
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.net = nn.DataParallel(self.net)
        self.train_dataset = FakeDataset(cfg.num_classes, [])
        self.val_dataset = FakeDataset(cfg.num_classes, [])

    def train_one_epoch(self):
        self.net.train()  # set model to training mode
        tqdm_loader = tqdm(self.train_loader, total=len(self.train_loader))
        for batch_idx, batch_data in enumerate(tqdm_loader):
            batch_data = nested_to_device(batch_data, torch.device("cuda:0"))
            # move all batch data to device
            self.optimizer.zero_grad()
            # Forward pass
            torch.cuda.synchronize()
            end_points = self.net(batch_data)
            loss, end_points = compute_loss(end_points, self.train_dataset, self.criterion)
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

    def train(self):
        for epoch in range(self.start_epoch, FLAGS.max_epoch):
            self.cur_epoch = epoch
            self.logger.info('**** EPOCH %03d ****' % (epoch))

            self.train_one_epoch()
            self.logger.info('**** EVAL EPOCH %03d ****' % (epoch))
            mean_iou = self.validate()
            # Save best checkpoint
            if mean_iou > self.highest_val_iou:
                self.hightest_val_iou = mean_iou
                checkpoint_file = os.path.join(self.log_dir, 'checkpoint.tar')
                self.save_checkpoint(checkpoint_file)

    def validate(self):
        self.net.eval()  # set model to eval mode (for bn and dp)
        iou_calc = IoUCalculator(cfg)

        tqdm_loader = tqdm(self.val_loader, total=len(self.val_loader))
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm_loader):
                batch_data = nested_to_device(batch_data, torch.device("cuda:0"))
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(cfg.num_layers):
                            batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                    else:
                        batch_data[key] = batch_data[key].cuda(non_blocking=True)

                # Forward pass
                torch.cuda.synchronize()
                end_points = self.net(batch_data)

                loss, end_points = compute_loss(end_points, self.train_dataset, self.criterion)

                acc, end_points = compute_acc(end_points)
                iou_calc.add_data(end_points)

        mean_iou, iou_list = iou_calc.compute_iou()
        self.logger.info('mean IoU:{:.1f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        self.logger.info(s)
        return mean_iou

    def save_checkpoint(self, fname):
        save_dict = {
            'epoch': self.cur_epoch+1,  # after training one epoch, the start_epoch should be epoch+1
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        # with nn.DataParallel() the net is added as a submodule of DataParallel
        try:
            save_dict['model_state_dict'] = self.net.module.state_dict()
        except AttributeError:
            save_dict['model_state_dict'] = self.net.state_dict()
        torch.save(save_dict, fname)


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
