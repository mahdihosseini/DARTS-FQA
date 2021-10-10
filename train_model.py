import argparse
import os
import time
from distutils.util import strtobool

from statistics import mean
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dartsfqa import Network
from dataset import FocusDataset
from plcc_loss import PLCCLoss

import genotypes

def parse_config():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--use_cuda", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--seed", type=int, default=0)

    # CNN architecture
    parser.add_argument("--arch", type=str, default='DARTS-FQA', help='options: DARTS-FQA, FocusLiteNN-1, FocusLiteNN-2, FocusLiteNN-10, EONSS, DenseNet13, ResNet18, ResNet50, ResNet101, MobileNetv2')

    #training dataset
    parser.add_argument('--trainset', type=str, default='focuspath', help='options: FocusPath, DeepFocus, BioImage, BioImage64')

    # hyperparameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    
    # For FocusLiteNN networks, the experiment uses an initial learning rate of 0.01 with a decay interval of 60 epochs. 
    # Other networks use an initial learning rate of 0.001 with a decay interval of 40 epochs.
    parser.add_argument("--initial_lr", type=float, default=0.001)
    parser.add_argument("--decay_interval", type=int, default=40)

    # The number of layers is limited to 1, 2, or 3 for this low-complexity application of DARTS.
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--auxiliary", action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--genotype', type=str, default='focuspath', help='which architecture to use')
    parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--init_channels', type=int, default=20, help='num of init channels')

    # utils
    parser.add_argument("--num_workers", type=int, default=0, help='num of threads to load data')
    parser.add_argument("--epochs_per_save", type=int, default=120)
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str, metavar='PATH', help='path to checkpoints')
    parser.add_argument('--board', default='./board', type=str, help='tensorboard log file path')
    parser.add_argument('--result_path', default='./train_results', type=str, metavar='PATH', help='path to print result .txt file and the training plots.')
    parser.add_argument('--training_path', default='./data/FocusPath/Training', type=str, metavar='PATH', help='path of the training dataset')
    parser.add_argument('--traincsv_path', default='./data/focuspath_training_metadata.csv')

    return parser.parse_args()


class Trainer(object):

    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.use_cuda = torch.cuda.is_available() and config.use_cuda

        # dataset
        if config.trainset.lower() == "focuspath64":
            self.train_csv = config.traincsv_path
            self.trainset = config.training_path
            self.results_file = "training_results_focuspath64"
            self.loss_file = "loss_plot_focuspath64"
            self.n_classes = 15
            self.train_transform = transforms.Compose(
                [transforms.RandomCrop(size=64), transforms.ToTensor(), transforms.Normalize(
                    mean=[x / 255.0 for x in [178.4304, 147.1354, 181.2755]], std=[x / 255.0 for x in [34.0320, 33.6027, 27.5137]])])
        elif config.trainset.lower() == "focuspath":
            self.train_csv = config.traincsv_path
            self.trainset = config.training_path
            self.results_file = "training_results_focuspath"
            self.loss_file = "loss_plot_focuspath"
            self.n_classes = 15
            self.train_transform = transforms.Compose(
                [transforms.RandomCrop(size=235), transforms.ToTensor(), transforms.Normalize(
                    mean=[x / 255.0 for x in [178.4304, 147.1354, 181.2755]], std=[x / 255.0 for x in [34.0320, 33.6027, 27.5137]])])
        elif config.trainset.lower() == "deepfocus":
            self.train_csv = config.traincsv_path
            self.trainset = config.training_path
            self.results_file = "training_results_deepfocus"
            self.loss_file = "loss_plot_deepfocus"
            self.n_classes = 11
            self.train_transform = transforms.Compose(
                [transforms.RandomCrop(size=64), transforms.ToTensor(), transforms.Normalize(
                    mean=[x / 255.0 for x in [164.9529, 151.1183, 169.3709]], std=[x / 255.0 for x in [28.3038, 27.5698, 24.8911]])])
        elif config.trainset.lower() == "bioimage":
            self.train_csv = config.traincsv_path
            self.trainset = config.training_path
            self.results_file = "training_results_bioimage"
            self.loss_file = "loss_plot_bioimage"
            self.n_classes = 11
            self.train_transform = transforms.Compose(
                [transforms.RandomCrop(size=235), transforms.ToTensor(), transforms.Normalize(
                    mean=[x / 255.0 for x in [150.5639, 150.5639, 150.5639]], std=[x / 255.0 for x in [37.7143, 37.7143, 37.7143]])])
        elif config.trainset.lower() == "bioimage64":
            self.train_csv = config.traincsv_path
            self.trainset = config.training_path
            self.results_file = "training_results_bioimage64"
            self.loss_file = "loss_plot_bioimage64"
            self.n_classes = 11
            self.train_transform = transforms.Compose(
                [transforms.RandomCrop(size=64), transforms.ToTensor(), transforms.Normalize(
                    mean=[x / 255.0 for x in [150.5639, 150.5639, 150.5639]], std=[x / 255.0 for x in [37.7143, 37.7143, 37.7143]])])

        self.train_batch_size = config.batch_size
        self.train_data = FocusDataset(csv_file=self.train_csv, root_dir=self.trainset, transform=self.train_transform, dataset=config.trainset)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=config.num_workers, drop_last=True)
        self.train_data_size = len(self.train_loader.dataset)
        self.num_steps_per_epoch = len(self.train_loader)
        self.initial_lr = config.initial_lr
        self.decay_interval = config.decay_interval
        if config.arch.lower() == "darts-fqa":
            self.genotype = eval("genotypes.%s" % config.genotype)

        # initialize the model
        if config.arch.lower() == "focuslitenn-1":
            from model.focuslitenn import FocusLiteNN
            self.model = FocusLiteNN(dataset=config.trainset, num_channel=1)
        elif config.arch.lower() == "focuslitenn-2":
            from model.focuslitenn import FocusLiteNN
            self.model = FocusLiteNN(dataset=config.trainset, num_channel=2)
        elif config.arch.lower() == "focuslitenn-10":
            from model.focuslitenn import FocusLiteNN
            self.model = FocusLiteNN(dataset=config.trainset, num_channel=10)
        elif config.arch.lower() == "eonss":
            from model.eonss import EONSS
            self.model = EONSS(dataset=config.trainset)
        elif config.arch.lower() in ["densenet13", "densenet"]:
            self.model = torchvision.models.DenseNet(block_config=(1, 1, 1, 1), num_classes=1)
        elif config.arch.lower() in ["resnet18", "resnet"]:
            from torchvision.models.resnet import BasicBlock
            self.model = torchvision.models.ResNet(block=BasicBlock, layers=[1, 1, 1, 1], num_classes=1)
        elif config.arch.lower() == "resnet50":
            self.model = torchvision.models.resnet50(num_classes=1)
        elif config.arch.lower() == "resnet101":
            self.model = torchvision.models.resnet101(num_classes=1)
        elif config.arch.lower() in ["mobilenet", "mobilenetv2"]:
            self.model = torchvision.models.mobilenet_v2(num_classes=1)
        elif config.arch.lower() in ["darts-fqa"]:
            self.model = Network(config.init_channels, self.n_classes, config.layers, config.auxiliary, self.genotype)
        else:
            raise NotImplementedError(f"[****] '{config.arch}' is not a valid architecture")
        self.model_name = type(self.model).__name__
        num_param = sum([p.numel() for p in self.model.parameters()])
        print(f"[*] Initializing model: {self.model_name}, num of params: {num_param}")

        if torch.cuda.device_count() > 1 and config.use_cuda:
            print("[*] GPU #", torch.cuda.device_count())
            if config.arch.lower() not in ["darts-fqa"]:
                self.model = nn.DataParallel(self.model)

        if self.use_cuda:
            self.model.cuda()

        self.crit_plcc = PLCCLoss()

        if self.use_cuda:
            self.crit_plcc = self.crit_plcc.cuda()

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.initial_lr)

        # lr scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)

        self.max_epochs = config.max_epochs
        self.epochs_per_save = config.epochs_per_save

        if config.arch.lower() not in ["darts-fqa"]:
            self.ckpt_path = os.path.join(config.ckpt_path, config.arch)
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            self.result_path = os.path.join(config.result_path, config.arch)
            if not os.path.exists(self.result_path):
                os.makedirs(self.result_path)
            self.writer = SummaryWriter(log_dir=os.path.join(config.board, config.arch))
        else:
            self.ckpt_path = os.path.join(config.ckpt_path, config.arch)
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            self.result_path = os.path.join(config.result_path, config.arch)
            if not os.path.exists(self.result_path):
                os.makedirs(self.result_path)


        self.arch = config.arch

    def fit(self, cfg):
        self.loss_plot = np.zeros((self.max_epochs, 2))
        for epoch in range(self.max_epochs):
            self.model.drop_path_prob = 0.0
            self._train_single_epoch(epoch, cfg)

        # Plotting the training loss
        plt.figure()
        plt.plot(self.loss_plot[:, 0], self.loss_plot[:, 1])
        plt.ylim((-1, 0))
        plt.title('Average Loss for each Epoch')
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss (PLCC)')
        save_str = self.loss_file + '.png'
        fig_path = os.path.join(self.result_path, save_str)
        plt.savefig(fig_path)
        plt.show()

    def _train_single_epoch(self, epoch, cfg):
        self.current_epoch = epoch
        local_counter = epoch * self.num_steps_per_epoch + 1
        start_time = time.perf_counter()
        self.train_loss_list = []
        self.train_loss = 0


        # start training
        for step, sample_batched in enumerate(self.train_loader, 0):
            images_batch, score_batch = sample_batched['image'], sample_batched['score']

            image = Variable(images_batch)  # shape: (batch_size, channel, H, W)
            score = Variable(score_batch.float())  # shape: (batch_size)

            if self.use_cuda:
                score = score.cuda()
                image = image.cuda()

            self.optimizer.zero_grad()
            q = self.model(image)

            # batch_size = int(q.nelement() / 1)
            q_avg = q[0].view(cfg.batch_size, 1).mean(1)  # shape: (batch_size)

            self.loss = -1 * self.crit_plcc(q_avg, score)

            self.loss.backward()
            self.optimizer.step()

            self.train_loss += self.loss.item()
            self.print_loss = self.train_loss/(step + 1)

            if self.arch.lower() == "eonss":
                if torch.cuda.device_count() > 1 and self.use_cuda:
                    self.model.module._gdn_param_proc()
                else:
                    self.model._gdn_param_proc()

            lr = self.optimizer.param_groups[0]['lr']
            if cfg.arch.lower() not in ["darts-fqa"]:
                self.writer.add_scalar('Train/TrainLoss', self.print_loss, local_counter)
                self.writer.add_scalar('lr', lr, local_counter)

            current_time = time.perf_counter()
            duration = current_time - start_time
            examples_per_sec = self.train_batch_size / duration

            format_str = '(E:%d, S:%d) [loss = %.4f, lr = %.6e] (%.1f samples/sec; %.3f sec/batch)'
            print_str = format_str % (epoch, step, self.print_loss, lr, examples_per_sec, duration)
            print(print_str)

            self.train_loss_list.append(self.print_loss)

            local_counter += 1
            start_time = time.perf_counter()

        self.scheduler.step()

        if (epoch + 1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'state_dict': self.model.state_dict(),
                'loss': self.loss.item(),
                'lr': lr
            }, model_name)

        self.summary_file = self.results_file + 'bs' + str(self.train_batch_size) + 'lr' + str(self.initial_lr) + 'di' + str(self.decay_interval) + '.txt'

        result_file = os.path.join(self.result_path, self.summary_file)
        train_result_file = open(result_file, 'a')
        format_out = '(E:%d) [average_loss = %.4f, lr = %.6e]'
        print_out = format_out % (epoch, mean(self.train_loss_list), lr)
        train_result_file.write(print_out + '\n')
        train_result_file.close()

        self.loss_plot[epoch, 0] = epoch
        self.loss_plot[epoch, 1] = mean(self.train_loss_list)



    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename):
        torch.save(state, filename)


if __name__ == "__main__":
    cfg = parse_config()
    t = Trainer(cfg)
    t.fit(cfg)
