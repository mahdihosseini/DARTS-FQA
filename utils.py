import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable

class EVLocalAvg(object):
    def __init__(self, args, window=5):
        """ Keep track of the eigenvalues local average.

        Args:
            window (int): number of elements used to compute local average.
                Default: 5
            ev_freq (int): frequency used to compute eigenvalues. Default:
                every 2 epochs
            total_epochs (int): total number of epochs that DARTS runs.
                Default: 50

        """
        self.window = window
        self.ev_freq = args.report_freq_hessian
        self.epochs = args.epochs

        self.stop_search = False
        self.stop_epoch = args.epochs - 1
        self.stop_genotype = None

        self.ev = []
        self.ev_local_avg = []
        self.genotypes = {}
        self.la_epochs = {}

        # start and end index of the local average window
        self.la_start_idx = 0
        self.la_end_idx = self.window

    def reset(self):
        self.ev = []
        self.ev_local_avg = []
        self.genotypes = {}
        self.la_epochs = {}

    def update(self, epoch, ev, genotype):
        """ Method to update the local average list.

        Args:
            epoch (int): current epoch
            ev (float): current dominant eigenvalue
            genotype (namedtuple): current genotype

        """
        self.ev.append(ev)
        self.genotypes.update({epoch: genotype})
        # set the stop_genotype to the current genotype in case the early stop
        # procedure decides not to early stop
        self.stop_genotype = genotype

        # since the local average computation starts after the dominant
        # eigenvalue in the first epoch is already computed we have to wait
        # at least until we have 3 eigenvalues in the list.
        if (len(self.ev) >= int(np.ceil(self.window/2))) and (epoch <
                                                              self.epochs - 1):
            # start sliding the window as soon as the number of eigenvalues in
            # the list becomes equal to the window size
            if len(self.ev) < self.window:
                self.ev_local_avg.append(np.mean(self.ev))
            else:
                assert len(self.ev[self.la_start_idx: self.la_end_idx]) == self.window
                self.ev_local_avg.append(np.mean(self.ev[self.la_start_idx:
                                                         self.la_end_idx]))
                self.la_start_idx += 1
                self.la_end_idx += 1

            # keep track of the offset between the current epoch and the epoch
            # corresponding to the local average. NOTE: in the end the size of
            # self.ev and self.ev_local_avg should be equal
            self.la_epochs.update({epoch: int(epoch -
                                              int(self.ev_freq*np.floor(self.window/2)))})

        elif len(self.ev) < int(np.ceil(self.window/2)):
          self.la_epochs.update({epoch: -1})

        # since there is an offset between the current epoch and the local
        # average epoch, loop in the last epoch to compute the local average of
        # these number of elements: window, window - 1, window - 2, ..., ceil(window/2)
        elif epoch == self.epochs - 1:
            for i in range(int(np.ceil(self.window/2))):
                assert len(self.ev[self.la_start_idx: self.la_end_idx]) == self.window - i
                self.ev_local_avg.append(np.mean(self.ev[self.la_start_idx:
                                                         self.la_end_idx + 1]))
                self.la_start_idx += 1

    def early_stop(self, epoch, factor=1.3, es_start_epoch=20, delta=2):
        """ Early stopping criterion

        Args:
            epoch (int): current epoch
            factor (float): threshold factor for the ration between the current
                and prefious eigenvalue. Default: 1.3
            es_start_epoch (int): until this epoch do not consider early
                stopping. Default: 20
            delta (int): factor influencing which previous local average we
                consider for early stopping. Default: 2
        """
        if int(self.la_epochs[epoch] - self.ev_freq*delta) >= es_start_epoch:
            # the current local average corresponds to
            # epoch - int(self.ev_freq*np.floor(self.window/2))
            current_la = self.ev_local_avg[-1]
            # by default take the local average corresponding to epoch
            # delta*self.ev_freq
            previous_la = self.ev_local_avg[-1 - delta]

            self.stop_search = current_la / previous_la > factor
            if self.stop_search:
                self.stop_epoch = int(self.la_epochs[epoch] - self.ev_freq*delta)
                self.stop_genotype = self.genotypes[self.stop_epoch]


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform

def _data_transforms_focuspath(args):
    FOCUSPATH_MEAN = [x / 255.0 for x in [178.4304, 147.1354, 181.2755]]
    FOCUSPATH_STD = [x / 255.0 for x in [34.0320, 33.6027, 27.5137]]

    train_transform = transforms.Compose([
        transforms.RandomCrop(size=64),
        transforms.ToTensor(),
        transforms.Normalize(mean=FOCUSPATH_MEAN, std=FOCUSPATH_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(FOCUSPATH_MEAN, FOCUSPATH_STD),
    ])

    return train_transform, valid_transform

def _data_transforms_deepfocus(args):
    DEEPFOCUS_MEAN = [x / 255.0 for x in [164.9529, 151.1183, 169.3709]]
    DEEPFOCUS_STD = [x / 255.0 for x in [28.3038, 27.5698, 24.8911]]

    train_transform = transforms.Compose([
        transforms.RandomCrop(size=64),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEEPFOCUS_MEAN, std=DEEPFOCUS_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.RandomCrop(size=64),
        transforms.ToTensor(),
        transforms.Normalize(DEEPFOCUS_MEAN, DEEPFOCUS_STD),
    ])

    return train_transform, valid_transform

def _data_transforms_bioimage(args):
    BIOIMAGE_MEAN = [x / 255.0 for x in [150.5639, 150.5639, 150.5639]]
    BIOIMAGE_STD = [x / 255.0 for x in [37.7143, 37.7143, 37.7143]]

    train_transform = transforms.Compose([
        transforms.RandomCrop(size=64),
        transforms.ToTensor(),
        transforms.Normalize(mean=BIOIMAGE_MEAN, std=BIOIMAGE_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(BIOIMAGE_MEAN, BIOIMAGE_STD),
    ])

    return train_transform, valid_transform

def _data_transforms_tcga(args):
    TCGA_MEAN = [x / 255.0 for x in [168.3571, 125.3874, 158.8208]]
    TCGA_STD = [x / 255.0 for x in [46.6840, 52.8810, 42.0803]]

    train_transform = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(TCGA_MEAN, TCGA_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(TCGA_MEAN, TCGA_STD),
    ])

    return train_transform, valid_transform
    
"""From https://github.com/chenxin061/pdarts/"""
def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

