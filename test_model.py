import argparse
import collections
import os
import re
import time
from collections import OrderedDict
from distutils.util import strtobool

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torchvision
from torch._six import int_classes, string_classes
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from dartsfqa import Network
from dataset import FocusDataset

import genotypes


def parse_config():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--use_cuda", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--seed", type=int, default=2020)

    # CNN architecture
    parser.add_argument("--arch", type=str, default='DARTS-FQA', help='options: DARTS-FQA, FocusLiteNN-1, FocusLiteNN-2, FocusLiteNN-10, EONSS, DenseNet13, ResNet18, ResNet50, ResNet101')
    parser.add_argument("--batch_size", type=int, default=1, help='adjust based on your GPU Memory')

    # The number of layers is limited to 1, 2, or 3 for this low-complexity application of DARTS.
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--auxiliary", action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--genotype', type=str, default='deepfocus', help='which architecture to use')
    parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--init_channels', type=int, default=20, help='num of init channels')

    parser.add_argument("--testing_set", type=str, default='TCGA', help='options: TCGA, FocusPath, DeepFocus, BioImage, BioImage64')
    parser.add_argument("--testing_path", type=str, default='./data/TCGA@Focus', metavar='PATH', help='Write the path at which the testing set is saved locally.')
    parser.add_argument("--testcsv_path", type=str, default='./data/TCGA@Focus/TCGA@Focus.txt', metavar='PATH', help='Write the path at which the testing set CSV is saved locally.')
    parser.add_argument("--results_path", type=str, default='./test_results/tcga', metavar='PATH', help='Write the path where testing results should be saved.')
    # checkpoint
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str, help='path to checkpoint')

    # utils
    parser.add_argument("--num_workers", type=int, default=0, help="num of threads to load data")

    return parser.parse_args()


class DenseSpatialCrop_collate(object):
    """Densely crop an image, where stride is equal to the output size.

    Args:
        output_size (tuple or int): Desired output size. If  int, square crop
            is made.
    """

    def __init__(self, output_size, stride, dataset):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        assert isinstance(stride, (int, tuple))
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            assert len(stride) == 2
            self.stride = stride

        self.dataset = dataset

    def __call__(self, image):
        w, h = image.size[:2]
        new_h, new_w = self.output_size
        stride_h, stride_w = self.stride

        if self.dataset.lower() == "deepfocus":
            h_start = np.arange(0, h, stride_h)
            w_start = np.arange(0, w, stride_w)
        else:
            h_start = np.arange(0, h - new_h/10, stride_h)
            w_start = np.arange(0, w - new_w/10, stride_w)

        patches = [image.crop((wv_s, hv_s, wv_s + new_w, hv_s + new_h)) for hv_s in h_start for wv_s in w_start]

        to_tensor = transforms.ToTensor()

        patches = [to_tensor(patch) for patch in patches]

        if self.dataset.lower() == "focuspath64":
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [177.4304, 147.1354, 181.2755]], std=[x / 255.0 for x in [34.0320, 33.6027, 27.5137]])
        elif self.dataset.lower() == "tcga":
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [168.3571, 125.3874, 158.8208]], std=[x / 255.0 for x in [46.6840, 52.8810, 42.0803]])
        elif self.dataset.lower() == "deepfocus":
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [164.9529, 151.1183, 169.3709]], std=[x / 255.0 for x in [28.3038, 27.5698, 24.8911]])
        elif self.dataset.lower() == "bioimage":
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [150.5639, 150.5639, 150.5639]], std=[x / 255.0 for x in [37.7143, 37.7143, 37.7143]])
        elif self.dataset.lower() == "bioimage64":
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [150.5639, 150.5639, 150.5639]], std=[x / 255.0 for x in [37.7143, 37.7143, 37.7143]])
        else:
            raise NotImplementedError(f"[****] {self.dataset} is not a recognized dataset.")

        patches = [normalize(patch) for patch in patches]
        patches = torch.stack(patches, dim=0)
        return patches


class Tester(object):

    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.use_cuda = torch.cuda.is_available() and config.use_cuda

        if cfg.testing_set.lower() == "focuspath64":
            cfg.test_csv = config.testcsv_path
            cfg.testset = config.testing_path
            cfg.results_file = "testing_results_focuspath.txt"
            cfg.results_path = config.results_path
            cfg.stride_val = 32
            cfg.test_transform = transforms.Compose(
                [DenseSpatialCrop_collate(output_size=64, stride=cfg.stride_val, dataset=cfg.testing_set)])
        elif cfg.testing_set.lower() == "tcga":
            cfg.test_csv = config.testcsv_path
            cfg.testset = config.testing_path
            cfg.results_file = "testing_results_tcga.txt"
            cfg.results_path = config.results_path
            cfg.stride_val = 32
            cfg.test_transform = transforms.Compose(
                [DenseSpatialCrop_collate(output_size=64, stride=cfg.stride_val, dataset=cfg.testing_set)])
        elif cfg.testing_set.lower() == "deepfocus":
            cfg.test_csv = config.testcsv_path
            cfg.testset = config.testing_path
            cfg.results_file = "testing_results_deepfocus.txt"
            cfg.results_path = config.results_path
            cfg.stride_val = 32
            cfg.test_transform = transforms.Compose(
                [DenseSpatialCrop_collate(output_size=64, stride=cfg.stride_val, dataset=cfg.testing_set)])
        elif cfg.testing_set.lower() == "bioimage":
            cfg.test_csv = config.testcsv_path
            cfg.testset = config.testing_path
            cfg.results_file = "testing_results_bioimage.txt"
            cfg.results_path = config.results_path
            cfg.stride_val = 128
            cfg.test_transform = transforms.Compose(
                [DenseSpatialCrop_collate(output_size=235, stride=cfg.stride_val, dataset=cfg.testing_set)])
        elif cfg.testing_set.lower() == "bioimage64":
            cfg.test_csv = config.testcsv_path
            cfg.testset = config.testing_path
            cfg.results_file = "testing_results_bioimage.txt"
            cfg.results_path = config.results_path
            cfg.stride_val = 32
            cfg.test_transform = transforms.Compose(
                [DenseSpatialCrop_collate(output_size=64, stride=cfg.stride_val,
                                            dataset=cfg.testing_set)])
        else:
            raise NotImplementedError(f"[****] {cfg.testing_set} is not a recognized dataset.")

        # pre-processing
        self.test_transform = config.test_transform
        self.arch = config.arch

        # initialize the model
        if config.arch.lower() == "focuslitenn-1":
            from models.focuslitenn import FocusLiteNN
            self.model = FocusLiteNN(dataset=config.testing_set, num_channel=1)
        elif config.arch.lower() == "focuslitenn-2":
            from models.focuslitenn import FocusLiteNN
            self.model = FocusLiteNN(dataset=config.testing_set, num_channel=2)    
        elif config.arch.lower() == "focuslitenn-10":
            from models.focuslitenn import FocusLiteNN
            self.model = FocusLiteNN(dataset=config.testing_set, num_channel=10)
        elif config.arch.lower() == "eonss":
            from models.eonss import EONSS
            self.model = EONSS(dataset=config.testing_set)
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
            if config.testing_set.lower() == 'deepfocus':
                self.n_classes = 6
            elif config.testing_set.lower() == 'focuspath64':
                self.n_classes = 15
            elif config.testing_set.lower() == 'tcga':
                self.n_classes = 6
            elif config.testing_set.lower() == 'bioimage64':
                self.n_classes = 11
            self.genotype = eval("genotypes.%s" % config.genotype)
            self.model = Network(config.init_channels, self.n_classes, config.layers, config.auxiliary, self.genotype)
            self.model.drop_path_prob = 0.0
        else:
            raise NotImplementedError(f"[****] '{config.arch}' is not a valid architecture")
        self.model_name = type(self.model).__name__
        num_param = sum([p.numel() for p in self.model.parameters()])
        print(f"[*] Initializing model: {self.model_name}, num of params: {num_param}")

        if torch.cuda.device_count() > 1 and config.use_cuda:
            print(f"[*] {torch.cuda.device_count()} GPU detected")
            if config.arch.lower() not in ["darts-fqa"]:
                self.model = nn.DataParallel(self.model)

        if self.use_cuda:
            self.model.cuda()
        else:
            self.device = torch.device("cpu")
            self.model.to(self.device)

        # load the pre-trained model
        if os.path.exists(config.ckpt_path):
            self._load_checkpoint(ckpt=config.ckpt_path)
        else:
            raise FileNotFoundError(f"[****] checkpoint file '{config.ckpt_path}' not found")

    def patch_collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        numpy_type_map = {
            'float64': torch.DoubleTensor,
            'float32': torch.FloatTensor,
            'float16': torch.HalfTensor,
            'int64': torch.LongTensor,
            'int32': torch.IntTensor,
            'int16': torch.ShortTensor,
            'int8': torch.CharTensor,
            'uint8': torch.ByteTensor,
        }
        error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
        elem_type = type(batch[0])
        if isinstance(batch[0], torch.Tensor):
            out = None
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
            tmp = torch.cat(batch, 0, out=out)
            return tmp
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))

                return torch.cat([torch.from_numpy(b) for b in batch], 0)
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
        elif isinstance(batch[0], int_classes):
            return torch.LongTensor(batch)
        elif isinstance(batch[0], float):
            return torch.DoubleTensor(batch)
        elif isinstance(batch[0], string_classes):
            return batch
        elif isinstance(batch[0], collections.Mapping):
            # return {key: dim1_collate([d[key] for d in batch]) for key in batch[0]}
            collated = {}
            collated["image"] = self.patch_collate([d["image"] for d in batch])
            collated["score"] = default_collate([d["score"] for d in batch])
            collated["image_name"] = default_collate([d["image_name"] for d in batch])
            collated["patch_num"] = default_collate([d["patch_num"] for d in batch])
            return collated

        raise TypeError((error_msg.format(type(batch[0]))))

    def _evaluateImage_denseCrop(self, test_config):
        if test_config is None:
            return None, None

        stats_dir = test_config['save_path']
        if not os.path.isdir(stats_dir):
            os.makedirs(stats_dir)

        self.test_data = FocusDataset(csv_file=test_config['input_csv'], root_dir=test_config['root_dir'], transform=self.test_transform, dataset=test_config['dataset_name'])
        self.test_loader = DataLoader(self.test_data,
                                      batch_size=test_config['test_batch_size'],
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=test_config['num_workers'],
                                      collate_fn=self.patch_collate)

        length = len(self.test_loader.dataset)
        print("%d images in this dataset" % length)
        image_name_list = []
        score_predict_list = np.zeros([length])
        score_list = np.zeros([length])
        fuzzy_label = 0
        batch_size = test_config['test_batch_size']

        for counter, sample_batched in enumerate(self.test_loader, 0):

            start_time = time.perf_counter()

            image_batch, score_batch, name_batch, patch_num_batch = sample_batched['image'], sample_batched['score'], sample_batched['image_name'], sample_batched['patch_num']

            image = Variable(image_batch)   # shape: (batch_size, channel, H, W)
            if self.use_cuda:
                image = image.cuda()
            else:
                image = image.to(self.device)
            score = score_batch.float().numpy()
            with torch.no_grad():
                score_predict = self.model(image)
                score_predict = score_predict[0].cpu().data.numpy()    # shape: (batch_size)

            patch_counter = 0
            for i in range(len(patch_num_batch)):
                score_predict_list[counter * batch_size + i] = np.mean(score_predict[patch_counter: patch_counter + patch_num_batch[i]])  # 1
                patch_counter += patch_num_batch[i]

            score_list[counter * batch_size: (counter + 1) * batch_size] = score
            image_name_list += name_batch

            stop_time = time.perf_counter()

            samples_per_sec = batch_size / (stop_time - start_time)

            if batch_size == 1:
                print(counter + 1, "/", length, name_batch[0], score[0], score_predict[0], '\tSamples/Sec',
                      samples_per_sec)
            else:
                print(batch_size, 1 + counter * batch_size, "/", length, '\tSamples/Sec', samples_per_sec)

            del score
            del score_predict

        test_result_file = os.path.join(stats_dir, test_config["name"] + '.txt')
        np.savetxt(test_result_file, np.column_stack([image_name_list, score_predict_list, score_list]), fmt="%s", delimiter=",")

        # Calculating Area Under the Curve (AUC) for ROC and PR.
        score_predict_list_classification = score_predict_list
        score_list_binary = (score_list > fuzzy_label).astype(np.float)

        # Out of Focus images get flagged as a 1. Change the fuzzy label to 2.
        fpr, tpr, thresholds = metrics.roc_curve(score_list_binary, score_predict_list_classification, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)

        # Plotting the ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.4f)' % roc_auc)
        plt.title('ROC Curve for %s' % test_config["dataset_name"])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        fig_path = os.path.join(stats_dir, 'roc_curve_%s.png' % test_config["dataset_name"])
        plt.savefig(fig_path)
        plt.show()

        precision, recall, thresholds = metrics.precision_recall_curve(score_list_binary, score_predict_list_classification, pos_label=1)
        pr_auc = metrics.auc(recall, precision)

        plt.figure()
        plt.plot(precision, recall, color='darkorange', lw=2, label='ROC curve (AUC = %0.4f)' % pr_auc)
        plt.title('PR Curve for %s' % test_config["dataset_name"])
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        fig_path = os.path.join(stats_dir, 'pr_curve_%s.png' % test_config["dataset_name"])
        plt.savefig(fig_path)
        plt.show()

        acc_list = []
        threshold_list = np.linspace(np.min(score_predict_list_classification), np.max(score_predict_list_classification), 100000)
        for i in threshold_list:
            acc = np.sum((score_predict_list_classification > i).astype(np.int) == score_list_binary) / len(score_list_binary)
            acc_list.append(acc)
        idx = np.argmax(np.array(acc_list))
        acc = acc_list[idx]

        if length >= 3:
            srcc = scipy.stats.mstats.spearmanr(x=score_list, y=score_predict_list)[0]
            plcc = scipy.stats.mstats.pearsonr(x=score_list, y=score_predict_list)[0]
        else:
            srcc = None
            plcc = None

        return srcc, plcc, roc_auc, pr_auc, acc

    def eval_test(self, *args):
        self.model.eval()
        results = {}
        for val_config in args:
            db_name = val_config["name"]
            print('\nEvaluating: {} database'.format(db_name))
            results[db_name] = list(self._evaluateImage_denseCrop(val_config))

        return results

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)

            # load checkpoint
            model_has_module = (list(self.model.state_dict().keys())[0].lower().find("module") != -1)
            checkpoint_has_module = (list(checkpoint['state_dict'].keys())[0].lower().find("module") != -1)
            if model_has_module and not checkpoint_has_module:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = "module." + k  # add `module.` in the state_dict which is saved with the "nn.DataParallel()"
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
            elif not model_has_module and checkpoint_has_module:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove `module.` in the state_dict which is saved with the "nn.DataParallel()"
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)

            print("[*] loaded checkpoint '{}'".format(ckpt))
        else:
            raise Exception("[!] no checkpoint found at '{}'".format(ckpt))


if __name__ == "__main__":
    cfg = parse_config()
    t = Tester(cfg)

    start_time = time.perf_counter()

    save_path = os.path.join(cfg.results_path, cfg.arch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_dataset = {
        "name": os.path.splitext(os.path.basename(cfg.test_csv))[0],
        "num_workers": cfg.num_workers,
        "root_dir": cfg.testset,
        "input_csv": cfg.test_csv,
        "save_path": save_path,
        "test_batch_size": cfg.batch_size,
        "dataset_name": cfg.testing_set,
        "stride_val": cfg.stride_val
    }

    test_results = t.eval_test(test_dataset)

    current_time = time.perf_counter()
    print("Total time: {:.4f}".format(current_time - start_time))

    for db_name in test_results:
        result = test_results[db_name]
        if result[0] is not None:
            out_str = '{}\tckpt: {}\tSRCC {:.7f}\tPLCC {:.7f}\tAUC-ROC {:.7f}\tAUC-PR {:.7f}\tAcc: {:.7f}'.format(db_name, cfg.ckpt_path, result[0], result[1], result[2], result[3], result[4])
        else:
            out_str = 'Dataset too small to calculate SRCC and PLCC'
        print(out_str)

        test_result_file = os.path.join(test_dataset["save_path"], cfg.results_file)
        test_result_file = open(test_result_file, 'a')
        test_result_file.write(out_str + '\n')
        test_result_file.close()
           