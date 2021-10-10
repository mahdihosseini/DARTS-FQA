import argparse
import math
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torchvision
import utils
from wsi_evaluation import wsi_evaluation
from PIL import Image
from dartsfqa import Network
import models.focuslitenn
import models.eonss
import genotypes

'''
    This program creates a heatmap of an entire Whole Slide Image using trained CNN options. 
'''

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="FocusLiteNN",
                        help="options: 'DARTS-FQA', 'FocusLiteNN-1', 'FocusLiteNN-2', 'FocusLiteNN-10', 'EONSS', 'DenseNet13', 'ResNet18', 'ResNet50', 'ResNet101', 'MobileNetv2'")
    parser.add_argument("--img", type=str, default="./whole_slide_images/example_1.svs", help='Path to SVS or TIFF image.')
    parser.add_argument("--trainset", type=str, default="FocusPath64", help="options: 'DeepFocus', 'FocusPath', 'BioImage'")
    parser.add_argument("--heatmap", type=bool, default=True, help='Value normalized to [0, 1]')
    parser.add_argument("--save_result", type=bool, default=True, help='Select whether or not to save the result')
    parser.add_argument("--use_cuda", type=bool, default=True, help='Select whether or not to use CUDA')
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint", metavar="PATH", help='Checkpoint path for network you wish to evaluate.')
    parser.add_argument("--genotype", type=str, default="focuspath", help="Name of the genotype for DARTS-FQA")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size for heatmap sampling.")
    parser.add_argument("--stride", type=int, default=32, help="Stride for heatmap sampling.")
    parser.add_argument("--result_path", type=str, default="./heatmap_outputs", help="Input path for saving heatmaps and heatmap information")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers for DARTS-FQA (1, 2, or 3)")
    parser.add_argument("--init_channels", type=int, default=20, help="Number of input channels for DARTS-FQA")
    # Boundary Coordinates for Heatmap
    parser.add_argument("--i_lower", type=int, default=1000, help="Lower bound of heatmap box on the y-axis")
    parser.add_argument("--i_upper", type=int, default=2000, help="Upper bound of heatmap box on the y-axis")
    parser.add_argument("--j_lower", type=int, default=1000, help="Lower bound of heatmap box on the x-axis")
    parser.add_argument("--j_upper", type=int, default=2000, help="Upper bound of heatmap box on the x-axis")
    return parser.parse_args()


def get_patches(image, output_size, stride):
    w, h = image.size[:2]
    new_h, new_w = output_size, output_size
    stride_h, stride_w = stride, stride

    h_start = np.arange(0, h - new_h, stride_h) # count from 0 to (image_size - 235) with a step of 128
    w_start = np.arange(0, w - new_w, stride_w)

    patches = [image.crop((wv_s, hv_s, wv_s + new_w, hv_s + new_h)) for hv_s in h_start for wv_s in w_start]

    to_tensor = torchvision.transforms.ToTensor()
    patches = [to_tensor(patch) for patch in patches]
    patches = torch.stack(patches, dim=0)
    return patches


class TestingSingle():
    def __init__(self, config):
        self.config = config
        self.use_cuda = torch.cuda.is_available() and self.config.use_cuda
        self.img = config.img

        # initialize the model
        if config.arch.lower() == "focuslitenn-1":
            from models.focuslitenn import FocusLiteNN
            self.model = FocusLiteNN(num_channel=1, dataset=self.config.trainset)
        elif config.arch.lower() == "focuslitenn-2":
            from models.focuslitenn import FocusLiteNN
            self.model = FocusLiteNN(num_channel=2, dataset=self.config.trainset)
        elif config.arch.lower() == "focuslitenn-10":
            from models.focuslitenn import FocusLiteNN
            self.model = FocusLiteNN(num_channel=10, dataset=self.config.trainset)
        elif config.arch.lower() == "eonss":
            from models.eonss import EONSS
            self.model = EONSS(dataset=self.config.trainset)
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
            if config.trainset.lower() == "focuspath64":
                genotype_name = "focuspath"
                num_classes = 15
            elif config.trainset.lower() == "deepfocus":
                genotype_name = "deepfocus"
                num_classes = 6
            elif config.trainset.lower() == "bioimage64":
                genotype_name = "bioimage"
                num_classes = 11
            genotype = eval("genotypes.%s" % genotype_name)
            self.model = Network(config.init_channels, num_classes, config.layers, False, genotype)
        else:
            raise NotImplementedError(f"[****] '{config.arch}' is not a valid architecture")
        self.model_name = type(self.model).__name__
        self.model.eval()

        if self.use_cuda:
            print("[*] Using GPU")
            # self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        else:
            print("[*] Using CPU")
            self.model.cpu()
        print("[*] Model %s initialized" % self.model_name)

        self._load_checkpoint(config.checkpoint_path)
        print("[*] Checkpoint %s loaded" % config.checkpoint_path)

    def eval(self):
        save_path = os.path.join(self.config.result_path, f"{self.config.trainset}\\{self.config.arch}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.img[-4:] == ".svs":
            t1 = time.time()
            self.model.drop_path_prob = 0.0
            image_scores, image_shape = wsi_evaluation(self.img, self.config.stride, self.config.patch_size, self.model, self.use_cuda, self.config)

            save_file = f"heatmap_{self.config.arch.lower()}.png"
            if self.config.heatmap:
                import matplotlib.pyplot as plt
                from skimage.transform import resize
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                new_h = math.floor(image_shape[0] / self.config.stride) + 1
                new_w = math.floor(image_shape[1] / self.config.stride) + 1
                heat_map = np.zeros((new_h, new_w))
                for patch in range(0, len(image_scores)):
                    i_idx = int(image_scores[patch, 0])
                    j_idx = int(image_scores[patch, 1])
                    heatmap_score = image_scores[patch, 2]
                    heat_map[i_idx, j_idx] = heatmap_score
                print("[*] Heatmap is produced")

                heat_map[heat_map == 0] = np.nan

                heat_map -= np.nanmin(heat_map)
                heat_map /= np.nanmax(heat_map)

                score_predict_mean = np.nanmean(heat_map)

                heatmap_interpolated = resize(heat_map, image_shape)

                fig, ax = plt.subplots(nrows=1, ncols=1)
                im = ax.imshow(heatmap_interpolated, cmap='jet', alpha=0.5, vmin=np.nanmin(heat_map), vmax=np.nanmax(heat_map))
                ax.axis('off')
                # divider = make_axes_locatable(ax)
                # cax1 = divider.append_axes("right", size="5%", pad=0.05)
                # cbar = fig.colorbar(im, cax=cax1)
                # cbar.ax.tick_params(labelsize=12)
                plt.savefig(os.path.join(save_path, save_file), bbox_inches='tight')
                print("[*] Heatmap is saved")

                t2 = time.time()

                print("[-] Image name:\t\t", self.img)
                print("[-] %s score:\t%f" % (self.config.arch, score_predict_mean))
                print("[-] Time consumed:\t %.4f s" % ((t2 - t1)))

                if self.config.save_result:
                    with open(os.path.join(save_path, "%s_result.txt" % self.model_name), 'a') as txt_file:
                        txt_file.write(
                            "Image name:\t\t" + str(self.img) + "\n" + "%s score:\t" % self.model_name + str(
                                score_predict_mean))

        else:
            if os.path.isfile(self.img):
                image = Image.open(self.img)
            else:
                raise Exception("[!] no image found at '{}'".format(self.img))

            t1 = time.time()

            stride = 128
            image_patches = get_patches(image, 235, stride)

            image_patches = torch.autograd.Variable(image_patches)
            if self.use_cuda:
                image_patches = image_patches.cuda()

            if self.config.heatmap:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                from skimage.transform import resize
                import matplotlib.pyplot as plt

                # Converts image to grayscale
                original_image = np.asarray(image.convert('L'))

                # Starts making patch crops of the image, reduce image size accordingly
                original_h, original_w = original_image.shape[0], original_image.shape[1]
                new_h = math.floor((original_h - self.config.patch_size) / stride) + 1
                new_w = math.floor((original_w - self.config.patch_size) / stride) + 1

                num_patches = int(new_w * new_h)
                heatmap = np.zeros(num_patches)
                for i in range(num_patches):
                    heatmap[i] = torch.squeeze(self.model(image_patches[i][None, :, :, :]).cpu().data).numpy()
                score_predict_mean = np.mean(heatmap)
                heatmap = heatmap.reshape([new_h, new_w])

                # normalize
                heatmap -= heatmap.min()
                heatmap /= heatmap.max()

                # interpolate
                heatmap_interpolated = resize(heatmap, (original_h, original_w))

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=200)
                ax.imshow(original_image, cmap='gray')
                im = ax.imshow(heatmap_interpolated, cmap='jet', alpha=0.2, vmin=0.0, vmax=1.0)
                ax.axis('off')
                divider = make_axes_locatable(ax)
                cax1 = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax1)
                cbar.ax.tick_params(labelsize=12)
                plt.savefig("heat_map_results/" + self.config.arch + "/heatmap_" + self.config.arch + ".png", bbox_inches='tight', dpi='figure')

            else:

                score_predict = self.model(image_patches).cpu().data
                score_predict = torch.squeeze(score_predict, dim=1).numpy()
                score_predict_mean = np.mean(score_predict)

            t2 = time.time()

            print("[-] Image name:\t\t", self.img)
            print("[-] %s score:\t%f" % (self.config.arch, score_predict_mean))
            print("[-] Time consumed:\t %.4f s" % ((t2 - t1)))

            if self.config.save_result:
                with open("heat_map_results/%s/%s_result.txt" % (self.config.arch, self.model_name), 'w') as txt_file:
                    txt_file.write("Image name:\t\t" + str(self.img) + "\n" + "%s score:\t" % self.model_name + str(score_predict_mean))

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            if not torch.cuda.is_available():
                checkpoint = torch.load(ckpt, map_location='cpu')
            else:
                checkpoint = torch.load(ckpt)
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
                self.model.load_state_dict(checkpoint['state_dict'])
        else:
            raise Exception("[!] no checkpoint found at '{}'".format(ckpt))

if __name__ == "__main__":
    cfg = parse_config()
    t = TestingSingle(cfg)
    t.eval()
