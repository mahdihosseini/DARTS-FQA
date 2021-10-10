import os
import warnings

import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

class FocusDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform, dataset):
        self.frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

        if csv_file.find("FocusPath") != -1:
            for idx in range(len(self.frame)):
                self.frame.iloc[idx, 0] = self.frame.iloc[idx, 0][:-4] + ".png"
        elif csv_file.find("testing_data") != -1:
            for idx in range(len(self.frame)):
                self.frame.iloc[idx, 0] = self.frame.iloc[idx, 0][:-4] + ".png"
        elif csv_file.find("Deepfocus") != -1:
            pass
        elif csv_file.find("Bioimage") != -1:
            pass
        elif csv_file.find("TCGA") != -1:
            pass
        else:
            raise Exception(f"Cannot handle dataset {root_dir}")

        self.dataset = dataset

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        image = Image.open(img_name)
        rgb_image = image

        if self.dataset == "BioImage" or self.dataset == "BioImage64":
            grayscale_image = np.array(image)
            rgb_image = np.repeat(grayscale_image[..., np.newaxis], 3, -1)
            rgb_image = Image.fromarray(rgb_image, 'RGB')

        rgb_image = self.transform(rgb_image)

        patch_num = rgb_image.shape[0]

        if self.dataset == 'DeepFocus':
            score = abs(int(self.frame.iloc[idx, -1]) - 5)
        else:
            score = abs(int(self.frame.iloc[idx, -1]))

        image.close()

        sample = {'image': rgb_image, 'score': score, 'image_name': img_name, 'patch_num': rgb_image.shape[0]}

        num_train = len(self.frame)

        return sample