import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocusLiteNN(nn.Module):
    def __init__(self, dataset, num_channel):
        super(FocusLiteNN, self).__init__()
        self.num_channel = num_channel
        self.dataset = dataset
        if self.dataset.lower() == "deepfocus" or self.dataset.lower() == "bioimage64" or self.dataset.lower() == "focuspath64" or self.dataset.lower() == "tcga":
            self.conv = nn.Conv2d(3, self.num_channel, 7, stride=1, padding=0)
            self.maxpool = nn.MaxPool2d(kernel_size=58)
        elif self.dataset.lower() == "bioimage":
            self.conv = nn.Conv2d(3, self.num_channel, 7, stride=5, padding=1)
            self.maxpool = nn.MaxPool2d(kernel_size=47)
        else:
            self.conv = nn.Conv2d(3, self.num_channel, 7, stride=5, padding=1)
            self.maxpool = nn.MaxPool2d(kernel_size=47)


        if self.num_channel > 1:
            self.fc = nn.Conv2d(self.num_channel, 1, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        batch_size = x.size()[0]

        x = self.conv(x)
        x = -self.maxpool(-x)    # minpooling
        if self.num_channel > 1:
            x = self.fc(x)
        x = x.view(batch_size, -1)

        return x


class FocusLiteNNMinMax(nn.Module):
    def __init__(self, num_channel=1):
        super(FocusLiteNNMinMax, self).__init__()
        self.num_channel = num_channel
        # Original CNN Parameters (FocusPath)
        # self.conv = nn.Conv2d(3, self.num_channel, 7, stride=5, padding=1)
        self.conv = nn.Conv2d(3, self.num_channel, 7, stride=1, padding=1)
        self.fc = nn.Conv2d(self.num_channel * 2, 1, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        batch_size = x.size()[0]

        x = self.conv(x)
        pool_size = x.shape[2:4]
        x1 = -F.max_pool2d(-x, pool_size)  # minpooling
        x2 = F.max_pool2d(x, pool_size)       # maxpooling
        x = torch.cat((x1, x2), 1)
        x = self.fc(x)
        x = x.view(batch_size, -1)

        return x
