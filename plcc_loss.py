import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class PLCCLoss(nn.Module):

    def __init__(self):
        super(PLCCLoss, self).__init__()

    def forward(self, input, target):
        # Pearson's Correlation Coefficient Implementation
        input0 = input - torch.mean(input)
        target0 = target - torch.mean(target)

        self.loss = torch.sum(input0 * target0) / (torch.sqrt(torch.sum(input0 ** 2)) * torch.sqrt(torch.sum(target0 ** 2)))

        return self.loss
