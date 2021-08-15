import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict


def create_model(opt):
    k = opt.widen_factor
    stages = torch.tensor({16, 16 * k, 32 * k, 64 * k})


class WRN(nn.Module):
    def __init__(self):
        super(WRN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 1), padding=1, bias=False)