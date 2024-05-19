import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from torchstat import stat
from timm.models.layers import trunc_normal_
from APRconv import  APRConv


class APR_Net(nn.Module):
    def __init__(self, train_shape, category):
        super(APR_Net, self).__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
        '''
        self.layer = nn.Sequential(

            APRConv(1,64,31,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            APRConv(64,128,31,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            APRConv(128,256,31,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, train_shape[-1]))
        self.fc = nn.Linear(256*train_shape[-1], category)

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]22
        '''
        x = self.layer(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x