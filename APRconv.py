import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from torchstat import stat
from timm.models.layers import trunc_normal_


def rel_pos(kernel_size):
    tensors = [torch.linspace(-1, 1, steps=kernel_size) for _ in range(2)]
    kernel_coord = torch.stack(torch.meshgrid(*tensors), dim=-0)
    kernel_coord = kernel_coord.unsqueeze(0)
    return kernel_coord


class APR(nn.Module):
    def __init__(self, planes, kernel_size, n_points, stride, padding, groups):
        super().__init__()

        self.planes = planes
        self.kernel_size = kernel_size
        self.n_points = n_points
        self.init_radius = 2 * (2/kernel_size)
        self.stride, self.padding, self.groups = stride, padding, groups
        # kernel_coord
        kernel_coord = rel_pos(kernel_size)
        self.register_buffer('kernel_coord', kernel_coord)

        # weight_coord
        weight_coord = torch.empty(1, n_points, 2)
        nn.init.trunc_normal_(weight_coord, std=0.2, a=-1., b=1.)
        self.weight_coord = nn.Parameter(weight_coord)

        self.radius = nn.Parameter(torch.empty(1, n_points).unsqueeze(-1).unsqueeze(-1))
        self.radius.data.fill_(value=self.init_radius)

        # weight
        weights = torch.empty(1, planes, n_points)
        trunc_normal_(weights, std=.02)
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        kernels = self.make_kernels().unsqueeze(1)
        x = x.contiguous()
        kernels = kernels.contiguous()
        x = F.conv2d(x,kernels,stride=self.stride, padding= self.padding, groups= self.groups)
        return x        

    def make_kernels(self):
        diff = self.weight_coord.unsqueeze(-2) - self.kernel_coord.reshape(1,2,-1).transpose(1,2)  # [1, n_points, kernel_size^2, 2]
        diff = diff.transpose(2,3).reshape(1, self.n_points, 2, self.kernel_size, self.kernel_size)
        diff = F.relu(1 - torch.sum(torch.abs(diff), dim=2) / self.radius)  # [1, n_points, kernel_size, kernel_size]
        
        # Apply weighted diff for average weighted kernel
        # non_zero = (diff != 0) # [1, n_points, kernel_size, kernel_size]
        # count_weight = 1 / (torch.sum(non_zero, dim=1, keepdim=True) + 1e-6)  # [1, 1, kernel_size, kernel_size]
        # weighted_diff = count_weight * diff  # [1, n_points, kernel_size, kernel_size]

        kernels = torch.matmul(self.weights, diff.reshape(1, self.n_points, -1)) # [1, planes, kernel_size*kernel_size]
        kernels = kernels.reshape(1, self.planes, *self.kernel_coord.shape[2:]) # [1, planes, kernel_size, kernel_size]
        kernels = kernels.squeeze(0)
        kernels = torch.flip(kernels.permute(0,2,1), dims=(1,))
        return kernels
    
    def radius_clip(self, min_radius=1e-3, max_radius=1.):
        r = self.radius.data
        r = r.clamp(min_radius, max_radius)
        self.radius.data = r

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, n_points=None):
    print(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, n_points)
    if n_points != None and in_channels == out_channels and out_channels == groups and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        print("APR")
        return APR(in_channels, kernel_size, n_points, stride, padding, groups)
    else:
        print("Original convolution")
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, n_points=None):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, 
                                         n_points=n_points))
    result.add_module('bn', nn.BatchNorm2d(out_channels))
    return result

class APRConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups, n_points=None, n_points_divide=12):
        super().__init__()
        self.kernel_size = kernel_size
        if n_points == None:
            n_points = int((kernel_size**2) // n_points_divide)

        padding     = kernel_size // 2
        self.apc    = conv_bn(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=1, groups=groups, n_points=n_points)
        
        self.small_kernel = 3
        self.small_conv = conv_bn(in_channels=in_channels, out_channels=in_channels, kernel_size=self.small_kernel,
                                   stride=stride, padding=self.small_kernel//2, groups=groups)

        self.pw1 = conv_bn(in_channels, out_channels, 1, 1, 0, groups=1)

    def forward(self, inputs):
        out = self.apc(inputs)
        out = self.small_conv(inputs)
        out = self.pw1(out)
        return out
    
def computing_feature_map_size( dataset, stride_size, layers = 3  ):
    input_shape =  {
        'har': (512,6)
    }
    h,w = input_shape[dataset]
    for layer in range(layers):
        h = ((h-1))//stride_size[0] + 1
        w = ((w-1))//stride_size[1] + 1
    return h*w