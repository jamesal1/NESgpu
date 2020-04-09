import modules
import torch
from torch import nn
import torch.nn.functional as F
import math
filters_84x84 = [
    [16, [8, 8], 4],
    [32, [4, 4], 2],
    [256, [11, 11], 1],
]
filters_42x42 = [
    [16, [4, 4], 2],
    [32, [4, 4], 2],
    [256, [11, 11], 1],
]

#same architecture as rllib
class ConvNet(nn.Module):


    def __init__(self, directions, action_size, in_channels=3,  big=True):
        super(ConvNet,self).__init__()
        filters = filters_84x84 if big else filters_42x42
        layers = []
        for out_channels, kernel_size, stride in filters[:-1]:
            padding = math.ceil((kernel_size[0]-1)/2)
            layers += [modules.PerturbedConv2d( in_channels, out_channels, kernel_size, directions, stride=stride,
                                    padding=padding)]
            layers += [nn.ELU()]
            in_channels = out_channels

        out_channels, kernel_size, stride = filters[-1]

        layers += [modules.PerturbedConv2d(in_channels, out_channels, kernel_size, directions, stride, padding=0)]
        layers += [nn.ELU()]
        layers += [modules.PerturbedConv2d(out_channels, action_size, [1,1], directions, stride, padding=0)]
        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return torch.max(self.layers.forward(input).squeeze(),dim=1)[1]
