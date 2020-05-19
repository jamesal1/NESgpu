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
        permutation="both"
        activation=nn.Tanh
        layers = []
        for out_channels, kernel_size, stride in filters[:-1]:
            padding = math.ceil((kernel_size[0]-1)/2)
            layers += [modules.PermutedConv2d( in_channels, out_channels, kernel_size, directions, stride=stride,
                                    padding=padding, permutation=permutation)]
            layers += [activation()]
            layers += [nn.BatchNorm2d(out_channels)]
            in_channels = out_channels

        out_channels, kernel_size, stride = filters[-1]

        layers += [modules.PermutedConv2d(in_channels, out_channels, kernel_size, directions, stride, padding=0, permutation=permutation)]
        layers += [activation()]
        layers += [nn.BatchNorm2d(out_channels)]
        layers += [modules.PermutedConv2d(out_channels, action_size, [1,1], directions, stride, padding=0, permutation=permutation)]
        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return torch.max(self.layers.forward(input).squeeze(),dim=1)[1]




class DenseNet(nn.Module):


    def __init__(self, directions, action_size, in_channels=128,  big=True):
        super(DenseNet,self).__init__()
        layertype=modules.PerturbedLinear
        layers = []
        for i in range(3):
            layers += [layertype(128, 128,directions)]
            layers += [nn.ELU()]


        layers += [layertype(128, action_size,directions)]
        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return torch.max(self.layers.forward(input),dim=1)[1]
