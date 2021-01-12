from modules import base
# from modules import binary
import torch
from torch import nn
import math

filters_mnist = [
    [16, [4, 4], 2],
    [32, [4, 4], 2],
    [256, [8, 8], 1],
]

class MNISTConvNet(nn.Module):


    def __init__(self, directions, action_size, in_channels=1):
        super(MNISTConvNet,self).__init__()
        filters = filters_mnist
        kwargs = {"permutation": "both", "in_sparsity": .01, "out_sparsity": .01}
        kwargs = {"permutation": "out", "out_sparsity": .5}
        # kwargs = {"permutation": "in", "in_sparsity": .5}
        layertype= base.PermutedConv2d

        # kwargs = {}
        # layertype = modules.PerturbedConv2d

        layers = []
        for out_channels, kernel_size, stride in filters[:-1]:
            padding = math.ceil((kernel_size[0]-1)/2)
            layers += [layertype(in_channels, out_channels, kernel_size, directions, stride=stride,
                                 padding=padding, **kwargs)]
            layers += [nn.Tanh()]
            in_channels = out_channels

        out_channels, kernel_size, stride = filters[-1]

        layers += [layertype(in_channels, out_channels, kernel_size, directions, stride, padding=0, **kwargs)]
        layers += [nn.Tanh()]
        layers += [layertype(out_channels, action_size, [1,1], directions, stride, padding=0, **kwargs)]
        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return torch.log_softmax(self.layers.forward(input).squeeze(),dim=1)


class MNISTDenseNet(nn.Module):


    def __init__(self, directions, action_size, in_channels=1):
        super(MNISTDenseNet,self).__init__()
        layertype= base.PermutedLinear
        kwargs = {"permutation": "in", "in_sparsity": .1}
        kwargs = {"permutation": "in"}
        layertype= base.SyntheticLinear
        kwargs = {"flip": "in"}

        # layertype=modules.PerturbedLinear
        # kwargs = {}
        layers = []
        for i in range(3):
            layers += [layertype(784, 784, directions, **kwargs)]
            layers += [nn.ELU()]


        layers += [layertype(784, action_size,directions, **kwargs)]
        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return self.layers.forward(input.reshape(-1,784)).squeeze()
        # return torch.log_softmax(self.layers.forward(input.reshape(-1,784)).squeeze(),dim=1)


class MNISTBinaryDenseNet(nn.Module):

    def __init__(self, directions, action_size, in_channels=1):
        super(MNISTBinaryDenseNet,self).__init__()
        layers = []
        layers += [binary.BinarizedLinear(784 * 8, 784, directions, dtype=torch.int8)]
        for i in range(2):
            layers += [binary.BinarizedLinear(784, 784, directions)]
        layers += [binary.BinarizedLinear(784, action_size, directions)]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return torch.log_softmax(self.layers.forward(input.view(-1, 784)).float(), dim=1)