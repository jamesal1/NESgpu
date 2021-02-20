from modules import base
# from modules import binary
import torch
from torch import nn
import math

def sequence_builder(li, in_channels, kernel_size, directions, padding, in_sparsity, options, batch_norm, dropout):
    layers = []
    for out_channels in li:
        if isinstance(out_channels, list):
            seq, in_channels = sequence_builder(out_channels, in_channels, kernel_size, directions, padding, in_sparsity, options, batch_norm, dropout)
            layers += [Residual(seq)]
        elif out_channels == "M":
            layers += [nn.MaxPool2d(2)]
        elif out_channels == "A":
            layers += [nn.AvgPool2d(2)]
        else:
            layers += [base.PermutedConv2d(in_channels, out_channels, kernel_size, directions,
                                           padding=padding, in_sparsity=in_sparsity, options=options)]
            if batch_norm:
                layers += [nn.BatchNorm2d(out_channels, affine=False)]
                layers += [base.PerturbedAffine((out_channels, 1, 1), directions)]
                # layers += [nn.GroupNorm(1,out_channels, affine=False), base.PerturbedAffine((out_channels, 1, 1), directions)]
                # layers += [base.PerturbedAffine((out_channels, 1, 1), directions)]
            # layers += [nn.ReLU(inplace=True)]
            # layers += [nn.Tanh(inplace=True)]
            layers += [nn.ELU(inplace=True)]
            # layers += [nn.ELU(inplace=True), base.PerturbedAffine((out_channels, 1, 1), directions)]
            if dropout > 0:
                layers+= [nn.Dropout2d(dropout)]
            in_channels = out_channels
    return nn.Sequential(*layers), in_channels


class MNISTConvNet(nn.Module):

    def __init__(self, directions, action_size, in_channels=1, device="cuda"):
        super(MNISTConvNet, self).__init__()
        kernel_size = 5
        padding = 2
        stride = 1
        batch_norm = False
        dropout = 0
        in_sparsity = 1/16
        options = {"combined": True, "allow_repeats":True}
        layer_config = [32, "M", 64, 'M']
        layers = []

        seq, in_channels = sequence_builder(layer_config, in_channels, kernel_size, directions, padding, in_sparsity, options, batch_norm, dropout)
        layers += [seq]
        layers += [nn.Flatten()]
        layers += [base.SyntheticLinear(in_channels * 49, 1024, directions)]
        layers += [base.SyntheticLinear(1024, action_size, directions)]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers.forward(input).squeeze()

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