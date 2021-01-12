from modules import base
# from modules import binary
import torch
from torch import nn
import math



class VGG11(nn.Module):


    def __init__(self, directions, action_size, binary_last_layer=True, in_channels=24, device="cuda"):
        super(VGG11, self).__init__()
        kernel_size = 3
        padding = 1
        stride = 1
        batch_norm = True
        layers = []
        # for out_channels in [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']:
        for out_channels in [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M']:
            if out_channels == "M":
                layers += [binary.BinarizedMaxPool2d()]
            else:
                layers += [binary.BinarizedConv2d(in_channels, out_channels, directions, kernel_size, batch_norm=batch_norm, padding=padding, device=device)]
                in_channels = out_channels
        layers += [nn.Flatten()]
        # for _ in range(2):
        for _ in range(0):
            layers += [binary.BinarizedLinear(512, 512, directions, batch_norm=batch_norm, device=device)]
        if binary_last_layer:
            layers += [binary.BinarizedLinear(512, action_size, directions, batch_norm=batch_norm, device=device)]
        else:
            layers += [binary.AsType(dtype=torch.float16)]
            layers += [base.PermutedLinear(512, action_size, directions, batch_norm=batch_norm).to(device).type(torch.float16)]
        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return self.layers.forward(input).squeeze().type(torch.float16)


class ResNet20(nn.Module):


    def __init__(self, directions, action_size, binary_last_layer=True, in_channels=24, device="cuda"):
        super(ResNet20, self).__init__()
        kernel_size = 3
        padding = 1
        stride = 1
        batch_norm = True
        layers = []
        # for out_channels in [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']:
        for out_channels in [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M']:
            if out_channels == "M":
                layers += [binary.BinarizedMaxPool2d()]
            else:
                layers += [binary.BinarizedConv2d(in_channels, out_channels, directions, kernel_size, batch_norm=batch_norm, padding=padding, device=device)]
                in_channels = out_channels
        layers += [nn.Flatten()]
        # for _ in range(2):
        for _ in range(0):
            layers += [binary.BinarizedLinear(512, 512, directions, batch_norm=batch_norm, device=device)]
        if binary_last_layer:
            layers += [binary.BinarizedLinear(512, action_size, directions, batch_norm=batch_norm, device=device)]
        else:
            layers += [binary.AsType(dtype=torch.float16)]
            layers += [base.PermutedLinear(512, action_size, directions, batch_norm=batch_norm).to(device).type(torch.float16)]
        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return self.layers.forward(input).squeeze().type(torch.float16)



class SmallNet(nn.Module):


    def __init__(self, directions, action_size, binary_last_layer=True, in_channels=24, device="cuda"):
        super(SmallNet, self).__init__()
        kernel_size = 3
        padding = 1
        stride = 1
        layers = []
        batch_norm = True
        # for out_channels in [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']:
        for out_channels in [64, 'M', 256, 'M']:
            if out_channels == "M":
                layers += [binary.BinarizedMaxPool2d()]
            else:
                layers += [binary.BinarizedConv2d(in_channels, out_channels, directions, kernel_size, batch_norm=batch_norm, padding=padding, device=device)]
                in_channels = out_channels
        layers += [nn.Flatten()]
        in_channels *= 8 ** 2
        if binary_last_layer:
            layers += [binary.BinarizedLinear(in_channels, action_size, directions, batch_norm=batch_norm, device=device)]
        else:
            layers += [binary.AsType(dtype=torch.float16)]
            layers += [base.PermutedLinear(in_channels, action_size, directions).to(device).type(torch.float16)]
        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return self.layers.forward(input).squeeze().type(torch.float16)


class Residual(nn.Module):

    def __init__(self, layers):
        nn.Module.__init__(self)
        self.layers = layers

    def forward(self, input):
        return input + self.layers.forward(input)


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


class ResNet9(nn.Module):

    def __init__(self, directions, action_size, binary_last_layer=True, in_channels=3, device="cuda"):
        super(ResNet9, self).__init__()
        kernel_size = 3
        padding = 1
        stride = 1
        batch_norm = True
        dropout = 0
        in_sparsity = 1/16
        width = 4
        options = {"combined": True, "allow_repeats":True}
        # layer_config = [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M']
        # layer_config = [64, 128, 'M', [128, 128], 256, 'M', 512, 'M', [512, 512], 'M']
        # layer_config = [64, 128, 'M', 128, 256, 256, 'M', 512, 1024, 'M', 1024, 2048]
        mult = [3, 4, 6, 3]
        layer_config = [64, 64, 'M', 128, [128, 128], 'M', 256, [256, 256], 'M', 512, [512, 512], 'M',]
        layer_config = [64 * width, 64 * width, 'M', 128 * width, [128 * width, 128 * width], 'M',
                        256 * width, [256 * width, 256 * width], 'M', 512 * width, [512 * width, 512 * width], 'M',]
        # layer_config = [64] + [[64] * 2] * mult[0] + ['M'] + [128] + [[128] * 2] * mult[1] +\
        #                 ['M'] + [256]+ [[256] * 2] * mult[2] + ['M'] + [512] + [[512] * 2] * mult[3] + ['A']
        layers = []

        seq, in_channels = sequence_builder(layer_config, in_channels, kernel_size, directions, padding, in_sparsity, options, batch_norm, dropout)
        layers += [seq]
        layers += [nn.Flatten()]
        # layers += [base.PermutedLinear(512, action_size, directions, in_sparsity=in_sparsity).to(device)]
        # layers += [base.SyntheticLinear(16 * in_channels, action_size, directions)]

        # layers += [base.SyntheticLinear(4 * in_channels, in_channels, directions)]
        # if batch_norm:
        #     layers += [nn.BatchNorm1d(in_channels, affine=False)]
        #     layers += [base.PerturbedAffine(in_channels, directions)]
        # layers += [nn.ELU(inplace=True)]
        # layers += [base.SyntheticLinear(in_channels, action_size, directions)]
        layers += [base.SyntheticLinear(4 * in_channels, action_size, directions)]
        # layers += [base.SyntheticLinear(in_channels, action_size, directions)]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers.forward(input).squeeze()