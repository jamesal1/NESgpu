from modules import base, binary
import torch
from torch import nn
import math

class AsType(nn.Module):

    def __init__(self,dtype):
        super(AsType, self).__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.type(self.dtype)

class VGG11(nn.Module):


    def __init__(self, directions, action_size, binary_last_layer=False, in_channels=24):
        super(VGG11, self).__init__()
        kernel_size = 3
        padding = 1
        stride = 1
        layers = []
        for out_channels in [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']:
            if out_channels == "M":
                layers += [binary.BinarizedMaxPool2d()]
            else:
                layers += [binary.BinarizedConv2d(in_channels, out_channels, directions, kernel_size, padding=padding)]
            in_channels = out_channels
        layers += [nn.Flatten()]
        for _ in range(2):
            layers += [binary.BinarizedLinear(512, 512, directions)]
        if binary_last_layer:
            layers += [binary.BinarizedLinear(512, action_size, directions)]
        else:
            layers += [AsType(dtype=torch.float16)]
            layers += [base.PermutedLinear(512, action_size, directions)]
        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return torch.log_softmax(self.layers.forward(input).squeeze(),dim=1)

