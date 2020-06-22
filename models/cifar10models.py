from modules import base, binary
import torch
from torch import nn
import math



class VGG11(nn.Module):


    def __init__(self, directions, action_size, binary_last_layer=False, in_channels=24, device="cuda"):
        super(VGG11, self).__init__()
        kernel_size = 3
        padding = 1
        stride = 1
        layers = []
        for out_channels in [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']:
            if out_channels == "M":
                layers += [binary.BinarizedMaxPool2d()]
            else:
                layers += [binary.BinarizedConv2d(in_channels, out_channels, directions, kernel_size, padding=padding, device=device)]
                in_channels = out_channels
        layers += [nn.Flatten()]
        for _ in range(2):
            layers += [binary.BinarizedLinear(512, 512, directions, device=device)]
        if binary_last_layer:
            layers += [binary.BinarizedLinear(512, action_size, directions, device=device)]
        else:
            layers += [binary.AsType(dtype=torch.float16)]
            layers += [base.PermutedLinear(512, action_size, directions).to(device).type(torch.float16)]
        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return self.layers.forward(input).squeeze().type(torch.float16)

