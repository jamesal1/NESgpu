# Uses code from https://github.com/cooooorn/Pytorch-XNOR-Net

from modules.base import *
from torch.nn.parameter import Parameter
from torch.utils.cpp_extension import load
boolop_cuda = load(name="boolop_cuda", sources=["extensions/booleanOperations.cpp","extensions/booleanOperationsCuda.cu"])
from extensions import booleanOperations






class ExtractBits(nn.Module):

    def __init__(self, dtype):
        nn.Module.__init__(self)
        self.dtype = dtype
        self.bitlength = torch.iinfo(dtype).bits
        self.mask = 2 ** torch.arange(self.bitlength, dtype=dtype)

    def forward(self, input):
        return (input.unsqueeze(-1) & self.mask).view(*input.shape[:-1],-1) > 0





class BinarizedLinear(Perturbed, nn.Module):

    def __init__(self,in_features, out_features,  directions, threshold=True, dtype=torch.int64, device="cuda"):
        nn.Module.__init__(self)
        Perturbed.__init__(self, directions, threshold)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.zeros(out_features, in_features, dtype=torch.float16, device=device))
        if threshold:
            self.bias = Parameter(torch.zeros(out_features, dtype=torch.float16,device=device))
        self.dtype = dtype
        self.threshold = threshold
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.bernoulli_()
            self.weight -= .5
            self.weight *= 5
            if self.threshold:
                self.bias.fill_(self.in_features / 2)

    def set_noise(self, noise_scale=None):
        if self.seed is None:
            self.set_seed()
        if noise_scale is not None:
            self.set_noise_scale(noise_scale)
        self.weight_noise = boolop_cuda.sample_bits(torch.sigmoid(self.weight), self.directions, torch.iinfo(self.dtype).bits, self.seed)
        self.bias_noise = self.bias + (torch.rand(size=(self.directions, self.out_features), dtype=torch.float16,
                                                  device=self.weight.device) - .5) * self.noise_scale

    def forward(self, input):

        if self.perturbed_flag:
            packed_input = boolop_cuda.pack(input, torch.iinfo(self.dtype).bits) if input.dtype == torch.bool else input
            activation = boolop_cuda.binary_bmm(self.weight_noise, packed_input.view(self.directions, -1, 1)).squeeze(dim=2)
            return activation > self.bias_noise if self.threshold else activation
        else:
            raise NotImplementedError

    def set_grad(self, weights):
        weights = weights.type(torch.float16)
        weights = (weights - weights.mean())
        self.weight.grad = boolop_cuda.binary_weighted_sum(self.weight_noise, weights, self.in_features)
        self.bias.grad = weights @ self.bias_noise


class BinarizedConv2d(Perturbed, nn.Module):

    def __init__(self,in_channels, out_channels,  directions, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 threshold=True, padding_mode='zeros', dtype=torch.int64, device="cuda"):
        nn.Module.__init__(self)
        Perturbed.__init__(self, directions, threshold)
        if dilation != 1 or groups != 1 or padding_mode != "zeros":
            raise NotImplementedError
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.weight = Parameter(torch.zeros(out_channels, * kernel_size, in_channels, dtype=torch.float16, device=device))
        if threshold:
            self.bias = Parameter(torch.zeros(out_channels, dtype=torch.float16, device=device))
        self.dtype = dtype
        self.threshold = threshold
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.bernoulli_()
            self.weight -= .5
            self.weight *= 5
            if self.threshold:
                self.bias.fill_(self.in_channels / 2)

    def set_noise(self, noise_scale=None):
        if self.seed is None:
            self.set_seed()
        if noise_scale is not None:
            self.set_noise_scale(noise_scale)
        self.weight_noise = boolop_cuda.sample_bits(torch.sigmoid(self.weight), self.directions, torch.iinfo(self.dtype).bits, self.seed)
        self.bias_noise = self.bias + (torch.rand(size=(self.directions, self.out_channels), dtype=torch.float16,
                                                  device=self.weight.device) - .5) * self.noise_scale

    def forward(self, input):

        if self.perturbed_flag:
            packed_input = boolop_cuda.pack(input, torch.iinfo(self.dtype).bits) if input.dtype == torch.bool else input
            activation = booleanOperations.conv_bmm()
            activation = boolop_cuda.binary_bmm(self.weight_noise, packed_input.view(self.directions, -1, 1)).squeeze(dim=2)
            return activation > self.bias_noise if self.threshold else activation
        else:
            raise NotImplementedError

    def set_grad(self, weights):
        weights = weights.type(torch.float16)
        weights = (weights - weights.mean())
        self.weight.grad = boolop_cuda.binary_weighted_sum(self.weight_noise, weights, self.in_channels)
        self.bias.grad = weights @ self.bias_noise