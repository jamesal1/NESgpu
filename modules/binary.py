# Uses code from https://github.com/cooooorn/Pytorch-XNOR-Net

from modules.base import *
from torch.nn.parameter import Parameter
from torch.utils.cpp_extension import load
boolop_cuda = load(name="boolop_cuda", sources=["extensions/booleanOperations.cpp","extensions/booleanOperationsCuda.cu"])
from extensions import booleanOperations
import cupy


class AsType(nn.Module):

    def __init__(self,dtype):
        super(AsType, self).__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.type(self.dtype)

class ExtractBits(nn.Module):

    def __init__(self, dtype):
        nn.Module.__init__(self)
        self.dtype = dtype
        self.bitlength = torch.iinfo(dtype).bits
        self.mask = 2 ** torch.arange(self.bitlength, dtype=dtype)

    def forward(self, input):
        return (input.unsqueeze(-1) & self.mask).view(*input.shape[:-1],-1) > 0


class Binarized(Perturbed, nn.Module):

    def __init__(self, in_degree, out_degree,  directions, threshold=True, dtype=torch.int64, device="cuda"):
        nn.Module.__init__(self)
        Perturbed.__init__(self, directions, threshold)
        self.in_degree = in_degree
        self.out_degree = out_degree
        self.dtype = dtype
        self.threshold = threshold
        self.device = device
        self.directions = directions

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.bernoulli_()
            self.weight -= .5
            if self.threshold:
                self.bias.fill_(0)


    def set_noise(self, noise_scale=None):
        if self.seed is None:
            self.set_seed()
        if noise_scale is not None:
            self.set_noise_scale(noise_scale)
        print(torch.sigmoid(self.weight.abs()).mean())
        self.weight_noise = booleanOperations.sample_bits(torch.sigmoid(self.weight), self.directions, self.dtype, self.seed)
        self.bias_noise = self.bias + (torch.rand(size=(self.directions, self.out_degree), dtype=torch.float16,
                                                  device=self.weight.device) - .5) * self.noise_scale


    def set_grad(self, weights, l1=1e-5, l2=1e-5):
        weights = weights.type(torch.float16)
        weights = (weights - weights.mean())
        self.weight.grad = booleanOperations.weighted_sum(self.weight_noise, weights, self.in_degree)
        if l1:
            self.weight.grad += l1 * torch.sign(self.weight)
        if l2:
            self.weight.grad += l2 * self.weight
        self.bias.grad = weights @ self.bias_noise


class BinarizedLinear(Binarized):

    def __init__(self, in_features, out_features,  directions, threshold=True, dtype=torch.int64, device="cuda"):
        Binarized.__init__(self,in_features, out_features,  directions, threshold, dtype, device)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.zeros(out_features, in_features, dtype=torch.float16, device=device))
        if threshold:
            self.bias = Parameter(torch.zeros(out_features, dtype=torch.float16,device=device))
        self.reset_parameters()


    def forward(self, input):
        if self.perturbed_flag:
            packed_input = booleanOperations.pack(input, self.dtype) if input.dtype == torch.bool else input

            activation = booleanOperations.bmm(self.weight_noise, packed_input.view(self.directions, -1, 1)).squeeze(dim=2)
            return (2 * activation > (self.bias_noise + self.in_features)) if self.threshold else 2 * activation - self.in_features

            # if self.threshold:
            #     return booleanOperations.bmm_act(self.weight_noise, packed_input.view(self.directions, -1, 1), self.bias_noise).squeeze(dim=2)
            # return booleanOperations.bmm(self.weight_noise, packed_input.view(self.directions, -1, 1)).squeeze(dim=2)
        else:
            raise NotImplementedError


class BinarizedConv2d(Binarized):

    def __init__(self,in_channels, out_channels,  directions, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 threshold=True, padding_mode='zeros', dtype=torch.int64, device="cuda"):
        Binarized.__init__(self,in_channels, out_channels,  directions, threshold, dtype, device)
        if dilation != 1 or groups != 1 or padding_mode != "zeros":
            raise NotImplementedError
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size,int) else kernel_size
        self.weight = Parameter(torch.zeros(out_channels * self.kernel_size[0] * self.kernel_size[1], in_channels, dtype=torch.float16, device=device))
        if threshold:
            self.bias = Parameter(torch.zeros(out_channels, dtype=torch.float16, device=device))
        self.reset_parameters()

    def reset_parameters(self):
        Binarized.reset_parameters(self)
        with torch.no_grad():
            if self.threshold:
                self.bias.fill_(self.in_degree * self.kernel_size[0] * self.kernel_size[1] / 2)

    def set_noise(self, noise_scale=None):
        Binarized.set_noise(self, noise_scale)
        self.weight_noise_reshaped = self.weight_noise.view(self.directions, self.out_channels, *self.kernel_size, -1)
        self.weight_noise_reshaped = self.weight_noise_reshaped.permute([0, 2, 3, 4, 1]).view(self.directions, -1, self.out_channels).contiguous()

    def forward(self, input):
        if self.perturbed_flag:
            if input.dtype == torch.bool:
                packed_input = booleanOperations.pack(input.view(-1, self.in_channels), self.dtype).view(*input.shape[:3], -1)
            else:
                packed_input = input
            offset = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            if self.threshold:
                return booleanOperations.conv_act(packed_input, self.weight_noise_reshaped, self.bias_noise.type(torch.int32) + offset, *self.kernel_size,
                                                  self.padding, self.padding, self.stride, self.stride)
            return 2 * booleanOperations.conv(packed_input, self.weight_noise_reshaped, *self.kernel_size,
                                          self.padding, self.padding, self.stride, self.stride) - offset
        else:
            raise NotImplementedError


    def free_memory(self):
        Perturbed.free_memory(self)
        self.weight_noise_reshaped = None

class BinarizedMaxPool2d(nn.Module):

    def __init__(self, kernel_size=(2,2),padding=(0,0),stride=(2,2),dtype=torch.int64):
        nn.Module.__init__(self)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dtype = dtype

    def forward(self, input):
        if input.dtype == torch.bool:
            packed_input = booleanOperations.pack(input.view(-1, input.size(3)), self.dtype).view(*input.shape[:3], -1)
        else:
            packed_input = input
        return booleanOperations.max_pool2d(packed_input, *self.kernel_size, *self.padding, *self.stride)