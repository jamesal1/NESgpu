# Uses code from https://github.com/cooooorn/Pytorch-XNOR-Net

from modules.base import *
from torch.nn.parameter import Parameter
from torch.utils.cpp_extension import load
boolop_cuda = load(name="boolop_cuda", sources=["extensions/booleanOperations.cpp","extensions/booleanOperationsCuda.cu"])

def init_weights(layer):
    with torch.no_grad():
        layer.weight.normal_(std=1e-10)
        layer.bias.fill_(layer.in_features / 2)




class ExtractBits(nn.Module):

    def __init__(self, dtype):
        nn.Module.__init__(self)
        self.dtype = dtype
        self.bitlength = torch.iinfo(dtype).bits
        self.mask = 2 ** torch.arange(self.bitlength, dtype=dtype)

    def forward(self, input):
        return (input.unsqueeze(-1) & self.mask).view(*input.shape[:-1],-1) > 0





class BinarizedLinear(Perturbed, nn.Module):

    def __init__(self, out_features, in_features, directions, threshold=True, dtype=torch.int64):
        nn.Module.__init__(self)
        Perturbed.__init__(self, directions, threshold)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.HalfTensor(out_features, in_features))
        if threshold:
            self.bias = Parameter(torch.HalfTensor(out_features))
        self.dtype = dtype
        self.threshold = threshold

    def set_noise(self, noise_scale=None):
        if self.seed is None:
            self.set_seed()
        if noise_scale is not None:
            self.set_noise_scale(noise_scale)
        print(torch.sigmoid(self.weight))
        self.weight_noise = boolop_cuda.sample_bits(torch.sigmoid(self.weight), self.directions, torch.iinfo(self.dtype).bits, self.seed)
        print(self.weight_noise)
        self.bias_noise = self.bias + (torch.rand(size=(self.directions, self.out_features), dtype=torch.float16,
                                                  device=self.weight.device) - .5) * self.noise_scale

    def forward(self, input):

        if self.perturbed_flag:
            packed_input = boolop_cuda.pack(input, torch.iinfo(self.dtype).bits) if input.dtype == torch.bool else input
            print(packed_input.shape)
            print(input.shape)
            print(input.dtype)
            print(self.weight_noise.shape)
            activation = boolop_cuda.binary_bmm(self.weight_noise, packed_input.view(-1, self.in_features, 1))
            return activation > self.bias_noise if self.threshold else activation
        else:
            raise NotImplementedError

    def set_grad(self, weights):
        weights = (weights - weights.mean())
        self.weight.grad = boolop_cuda.weighted_sum(self.weight_noise, weights, self.in_features)
        self.bias.grad = weights @ self.bias_noise