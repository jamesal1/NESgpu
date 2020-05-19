# Uses code from https://github.com/cooooorn/Pytorch-XNOR-Net

from modules import *
from torch.nn.parameter import Parameter


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

    def __init__(self, out_features, in_features, directions):
        nn.Module.__init__(self)
        Perturbed.__init__(self, directions, True)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))

    def set_noise(self, noise_scale=None):
        if self.seed is None:
            self.set_seed()
        if noise_scale is not None:
            self.set_noise_scale(noise_scale)
        torch.cuda.manual_seed(self.seed)
        self.weight_noise = torch.sigmoid(self.weight) > torch.rand(size=(self.directions,
                                                                          self.out_features,
                                                                          self.in_features),
                                                                    dtype=torch.float16,
                                                                    device=self.weight.device)
        self.bias_noise = self.bias + (torch.rand(size=(self.directions, self.out_features), dtype=torch.float16,
                                                  device=self.weight.device) - .5) * self.noise_scale

    def forward(self, input):
        if self.perturbed_flag:
            activation = (self.weight_noise == input.view(-1, self.directions, 1, self.in_features)).sum(dim=-1)
            return activation > self.bias_noise
        else:
            activation = (self.weight.sign() == input.unsqueeze(-2)).sum(dim=-1)
            return activation > self.bias

    def set_grad(self, weights):
        weights = (weights - weights.mean())
        self.weight.grad = (weights @ self.weight_noise.view(self.directions, -1)).view_as(self.weight)
        self.bias.grad = weights @ self.bias_noise
