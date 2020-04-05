import torch
from torch import nn
import torch.nn.functional as F

import time
import random


class Perturbed():

    def __init__(self,directions,bias=True):
        self.directions = directions
        self.perturbed_flag = False
        self.neg_mask = None
        self.noise_scale = None
        self.weight_noise = None
        self.weight_noise_data = None
        if bias:
            self.bias_noise = None
            self.bias_noise_data = None


    def set_noise_scale(self, noise_scale):
        self.noise_scale = noise_scale

    def set_noise(self):
        gen = torch.cuda.manual_seed(self.seed)
        self.weight_noise_data.normal_(std=self.noise_scale, generator=gen)
        if self.bias is not None:
            self.bias_noise_data.normal_(std=self.noise_scale, generator=gen)

    def set_seed(self, seed=None):
        self.seed = seed if seed is not None else random.randrange(100000000)

    def allocate_memory(self, low_memory=False):
        if low_memory:
            self.weight_noise_data = torch.empty(self.directions+self.weight.nelement() - 1,
                                     device=self.weight.device, dtype=self.weight.dtype)
            #shouldn't use self.weight.stride but it's convenient...
            self.weight_noise = torch.as_strided(self.weight_noise_data, (self.directions,)+self.weight.size(),
                                                 (1,)+self.weight.stride())
        else:
            self.weight_noise_data = torch.empty((self.directions,)+self.weight.size(),
                                                 device=self.weight.device, dtype=self.weight.dtype)
            self.weight_noise = self.weight_noise_data
        if self.bias is not None:
            if low_memory:
                self.bias_noise_data = torch.empty(self.directions+self.bias.nelement() - 1,
                                                   device=self.bias.device, dtype=self.bias.dtype)
                self.bias_noise = torch.as_strided(self.bias_noise_data, (self.directions,)+self.bias.size(),
                                                   (1,)+self.bias.stride())
            else:
                self.bias_noise_data = torch.empty((self.directions,)+self.bias.size(),
                                                   device=self.bias.device, dtype=self.bias.dtype)
                self.bias_noise = self.bias_noise_data

    def free_memory(self):
        self.weight_noise = None
        self.weight_noise_data = None
        if self.bias is not None:
            self.bias_noise = None
            self.bias_noise_data = None
        # self.neg_mask = None


    def set_grad(self, weights):
        self.weight.grad = (weights @ self.weight_noise.view(self.directions, -1)).view(*self.weight.size())
        if self.bias is not None:
            self.bias.grad = weights @ self.bias_noise
        self.free_memory()

class PerturbedLinear(nn.Linear,Perturbed):

    def __init__(self, in_features, out_features, directions, bias=True):
        nn.Linear.__init__(self,in_features,out_features,bias)
        nn.Perturbed.__init__(self,directions,bias)


    def forward(self, input):
        start = time.time()
        unperturbed = F.linear(input, self.weight, self.bias)
        # print("unperturbed",self.perturbed_flag, time.time() - start)
        if self.perturbed_flag:
            start = time.time()
            # self.set_noise()
            # print("noise", time.time() - start)
            batch_view_input = input.view(self.directions, -1, self.in_features)
            repeat_size = batch_view_input.size(1)
            # start = time.time()
            if self.bias is not None:
                perturbations = torch.baddbmm(self.bias_noise.view(self.directions, 1, self.out_features),
                                              batch_view_input,
                                              self.weight_noise.permute([0, 2, 1]))
            else:
                perturbations = torch.bmm(batch_view_input, self.weight_noise.permute([0, 2, 1]))
            # print("perturbed", time.time() - start)
            # start = time.time()
            if self.neg_mask is None or self.neg_mask.size(1)!=repeat_size: #seems faster than negating half of perturbations
                self.neg_mask = torch.ones((1,repeat_size,1), device=cuda_device, dtype=precision)
                self.neg_mask[:, repeat_size // 2:, :] *= -1
            # print("negative", time.time() - start)
            # start = time.time()
            # add = (perturbations*self.neg_mask).view(*unperturbed.size()) + unperturbed
            add = torch.addcmul(unperturbed.view(*perturbations.size()),perturbations,self.neg_mask).view(*unperturbed.size())
            # print("add", time.time() - start)
            # self.clear_noise()
            return add
        return unperturbed


class PerturbedConv2d(nn.Conv2d,Perturbed):

    def __init__(self, in_features, out_features, directions, bias=True):
        nn.Linear.__init__(self,in_features,out_features,bias)
        nn.Perturbed.__init__(self,directions,bias)

    def forward(self, input):
        pass


class PerturbedModel():

    def __init__(self, model, directions):
        self.model = model
        self.directions = directions
        self.perturbed_layers = []

        def get_perturbed_layers(m):
            if isinstance(m,Perturbed):
                self.perturbed_layers+=[m]
        self.model.apply(get_perturbed_layers)


    def forward(self,*args):
        for l in self.perturbed_layers:
            l.perturbed_flag = True
        ret = self.model.forward(*args)
        for l in self.perturbed_layers:
            l.perturbed_flag = False
        return ret



    def __getattr__(self, name):
        def ret(*args):
            for l in self.perturbed_layers:
                getattr(l,name)(*args)
        return ret



def speed_tests(device="cpu"):

    k = 2 ** 10
    l = 2 ** 11
    m = 2 ** 12
    n = 128
    base = torch.empty((k+l*m), device=device)
    inp = torch.empty((k, m, n), device=device)
    base.normal_()
    inp.normal_()
    linear_weight = torch.as_strided(base,(k,l,m),(1,m,1))

    start = time.time()
    # res = torch.einsum('ijk,ikl->ijl', [linear_weight, inp])
    res = torch.bmm(linear_weight, inp)
    print("overlap", time.time()-start)
    start = time.time()
    linear_weight_regular = linear_weight.clone()
    print("clone time",time.time() - start)
    expected = torch.bmm(linear_weight_regular,inp)
    start = time.time()
    print("regular", time.time()-start)
    print(torch.allclose(res,expected))


if __name__ == "__main__":
    with torch.no_grad():
        # speed_tests()

        speed_tests("cuda")