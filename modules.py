import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.modules.utils as module_utils

import time
import random


class PerturbedModel():

    def __init__(self, model, directions):
        self.model = model
        self.directions = directions
        self.perturbed_layers = []

        def get_perturbed_layers(m):
            if isinstance(m, Perturbed):
                self.perturbed_layers+=[m]
        self.model.apply(get_perturbed_layers)


    def forward(self, input, *args):
        for l in self.perturbed_layers:
            l.perturbed_flag = True
        ret = self.model.forward(input, *args)
        for l in self.perturbed_layers:
            l.perturbed_flag = False
        return ret



    def __getattr__(self, name):
        def ret(*args, **kwargs):
            if hasattr(Perturbed, name):
                for l in self.perturbed_layers:
                    getattr(l, name)(*args, **kwargs)
            else:
                for l in self.perturbed_layers:
                    l.perturbed_flag = True
                res = getattr(self.model, name)(*args, **kwargs)
                for l in self.perturbed_layers:
                    l.perturbed_flag = False
                return res
        return ret

class Perturbed():

    def __init__(self, directions, bias=True):
        self.directions = directions
        self.perturbed_flag = False
        self.noise_scale = None
        self.weight_noise = None
        self.weight_noise_data = None
        self.seed = None
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

    def allocate_weight(self, low_memory=False):
        if low_memory:
            self.weight_noise_data = torch.empty(self.directions+self.weight.nelement() - 1, 
                                                 device=self.weight.device, dtype=self.weight.dtype)
            #shouldn't use self.weight.stride but it's convenient
            self.weight_noise = torch.as_strided(self.weight_noise_data, (self.directions, )+self.weight.size(), 
                                                 (1, )+self.weight.stride())
        else:
            self.weight_noise_data = torch.empty((self.directions, )+self.weight.size(), 
                                                 device=self.weight.device, dtype=self.weight.dtype)
            self.weight_noise = self.weight_noise_data

    def allocate_bias(self, low_memory=False):
        if self.bias is not None:
            if low_memory:
                self.bias_noise_data = torch.empty(self.directions+self.bias.nelement() - 1, 
                                                   device=self.bias.device, dtype=self.bias.dtype)
                self.bias_noise = torch.as_strided(self.bias_noise_data, (self.directions, )+self.bias.size(), 
                                                   (1, )+self.bias.stride())
            else:
                self.bias_noise_data = torch.empty((self.directions, )+self.bias.size(), 
                                                   device=self.bias.device, dtype=self.bias.dtype)
                self.bias_noise = self.bias_noise_data


    def allocate_memory(self, low_memory=False):
        self.allocate_weight(low_memory)
        if self.bias is not None:
            self.allocate_bias(low_memory)


    def free_memory(self):
        self.weight_noise = None
        self.weight_noise_data = None
        if self.bias is not None:
            self.bias_noise = None
            self.bias_noise_data = None


    def set_grad(self, weights):
        self.weight.grad = (weights @ self.weight_noise.view(self.directions, -1)).view(*self.weight.size())
        if self.bias is not None:
            self.bias.grad = weights @ self.bias_noise
        self.free_memory()



class Permuted(Perturbed):

    def __init__(self,  in_degree, out_degree, directions, bias=True, permutation="auto"):
        Perturbed.__init__(self, directions, bias)
        if permutation == "auto":
            if 1 < in_degree < 32 and 1 < out_degree < 32:
                permutation = "both"
            elif in_degree > out_degree:
                permutation = "in"
            else:
                permutation = "out"
        if permutation == "both":
            self.permute_inputs = True
            self.permute_outputs = True
        elif permutation == "in":
            self.permute_inputs = True
            self.permute_outputs = False
        elif permutation == "out":
            self.permute_inputs = False
            self.permute_outputs = True
        else:
            print("Permutation setting not recognized")
            raise NotImplementedError
        self.input_permutations = None
        self.output_permutations = None
        self.in_degree = in_degree
        self.out_degree = out_degree

    def allocate_weight(self, low_memory=False):
        self.weight_noise_data = torch.empty_like(self.weight)
        self.weight_noise = self.weight_noise_data


    def free_memory(self):
        Perturbed.free_memory(self)
        self.input_permutations = None
        self.output_permutations = None

    def set_noise(self):
        Perturbed.set_noise(self)
        gen = torch.manual_seed(self.seed)
        if self.permute_outputs:
            self.output_permutations = torch.empty(self.directions, self.out_degree, dtype=torch.long)
            for i in range(self.directions):
                torch.randperm(self.out_degree, out=self.output_permutations[i])
            self.output_permutations = self.output_permutations.to(self.weight.device)
        if self.permute_inputs:
            self.input_permutations = torch.empty(self.directions, self.in_degree, dtype=torch.long)
            for i in range(self.directions):
                torch.randperm(self.in_degree, out=self.input_permutations[i])
            self.input_permutations = self.input_permutations.to(self.weight.device)


class SparsePerturbed(Perturbed):

    def __init__(self,  in_degree, out_degree, directions, bias=True, sparsity="auto"):
        Perturbed.__init__(self, directions, bias)
        if sparsity == "auto":
            sparsity = max(1, in_degree // directions)
        self.selections = None
        self.in_degree = in_degree
        self.out_degree = out_degree

    def allocate_weight(self, low_memory=False):
        self.weight_noise_data = torch.empty_like(self.weight)
        self.weight_noise = self.weight_noise_data

    def allocate_bias(self, low_memory=False):
        if self.bias is not None:
            self.bias_noise_data = torch.empty(self.directions, self.out_degree,
                                               device=self.bias.device, dtype=self.bias.dtype)
            self.bias_noise = self.bias_noise_data

    def free_memory(self):
        Perturbed.free_memory(self)
        self.selections = None

    def set_noise(self):
        Perturbed.set_noise(self)
        gen = torch.manual_seed(self.seed)
        if self.permute_outputs:
            self.output_permutations = torch.empty(self.directions, self.out_degree, dtype=torch.long)
            for i in range(self.directions):
                torch.randperm(self.out_degree, out=self.output_permutations[i])
            self.output_permutations = self.output_permutations.to(self.weight.device)
        if self.permute_inputs:
            self.input_permutations = torch.empty(self.directions, self.in_degree, dtype=torch.long)
            for i in range(self.directions):
                torch.randperm(self.in_degree, out=self.input_permutations[i])
            self.input_permutations = self.input_permutations.to(self.weight.device)




class PerturbedLinear(nn.Linear, Perturbed):

    def __init__(self, in_features, out_features, directions, bias=True):
        nn.Linear.__init__(self, in_features, out_features, bias)
        Perturbed.__init__(self, directions, bias)


    def forward(self, input):
        unperturbed = F.linear(input, self.weight, self.bias)
        if self.perturbed_flag:
            repeat_size = input.size(1)
            if self.bias is not None:
                perturbations = torch.baddbmm(self.bias_noise.view(self.directions, 1, self.out_features), 
                                              input.view(self.directions, -1, self.in_features),
                                              self.weight_noise.permute([0, 2, 1]))
            else:
                perturbations = torch.bmm(input.view(self.directions, -1, self.in_features),
                                          self.weight_noise.permute([0, 2, 1]))
            perturbations[:, (repeat_size + 1) // 2:] *= -1
            add = unperturbed + perturbations.view_as(unperturbed)
            return add
        return unperturbed


class PerturbedConv2d(nn.Conv2d, Perturbed):

    def __init__(self, in_channels, out_channels, kernel_size, directions, stride=1, 
                 padding=0, dilation=1, groups=1, 
                 bias=True, padding_mode='zeros'):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        Perturbed.__init__(self, directions, bias)

    def allocate_weight(self, low_memory=False):
        if low_memory:
            c_out = self.out_channels
            # c_in = self.in_channels // self.groups
            x, y = self.kernel_size
            self.weight_noise_data = torch.empty((self.directions - 1) * c_out + self.weight.nelement(), 
                                                 device=self.weight.device, dtype=self.weight.dtype)

            self.weight_noise = torch.as_strided(self.weight_noise_data, (self.directions * c_out, )+self.weight.size()[1:], 
                                                 (1, c_out * x * y, c_out * y, c_out))
        else:
            Perturbed.allocate_weight(self, False)
        print(self.weight_noise_data.shape)

    def allocate_memory(self, low_memory=False):
        self.allocate_weight(low_memory)
        self.allocate_bias(False)

    #based on https://github.com/pytorch/pytorch/issues/17983
    def forward(self, input):
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._padding_repeated_twice, mode=self.padding_mode)
            padding = module_utils._pair(0)
        else:
            padding = self.padding
        start = time.time()
        unperturbed = F.conv2d(input, self.weight, self.bias, self.stride, 
                               padding, self.dilation, self.groups)

        if self.perturbed_flag:
            bias_noise = self.bias_noise.view(-1) if self.bias is not None else None
            inp_view = input.view(self.directions, -1, *input.size()[1:]).permute([1,0,2,3,4]).contiguous()\
                            .view(-1, self.directions * self.in_channels, *input.size()[-2:])
            repeat_size = inp_view.size(0)
            perturbations = F.conv2d(inp_view, self.weight_noise.view(-1, *self.weight.size()[1:]), bias_noise, self.stride,
                                     padding, self.dilation, self.groups * self.directions)
            perturbations = perturbations.view(repeat_size, self.directions, self.out_channels,
                                               perturbations.shape[-2], perturbations.shape[-1]).permute([1,0,2,3,4]).contiguous()
            perturbations[:, (repeat_size + 1) // 2:] *= -1
            add = unperturbed + perturbations.view_as(unperturbed)
            return add
        return unperturbed


class PermutedLinear(nn.Linear, Permuted):

    def __init__(self, in_features, out_features, directions, bias=True, permutation="auto"):
        nn.Linear.__init__(self, in_features, out_features, bias)
        Permuted.__init__(self, in_features, out_features, directions, bias=True, permutation=permutation)

    def forward(self, input):
        unperturbed = F.linear(input, self.weight, self.bias)
        if self.perturbed_flag:
            permuted_input = torch.gather(input, 1, self.input_permutations) if self.permute_inputs else input
            perturbations = torch.mm(permuted_input, self.weight_noise.t())
            permuted_output = torch.gather(perturbations, 1, self.output_permutations) if self.permute_outputs else perturbations
            add = unperturbed + permuted_output
            if self.bias is not None:
                add += self.bias_noise
            return add
        return unperturbed


    def set_grad(self, weights):
        if self.permute_inputs and self.permute_outputs:
            inverse = torch.argsort(self.input_permutations, dim=1)
            mat_size = self.out_features * self.in_features
            permutations = inverse.view(self.directions,1,self.in_features) + (self.output_permutations * self.in_features).view(self.directions,self.out_features,1)
            ar = torch.arange(mat_size, device=self.weight.device)
            permutations_1d = permutations.view(self.directions, mat_size) + ar * mat_size
            weighted_perms = torch.zeros((mat_size, mat_size), device=self.weight.device, dtype=self.weight.dtype)
            weighted_perms.put_(permutations_1d, weights.view(-1,1).expand(self.directions, mat_size), accumulate=True)
            self.weight.grad = torch.mm(weighted_perms, self.weight_noise.view(-1,1)).reshape(self.out_features,self.in_features)
            # self.weight.grad = torch.zeros_like(self.weight)
            # for i in range(self.directions):
            #     self.weight.grad += self.weight_noise[self.output_permutations[i]][:,inverse[i]] * weights[i]
        elif self.permute_inputs:
            # weight_grad = torch.zeros_like(self.weight)
            # inverse = torch.argsort(self.input_permutations, dim=1)
            # for i in range(self.directions):
            #     weight_grad += self.weight_noise[:, inverse[i]] * weights[i]
            weighted_perms = torch.zeros((self.in_features, self.in_features), device=self.weight.device, dtype=self.weight.dtype)
            ar = torch.arange(self.in_features, device=self.weight.device)
            input_permutations_1d = self.input_permutations + ar * self.in_features
            weighted_perms.put_(input_permutations_1d, weights.view(-1,1).expand(self.directions,self.in_features), accumulate=True)
            self.weight.grad = torch.mm(self.weight_noise, weighted_perms)
        else:
            # weight_grad = torch.zeros_like(self.weight)
            # for i in range(self.directions):
            #     weight_grad += self.weight_noise[self.output_permutations[i]] * weights[i]
            weighted_perms = torch.zeros((self.out_features, self.out_features), device=self.weight.device, dtype=self.weight.dtype)
            ar = torch.arange(self.out_features, device=self.weight.device)
            output_permutations_1d = self.output_permutations + ar * self.out_features
            weighted_perms.put_(output_permutations_1d,weights.view(-1,1).expand(self.directions, self.out_features), accumulate=True)
            self.weight.grad = torch.mm(weighted_perms, self.weight_noise)
        if self.bias is not None:
            self.bias.grad = weights @ self.bias_noise


class PermutedConv2d(nn.Conv2d, Permuted):

    def __init__(self, in_channels, out_channels, kernel_size, directions, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', permutation="auto"):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        Permuted.__init__(self, in_channels, out_channels, directions, bias=True, permutation=permutation)


    def forward(self, input):
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._padding_repeated_twice, mode=self.padding_mode)
            padding = module_utils._pair(0)
        else:
            padding = self.padding
        unperturbed = F.conv2d(input, self.weight, self.bias, self.stride,
                               padding, self.dilation, self.groups)

        if self.perturbed_flag:
            if self.permute_inputs:
                input_permutations_1d = (self.input_permutations + (torch.arange(self.directions, device=input.device) * self.in_channels).unsqueeze(1)).flatten()
                permuted_input = torch.index_select(input.view(-1,*input.shape[2:]), 0, input_permutations_1d).view_as(input)
            else:
                permuted_input = input
            perturbations = F.conv2d(permuted_input, self.weight_noise, None, self.stride,
                                     padding, self.dilation, self.groups)
            if self.permute_outputs:
                output_permutations_1d = (self.output_permutations + (torch.arange(self.directions, device=input.device) * self.out_channels).unsqueeze(1)).flatten()
                permuted_output = torch.index_select(perturbations.view(-1,*perturbations.shape[2:]), 0, output_permutations_1d).view_as(perturbations)
            else:
                permuted_output = perturbations
            add = unperturbed + permuted_output

            if self.bias is not None:
                add += self.bias_noise.unsqueeze(-1).unsqueeze(-1)

            return add
        return unperturbed



    def set_grad(self, weights):
        if self.permute_inputs and self.permute_outputs:
            inverse = torch.argsort(self.input_permutations, dim=1)
            mat_size = self.out_channels * self.in_channels
            permutations = inverse.view(self.directions,1,self.in_channels) + (self.output_permutations * self.in_channels).view(self.directions,self.out_channels,1)
            ar = torch.arange(mat_size, device=self.weight.device)
            permutations_1d = permutations.view(self.directions, mat_size) + ar * mat_size
            weighted_perms = torch.zeros((mat_size, mat_size), device=self.weight.device, dtype=self.weight.dtype)
            weighted_perms.put_(permutations_1d, weights.view(-1,1).expand(self.directions, mat_size), accumulate=True)
            self.weight.grad = torch.mm(weighted_perms, self.weight_noise.view(mat_size,-1)).view_as(self.weight)
            # self.weight.grad = torch.zeros_like(self.weight)
            # for i in range(self.directions):
            #     self.weight.grad += self.weight_noise[self.output_permutations[i]][:,inverse[i]] * weights[i]
        elif self.permute_inputs:
            # weight_grad = torch.zeros_like(self.weight)
            # inverse = torch.argsort(self.input_permutations, dim=1)
            # for i in range(self.directions):
            #     weight_grad += self.weight_noise[:, inverse[i]] * weights[i]
            weighted_perms = torch.zeros((self.in_channels, self.in_channels), device=self.weight.device, dtype=self.weight.dtype)
            ar = torch.arange(self.in_channels, device=self.weight.device)
            input_permutations_1d = self.input_permutations + ar * self.in_channels
            weighted_perms.put_(input_permutations_1d, weights.view(-1,1).expand(self.directions,self.in_channels), accumulate=True)
            self.weight.grad = torch.mm(self.weight_noise.permute([0,2,3,1]).contiguous().view(-1,self.in_channels),
                                        weighted_perms).view(self.out_channels, *self.kernel_size,self.in_channels).permute([0,3,1,2])
        else:
            # weight_grad = torch.zeros_like(self.weight)
            # for i in range(self.directions):
            #     weight_grad += self.weight_noise[self.output_permutations[i]] * weights[i]
            weighted_perms = torch.zeros((self.out_channels, self.out_channels), device=self.weight.device, dtype=self.weight.dtype)
            ar = torch.arange(self.out_channels, device=self.weight.device)
            output_permutations_1d = self.output_permutations + ar * self.out_channels
            weighted_perms.put_(output_permutations_1d,weights.view(-1,1).expand(self.directions, self.out_channels), accumulate=True)
            self.weight.grad = torch.mm(weighted_perms, self.weight_noise.view(self.out_channels,-1)).view_as(self.weight)
        if self.bias is not None:
            self.bias.grad = weights @ self.bias_noise
        self.free_memory()



def lin_vs_conv(device="cpu"):
    times = 5
    d = 2 ** 9
    p = 2
    x = 128
    x_sq = x ** 2
    out = 64
    batch_weight = torch.empty(d * x_sq * out, device=device)
    weight = torch.empty(x_sq * out, device=device)
    input = torch.empty(d * p * x_sq, device=device)
    batch_input_linear = input.view(d, x_sq, p)
    batch_input_conv_x = input.view(p, d, x, x)
    batch_input_conv_1 = input.view(p, d * x_sq, 1, 1)
    input_linear = input.view(x_sq, p * d)
    input_conv_x = input.view(p * d, 1, x, x)
    input_conv_1 = input.view(p * d, x_sq, 1, 1)
    batch_weight_linear = batch_weight.view(d, out, x_sq)
    batch_weight_conv_x = batch_weight.view(d * out, 1, x, x)
    batch_weight_conv_1 = batch_weight.view(d * out, x_sq, 1, 1)
    weight_linear = weight.view(out, x_sq)
    weight_conv_x = weight.view(out, 1, x, x)
    weight_conv_1 = weight.view(out, x_sq, 1, 1)
    for _ in range(2):
        for i in range(times):
            torch.cuda.synchronize()
            start = time.time()
            torch.bmm(batch_weight_linear, batch_input_linear)
            torch.cuda.synchronize()
            print("batch_linear", time.time() - start)
        for i in range(times):
            torch.cuda.synchronize()
            start = time.time()
            torch.mm(weight_linear, input_linear)
            torch.cuda.synchronize()
            print("linear", time.time() - start)
        for i in range(times):
            torch.cuda.synchronize()
            start = time.time()
            F.conv2d(batch_input_conv_x, weight=batch_weight_conv_x, groups=d)
            torch.cuda.synchronize()
            print("batch_conv_x", time.time() - start)
        for i in range(times):
            torch.cuda.synchronize()
            start = time.time()
            F.conv2d(input_conv_x, weight=weight_conv_x)
            torch.cuda.synchronize()
            print("conv_x", time.time() - start)
        for i in range(times):
            torch.cuda.synchronize()
            start = time.time()
            F.conv2d(batch_input_conv_1, weight=batch_weight_conv_1, groups=d)
            torch.cuda.synchronize()
            print("batch_conv_1", time.time() - start)
        for i in range(times):
            torch.cuda.synchronize()
            start = time.time()
            F.conv2d(input_conv_1, weight=weight_conv_1)
            torch.cuda.synchronize()
            print("conv_1", time.time() - start)


def randperm_speed(device="cuda"):
    start = time.time()
    m = 256
    n = 256


    for _ in range(20):

        x = torch.empty(m,n, dtype=torch.long)
        w = torch.zeros(n,n, dtype=torch.int32)
        f = torch.zeros(n,n)
        g = torch.zeros(n,n)
        y = torch.empty(m,n)
        a = torch.arange(n)
        b = torch.zeros(n, dtype=torch.long)
        y.random_()
        # f.random_()
        for i in range(m):
            torch.randperm(n, out=x[i])
        a = a.to(device)
        f = f.to(device)
        g = g.to(device)
        w = w.to(device)
        x = x.to(device)
        y = y.to(device)
        b = b.to(device)
        torch.cuda.synchronize()
        start = time.time()
        torch.argsort(x,dim=1)
        z = torch.gather(y, 1, x)
        # for i in range(m):
        #     g+=f[:,x[i]]

        for i in range(m):
            f[b,b] += 1
        print(f.sum())
        torch.mm(f,f)

            # print(w)
        torch.cuda.synchronize()
        # print(y,z)
        print(time.time() - start)


def test_perm(device="cuda"):
    d = 128
    m = 128
    n = 128
    l = PermutedLinear(n,m,d,permutation="out").to(device)
    # l = PermutedLinear(n,m,d).to(device)

    for _ in range(10):
        l.allocate_memory()
        l.set_noise_scale(1.)
        l.set_seed()
        l.set_noise()
        w = torch.rand(d,device=device)
        # w = torch.ones(d,device=device)
        torch.cuda.synchronize()
        start = time.time()
        l.set_grad(w)
        torch.cuda.synchronize()
        print(time.time() - start)

def test_perm_conv(device="cuda"):
    d = 2048
    c_in = 16
    c_out = 16
    h = 32
    w = 32
    x = 3
    y = 3
    inp = torch.empty((d, c_in, h, w), device=device)
    inp.normal_()
    l = PermutedConv2d(c_in,c_out,(x,y),d,permutation="both").to(device)
    # l = PerturbedConv2d(c_in,c_out,(x,y),d).to(device)

    for _ in range(10):
        l.allocate_memory()
        l.set_noise_scale(1.)
        l.set_seed()
        l.set_noise()
        w = torch.rand(d,device=device)
        # w = torch.ones(d,device=device)
        torch.cuda.synchronize()
        start = time.time()
        l.perturbed_flag = True
        l.forward(inp)
        # l.set_grad(w)
        torch.cuda.synchronize()
        print(time.time() - start)
    for _ in range(10):
        l.allocate_memory()
        l.set_noise_scale(1.)
        l.set_seed()
        l.set_noise()
        w = torch.rand(d,device=device)
        # w = torch.ones(d,device=device)
        torch.cuda.synchronize()
        start = time.time()
        l.perturbed_flag = False
        l.forward(inp)
        # l.set_grad(w)
        torch.cuda.synchronize()
        print("off",time.time() - start)

def test_add(device="cuda"):
    n = 1024


    for _ in range(10):
        t = torch.zeros(n,n, device=device,dtype=torch.long)
        ar = torch.arange(n,device=device)
        t2 = torch.as_strided(t,(n,n),(0,0))
        z = torch.zeros(n,device=device,dtype=torch.long)
        # w = torch.ones(d,device=device)
        torch.cuda.synchronize()
        start = time.time()
        t2[ar,ar] += ar
        torch.cuda.synchronize()
        print(time.time() - start)
        print(t)
        print(t2)

def sparse_test(device="cuda"):
    l = 1024
    m = 1024
    n = 1024
    z = torch.zeros(l,m,device=device)
    zsp = z.to_sparse()
    o = torch.ones(m,n,device=device)
    osp = o.to_sparse()
    e = torch.eye(m,device=device)
    esp = e.to_sparse()

    for _ in range(10):
        torch.cuda.synchronize()
        start = time.time()
        zsp @ o
        torch.cuda.synchronize()
        print("sparse",time.time() - start)
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.time()
        z @ o
        torch.cuda.synchronize()
        print("dense",time.time() - start)




if __name__ == "__main__":
    with torch.no_grad():
        pass
        # speed_test_linear("cuda")
        # speed_test_conv("cuda")
        # lin_vs_conv("cpu")
        # randperm_speed("cuda")
        # sparse_test("cuda")
        test_perm_conv("cuda")
        # test_add("cuda")