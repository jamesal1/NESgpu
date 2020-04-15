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
            for l in self.perturbed_layers:
                getattr(l, name)(*args, **kwargs)
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

class PerturbedLinear(nn.Linear, Perturbed):

    def __init__(self, in_features, out_features, directions, bias=True):
        nn.Linear.__init__(self, in_features, out_features, bias)
        nn.Perturbed.__init__(self, directions, bias)


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


class PermutedLinear(nn.Linear, Perturbed):

    def __init__(self, in_features, out_features, directions, bias=True, permute_inputs=True, permute_outputs=True):
        nn.Linear.__init__(self, in_features, out_features, bias)
        Perturbed.__init__(self, directions, bias)
        self.permute_inputs = permute_inputs and in_features > 1
        self.permute_outputs = permute_outputs and out_features > 1
        self.input_permutations = None
        self.output_permutations = None

    def allocate_weight(self, low_memory=False):
        self.weight_noise_data = torch.empty_like(self.weight)
        self.weight_noise = self.weight_noise_data

    def allocate_bias(self, low_memory=False):
        if self.bias is not None:
            self.bias_noise_data = torch.empty(self.directions, self.out_features,
                                               device=self.bias.device, dtype=self.bias.dtype)
            self.bias_noise = self.bias_noise_data

    def set_noise(self):
        Perturbed.set_noise(self)
        gen = torch.manual_seed(self.seed)
        if self.permute_outputs:
            self.output_permutations = torch.empty(self.directions, self.out_features, dtype=torch.long)
            for i in range(self.directions):
                torch.randperm(self.out_features, out=self.output_permutations[i])
            self.output_permutations = self.output_permutations.to(self.weight.device)
        if self.permute_inputs:
            self.input_permutations = torch.empty(self.directions, self.in_features, dtype=torch.long)
            for i in range(self.directions):
                torch.randperm(self.in_features, out=self.input_permutations[i])
            self.input_permutations = self.input_permutations.to(self.weight.device)


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

    #pretty slow. On the bright side, NES is relatively suited for long epsiodes anyways.
    def set_grad(self, weights):

        if self.permute_inputs and self.permute_outputs: # atrocious performance on cpu
            self.weight.grad = torch.zeros_like(self.weight)
            inverse = torch.argsort(self.input_permutations, dim=1)
            for i in range(self.directions):
                self.weight.grad += self.weight_noise[self.output_permutations[i]][:,inverse[i]] * weights[i]
        elif self.permute_inputs:
            # self.weight.grad = torch.zeros_like(self.weight)
            # inverse = torch.argsort(self.input_permutations, dim=1)
            # for i in range(self.directions):
            #     self.weight.grad += self.weight_noise[:, inverse[i]] * weights[i]
            weighted_perms = torch.zeros((self.in_features, self.in_features), device=self.weight.device, dtype=self.weight.dtype)
            ar = torch.arange(self.in_features, device=self.weight.device)
            for i in range(self.directions):
                weighted_perms[ar, self.input_permutations[i]] += weights[i]
            self.weight.grad = torch.mm(self.weight_noise, weighted_perms)
        else:
            # self.weight.grad = torch.zeros_like(self.weight)
            # for i in range(self.directions):
            #     self.weight.grad += self.weight_noise[self.output_permutations[i]] * weights[i]
            weighted_perms = torch.zeros((self.out_features, self.out_features), device=self.weight.device, dtype=self.weight.dtype)
            ar = torch.arange(self.out_features, device=self.weight.device)
            for i in range(self.directions):
                weighted_perms[ar, self.output_permutations[i]] += weights[i]
            self.weight.grad = torch.mm(weighted_perms, self.weight_noise)
        if self.bias is not None:
            self.bias.grad = weights @ self.bias_noise
        self.free_memory()



def print_all_torch_tensors():
    import gc
    # print(torch.cuda.memory_stats())
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(obj.nelement(), obj.size())
        except:
            pass

def speed_test_linear(device="cpu"):

    k = 2 ** 10
    l = 2 ** 8
    m = 2 ** 9
    stride = l * m
    n = 128
    base = torch.empty((stride * k+l*m), device=device)
    # base = torch.empty((stride * k), device=device)
    inp = torch.empty((k, m, n), device=device)

    base.normal_()
    inp.normal_()
    inp_transposed = inp.permute([1, 0, 2]).contiguous()
    linear_weight = torch.as_strided(base, (k, l, m), (1, m, 1))
    print(linear_weight.is_contiguous())
    start = time.time()
    linear_weight_regular = linear_weight.clone().view(linear_weight.size())
    print("clone time", time.time() - start)

    for _ in range(10):
        start = time.time()
        expected = torch.bmm(linear_weight_regular, inp)
        print("regular", time.time()-start)

    for _ in range(10):
        start = time.time()
        # res = torch.einsum('ijk, ikl->ijl', [linear_weight, inp])
        res = torch.bmm(linear_weight, inp_transposed.permute([1, 0, 2]))
        print("overlap", time.time()-start)
    print(torch.allclose(res, expected))

def speed_test_conv(device="cpu"):

    h = 8
    w = 8
    c_in = 64
    c_out = 4
    d = 2 ** 16
    p = 2
    x = 8
    y = 8
    offset = 100
    weight_size = c_out * c_in * x * y
    bias = torch.ones(())
    base = torch.empty(weight_size + d * offset * c_out, device=device)
    inp = torch.empty((d, p, c_in, h, w), device=device)
    base.normal_()
    inp.normal_()
    conv_weight = torch.as_strided(base, (d, c_out, c_in, x, y), (c_out, 1, c_out * x * y, c_out * y, c_out))

    start = time.time()
    # conv_weight_regular = conv_weight.clone()
    conv_weight_contiguous = conv_weight.view(d * c_out, c_in, x, y).contiguous()
    print("clone time", time.time() - start)

    print_all_torch_tensors()
    time.sleep(1000)
    for _ in range(10):
        start = time.time()
        inp_view = inp.permute([1, 0, 2, 3, 4]).contiguous().view(p, d*c_in, h, w)
        expected = F.conv2d(inp_view, weight=conv_weight.view(d * c_out, c_in, x, y), bias=None, stride=1, groups=d )
        print("regular", time.time()-start)
    print_all_torch_tensors()
    time.sleep(1000)
    for _ in range(10):
        inp_permuted = inp.permute([1, 0, 2, 3, 4]).contiguous()
        inp_view = inp_permuted.view(p, d*c_in, h, w).contiguous()
        start = time.time()
        res = F.conv2d(inp_view, weight=conv_weight_contiguous, bias=None, stride=1, groups=d )
        print("overlap", time.time()-start)

    print(torch.allclose(res, expected))

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
    d = 1024
    m = 1024
    n = 1024
    l = PermutedLinear(n,m,d,permute_inputs=False).to(device)
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


if __name__ == "__main__":
    with torch.no_grad():
        pass
        # speed_test_linear("cuda")
        # speed_test_conv("cuda")
        # lin_vs_conv("cpu")
        # randperm_speed("cuda")
        test_perm()