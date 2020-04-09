import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.modules.utils as module_utils

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

    def allocate_weight(self, low_memory=False):
        if low_memory:
            self.weight_noise_data = torch.empty(self.directions+self.weight.nelement() - 1,
                                                 device=self.weight.device, dtype=self.weight.dtype)
            #shouldn't use self.weight.stride but it's convenient
            self.weight_noise = torch.as_strided(self.weight_noise_data, (self.directions,)+self.weight.size(),
                                                 (1,)+self.weight.stride())
        else:
            self.weight_noise_data = torch.empty((self.directions,)+self.weight.size(),
                                                 device=self.weight.device, dtype=self.weight.dtype)
            self.weight_noise = self.weight_noise_data

    def allocate_bias(self, low_memory=False):
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


    def allocate_memory(self, low_memory=False):
        self.allocate_weight(low_memory)
        self.allocate_bias(low_memory)


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
            repeat_size = input.size(1)
            # start = time.time()
            if self.bias is not None:
                perturbations = torch.baddbmm(self.bias_noise.view(self.directions, 1, self.out_features),
                                              input,
                                              self.weight_noise.permute([0, 2, 1]))
            else:
                perturbations = torch.bmm(input, self.weight_noise.permute([0, 2, 1]))
            # print("perturbed", time.time() - start)
            # start = time.time()
            if self.neg_mask is None or self.neg_mask.size(1)!=repeat_size: #seems faster than negating half of perturbations
                self.neg_mask = torch.ones((1,repeat_size,1), device=input.device, dtype=input.dtype)
                self.neg_mask[:, repeat_size // 2:, :] *= -1
            # print("negative", time.time() - start)
            # start = time.time()
            # add = (perturbations*self.neg_mask).view(*unperturbed.size()) + unperturbed
            add = torch.addcmul(unperturbed.view(*perturbations.size()),perturbations,self.neg_mask)
            # print("add", time.time() - start)
            # self.clear_noise()
            return add
        return unperturbed

#note: in the interest of speed, the input should be transposed so the order is (parallel , directions,...)
#which is the opposite of linear. This avoids having to transpose it every convolutional layer.
class PerturbedConv2d(nn.Conv2d,Perturbed):

    def __init__(self, in_channels, out_channels, kernel_size, directions, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        Perturbed.__init__(self,directions,bias)

    def allocate_weight(self, low_memory=False):
        if low_memory:
            c_out = self.out_channels
            # c_in = self.in_channels // self.groups
            x, y = self.kernel_size
            self.weight_noise_data = torch.empty((self.directions - 1) * c_out + self.weight.nelement(),
                                                 device=self.weight.device, dtype=self.weight.dtype)

            self.weight_noise = torch.as_strided(self.weight_noise_data, (self.directions * c_out,)+self.weight.size()[1:],
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
            inp_view = input.view(-1, self.directions * self.in_channels, *input.size()[2:])
            print("unperturbed time", time.time() - start)
            start = time.time()
            perturbations = F.conv2d(inp_view, self.weight_noise.view(-1,*self.weight.size()[1:]), bias_noise, self.stride,
                                     padding, self.dilation, self.groups * self.directions)
            print("perturbed time", time.time() - start)
            start = time.time()
            repeat_size = inp_view.size(0)

            if self.neg_mask is None or self.neg_mask.size(0)!=repeat_size:
                self.neg_mask = torch.ones((repeat_size, 1, 1, 1), device=input.device, dtype=input.dtype)
                self.neg_mask[repeat_size // 2:] *= -1
            add = torch.addcmul(unperturbed.view(*perturbations.size()), perturbations, self.neg_mask).view(*unperturbed.size())
            print("add time", time.time() - start)
            return add
        return unperturbed


class PerturbedModel():

    def __init__(self, model, directions):
        self.model = model
        self.directions = directions
        self.perturbed_layers = []

        def get_perturbed_layers(m):
            if isinstance(m,Perturbed):
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
        def ret(*args,**kwargs):
            for l in self.perturbed_layers:
                getattr(l,name)(*args,**kwargs)
        return ret



def print_all_torch_tensors():
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.nelement())
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
    inp_transposed = inp.permute([1,0,2]).contiguous()
    linear_weight = torch.as_strided(base,(k,l,m),(1,m,1))
    print(linear_weight.is_contiguous())
    start = time.time()
    linear_weight_regular = linear_weight.clone().view(linear_weight.size())
    print("clone time",time.time() - start)

    for _ in range(10):
        start = time.time()
        expected = torch.bmm(linear_weight_regular,inp)
        print("regular", time.time()-start)

    for _ in range(10):
        start = time.time()
        # res = torch.einsum('ijk,ikl->ijl', [linear_weight, inp])
        res = torch.bmm(linear_weight, inp_transposed.permute([1,0,2]))
        print("overlap", time.time()-start)
    print(torch.allclose(res,expected))

def speed_test_conv(device="cpu"):

    h = 100
    w = 100
    c_in = 16
    c_out = 32
    d = 512
    p = 2
    x = 3
    y = 3
    offset = 100
    weight_size = c_out * c_in * x * y
    bias = torch.ones(())
    base = torch.empty(weight_size + d * offset * c_out, device=device)
    inp = torch.empty((d, p, c_in, h, w), device=device)
    base.normal_()
    inp.normal_()
    conv_weight = torch.as_strided(base, (d,c_out, c_in, x, y), (c_out,1,c_out * x * y,c_out * y,c_out))

    start = time.time()
    # conv_weight_regular = conv_weight.clone()
    conv_weight_contiguous = conv_weight.view(d * c_out,c_in,x,y).contiguous()
    print("clone time",time.time() - start)
    print_all_torch_tensors()
    for _ in range(10):
        start = time.time()
        inp_view = inp.permute([1,0,2,3,4]).contiguous().view(p,d*c_in,h,w)
        expected = F.conv2d(inp_view, weight=conv_weight.view(d * c_out,c_in,x,y), bias=None, stride=1, groups=d )
        print("regular", time.time()-start)
    print_all_torch_tensors()
    time.sleep(1000)
    for _ in range(10):
        inp_permuted = inp.permute([1,0,2,3,4]).contiguous()
        inp_view = inp_permuted.view(p,d*c_in,h,w).contiguous()
        start = time.time()
        res = F.conv2d(inp_view, weight=conv_weight_contiguous, bias=None, stride=1, groups=d )
        print("overlap", time.time()-start)

    print(torch.allclose(res,expected))



if __name__ == "__main__":
    with torch.no_grad():

        # speed_test_linear("cuda")
        speed_test_conv("cuda")