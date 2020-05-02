import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.modules.utils as module_utils
import time
import random


class PerturbedModel:

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


class Perturbed:

    def __init__(self, directions, bias=True):
        self.directions = directions
        self.perturbed_flag = False
        self.noise_scale = None
        self.weight_noise = None
        self.seed = None
        if bias:
            self.bias_noise = None

    def set_noise_scale(self, noise_scale):
        self.noise_scale = noise_scale

    def set_noise(self, noise_scale=None):
        if self.seed is None:
            self.set_seed()
        if noise_scale is not None:
            self.set_noise_scale(noise_scale)
        if self.weight_noise is None:
            self.allocate_memory()
        gen = torch.cuda.manual_seed(self.seed)
        self.weight_noise.normal_(std=self.noise_scale, generator=gen)
        if self.bias is not None:
            self.bias_noise.normal_(std=self.noise_scale, generator=gen)

    def set_seed(self, seed=None):
        self.seed = seed if seed is not None else random.randrange(100000000)

    def allocate_weight(self):
        self.weight_noise = torch.empty((self.directions, )+self.weight.size(),
                                             device=self.weight.device, dtype=self.weight.dtype)

    def allocate_bias(self):
        if self.bias is not None:
            self.bias_noise = torch.empty((self.directions, )+self.bias.size(),
                                               device=self.bias.device, dtype=self.bias.dtype)

    def allocate_memory(self):
        self.allocate_weight()
        if self.bias is not None:
            self.allocate_bias()

    def free_memory(self):
        self.weight_noise = None
        if self.bias is not None:
            self.bias_noise = None

    def set_grad(self, weights):
        self.weight.grad = (weights @ self.weight_noise.view(self.directions, -1)).view(*self.weight.size())
        if self.bias is not None:
            self.bias.grad = weights @ self.bias_noise
        self.free_memory()



class Permuted(Perturbed):

    def __init__(self,  in_degree, out_degree, directions, bias=True, permutation="auto", in_sparsity=0, out_sparsity=0):
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
        self.in_sparsity = int(in_sparsity * self.in_degree) if isinstance(in_sparsity, float) else in_sparsity
        if self.in_sparsity:
            self.permute_inputs = True
        else:
            self.in_sparsity = self.in_degree
        self.out_sparsity = int(out_sparsity * self.out_degree) if isinstance(out_sparsity, float) else out_sparsity
        if self.out_sparsity:
            self.permute_outputs = True
        else:
            self.out_sparsity = self.out_degree

    def allocate_weight(self):
        self.weight_noise = torch.empty(self.out_sparsity, self.in_sparsity, *self.weight.shape[2:],
                                        device=self.weight.device, dtype=self.weight.dtype)

    def free_memory(self):
        Perturbed.free_memory(self)
        self.input_permutations = None
        self.output_permutations = None

    def set_noise(self, noise_scale=None):
        Perturbed.set_noise(self, noise_scale)
        gen = torch.manual_seed(self.seed)
        if self.permute_outputs:
            self.output_permutations = torch.empty(self.directions, self.out_degree, dtype=torch.long)
            for i in range(self.directions):
                torch.randperm(self.out_degree, out=self.output_permutations[i], generator=gen)
            self.output_permutations = self.output_permutations.to(self.weight.device)
        if self.permute_inputs:
            self.input_permutations = torch.empty(self.directions, self.in_degree, dtype=torch.long)
            for i in range(self.directions):
                torch.randperm(self.in_degree, out=self.input_permutations[i], generator=gen)
            self.input_permutations = self.input_permutations[:, :self.in_sparsity].to(self.weight.device)

    def apply_input_permutation(self, input):
        if self.permute_inputs:
            input_permutations_1d = (self.input_permutations + (torch.arange(self.directions, device=input.device) * self.in_degree).unsqueeze(1)).flatten()
            return torch.index_select(input, 1, input_permutations_1d)
        return input


    def apply_output_permutation(self, output):
        if self.permute_outputs:
            output_permutations_1d = (self.output_permutations + (torch.arange(self.directions, device=output.device) * self.out_degree).unsqueeze(1)).flatten()
            return torch.index_select(output, 1, output_permutations_1d)
        return output

class SparsePerturbed(Perturbed):

    def __init__(self,  in_degree, out_degree, directions, bias=True, sparsity="auto"):
        Perturbed.__init__(self, directions, bias)
        if sparsity == "auto":
            sparsity = max(1, in_degree // directions)
        self.sparsity = sparsity
        self.selections = None
        self.in_degree = in_degree
        self.out_degree = out_degree

    def allocate_weight(self):
        raise NotImplementedError


    def free_memory(self):
        Perturbed.free_memory(self)
        self.selections = None

    def set_noise(self, noise_scale=None):
        if self.seed is None:
            self.set_seed()
        if noise_scale is not None:
            self.set_noise_scale(noise_scale)
        if self.weight_noise is None:
            self.allocate_memory()
        gen = torch.cuda.manual_seed(self.seed)
        self.weight_noise.normal_(std=self.noise_scale * (self.in_degree / self.sparsity) ** .5, generator=gen)
        if self.bias is not None:
            self.bias_noise.normal_(std=self.noise_scale, generator=gen)
        self.selections.random_(self.in_degree, generator=gen)


class PerturbedLinear(nn.Linear, Perturbed):

    def __init__(self, in_features, out_features, directions, bias=True):
        nn.Linear.__init__(self, in_features, out_features, bias)
        Perturbed.__init__(self, directions, bias)

    def forward(self, input):
        unperturbed = F.linear(input, self.weight, self.bias)
        if self.perturbed_flag:
            input_by_direction = input.view(-1, self.directions, self.in_features).permute([1, 0, 2])
            repeat_size = input_by_direction.size(1)
            if self.bias is not None:
                perturbations = torch.baddbmm(self.bias_noise.view(self.directions, 1, self.out_features),
                                              input_by_direction,
                                              self.weight_noise.permute([0, 2, 1])).permute([1, 0, 2])
            else:
                perturbations = torch.bmm(input_by_direction,
                                          self.weight_noise.permute([0, 2, 1])).permute([1, 0, 2])
            perturbations[(repeat_size + 1) // 2:] *= -1
            add = unperturbed + perturbations.view_as(unperturbed)
            return add
        return unperturbed


class PerturbedConv2d(nn.Conv2d, Perturbed):

    def __init__(self, in_channels, out_channels, kernel_size, directions, stride=1, 
                 padding=0, dilation=1, groups=1, 
                 bias=True, padding_mode='zeros'):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        Perturbed.__init__(self, directions, bias)



    #based on https://github.com/pytorch/pytorch/issues/17983
    def forward(self, input):
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._padding_repeated_twice, mode=self.padding_mode)
            padding = module_utils._pair(0)
        else:
            padding = self.padding
        unperturbed = F.conv2d(input, self.weight, self.bias, self.stride,
                               padding, self.dilation, self.groups)

        if self.perturbed_flag:
            bias_noise = self.bias_noise.view(-1) if self.bias is not None else None
            input_view = input.view(-1, self.directions * self.in_channels, *input.size()[-2:])
            repeat_size = input_view.size(0)
            perturbations = F.conv2d(input_view, self.weight_noise.view(-1, *self.weight.size()[1:]), bias_noise, self.stride,
                                     padding, self.dilation, self.groups * self.directions)
            perturbations = perturbations.view(repeat_size, self.directions, self.out_channels,
                                               *perturbations.size()[-2:])
            perturbations[(repeat_size + 1) // 2:] *= -1
            add = unperturbed + perturbations.view_as(unperturbed)
            return add
        return unperturbed


class PermutedLinear(nn.Linear, Permuted):

    def __init__(self, in_features, out_features, directions, bias=True, permutation="auto", in_sparsity=0, out_sparsity=0):
        nn.Linear.__init__(self, in_features, out_features, bias)
        Permuted.__init__(self, in_features, out_features, directions, bias=bias,
                          permutation=permutation, in_sparsity=in_sparsity, out_sparsity=out_sparsity)

    def forward(self, input):
        unperturbed = F.linear(input, self.weight, self.bias)
        if self.perturbed_flag:
            input_view = input.view(-1, self.directions * self.in_features)
            repeat_size = input_view.size(0)
            permuted_input = self.apply_input_permutation(input_view).view(-1, self.in_sparsity)
            if self.out_sparsity < self.out_degree:
                perturbations = torch.zeros_like(unperturbed)
                torch.mm(permuted_input, self.weight_noise.t(),out=perturbations.view(-1,self.out_features)[:self.out_sparsity])
                perturbations = perturbations.view(repeat_size,
                                                          self.directions * self.out_features)
            else:
                perturbations = torch.mm(permuted_input, self.weight_noise.t()).view(repeat_size,
                                                                                     self.directions * self.out_features)
            permuted_output = self.apply_output_permutation(perturbations).view_as(unperturbed)
            if self.bias is not None:
                permuted_output += self.bias_noise
            permuted_output.view(repeat_size, self.directions, self.out_features)[(repeat_size + 1) // 2:] *= -1
            add = unperturbed + permuted_output
            return add
        return unperturbed

    def set_grad(self, weights):
        if self.permute_inputs and self.permute_outputs:
            if self.out_sparsity < self.out_features and self.in_sparsity < self.in_features:
                raise NotImplementedError
                mat_size = self.out_sparsity * self.in_sparsity

            inverse = torch.argsort(self.input_permutations, dim=1)
            mat_size = self.out_features * self.in_features
            permutations = inverse.view(self.directions, 1, self.in_features) + (self.output_permutations * self.in_features).view(self.directions, self.out_features, 1)
            ar = torch.arange(mat_size, device=self.weight.device)
            permutations_1d = permutations.view(self.directions, mat_size) + ar * mat_size
            weighted_perms = torch.zeros((mat_size, mat_size), device=self.weight.device, dtype=self.weight.dtype)
            weighted_perms.put_(permutations_1d, weights.view(-1, 1).expand(self.directions, mat_size), accumulate=True)
            self.weight.grad = torch.mm(weighted_perms, self.weight_noise.view(-1, 1)).reshape(self.out_features, self.in_features)
            # weight_grad = torch.zeros_like(self.weight)
            # for i in range(self.directions):
            #     self.weight.grad += self.weight_noise[self.output_permutations[i]][:,inverse[i]] * weights[i]
        elif self.permute_inputs:
            weighted_perms = torch.zeros((self.in_sparsity, self.in_features), device=self.weight.device, dtype=self.weight.dtype)
            ar = torch.arange(self.in_sparsity, device=self.weight.device)
            input_permutations_1d = self.input_permutations + ar * self.in_features
            weighted_perms.put_(input_permutations_1d, weights.view(-1,1).expand(self.directions,self.in_sparsity), accumulate=True)
            self.weight.grad = torch.mm(self.weight_noise, weighted_perms)
            # weight_grad = torch.zeros_like(self.weight)
            # inverse = torch.argsort(self.input_permutations, dim=1)
            # for i in range(self.directions):
            #     weight_grad += self.weight_noise[:, inverse[i]] * weights[i]
        else:
            weighted_perms = torch.zeros((self.out_features, self.out_features), device=self.weight.device, dtype=self.weight.dtype)
            ar = torch.arange(self.out_features, device=self.weight.device)
            output_permutations_1d = self.output_permutations + ar * self.out_features
            weighted_perms.put_(output_permutations_1d,weights.view(-1,1).expand(self.directions, self.out_features), accumulate=True)
            self.weight.grad = torch.mm(weighted_perms[:, :self.out_sparsity], self.weight_noise)
            # weight_grad = torch.zeros_like(self.weight)
            # for i in range(self.directions):
            #     weight_grad += self.weight_noise[self.output_permutations[i]] * weights[i]
        if self.bias is not None:
            self.bias.grad = weights @ self.bias_noise


class PermutedConv2d(nn.Conv2d, Permuted):

    def __init__(self, in_channels, out_channels, kernel_size, directions, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', permutation="auto", in_sparsity=0, out_sparsity=0):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        Permuted.__init__(self, in_channels, out_channels, directions, bias=bias,
                          permutation=permutation, in_sparsity=in_sparsity, out_sparsity=out_sparsity)
        if in_sparsity or out_sparsity:
            print("Sparse mode not implemented for conv")
            raise NotImplementedError

    def forward(self, input):
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._padding_repeated_twice, mode=self.padding_mode)
            padding = module_utils._pair(0)
        else:
            padding = self.padding
        unperturbed = F.conv2d(input, self.weight, self.bias, self.stride,
                               padding, self.dilation, self.groups)

        if self.perturbed_flag:
            input_view = input.view(-1, self.directions * self.in_channels, *input.shape[-2:])
            repeat_size = input_view.size(0)
            permuted_input = self.apply_input_permutation(input_view).view_as(input)
            perturbations = F.conv2d(permuted_input, self.weight_noise, None, self.stride,
                                     padding, self.dilation, self.groups)
            permuted_output = self.apply_output_permutation(perturbations.view(repeat_size, self.directions * self.out_channels,
                                                                     *perturbations.shape[-2:])).view_as(unperturbed)
            if self.bias is not None:
                permuted_output += self.bias_noise.unsqueeze(-1).unsqueeze(-1)
            permuted_output.view(repeat_size, -1)[(repeat_size + 1) // 2:] *= -1
            add = unperturbed + permuted_output
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
            # weight_grad = torch.zeros_like(self.weight)
            # for i in range(self.directions):
            #     weight_grad += self.weight_noise[self.output_permutations[i]][:,inverse[i]] * weights[i]
            # print(torch.allclose(self.weight.grad, weight_grad))
        elif self.permute_inputs:

            weighted_perms = torch.zeros((self.in_channels, self.in_channels), device=self.weight.device, dtype=self.weight.dtype)
            ar = torch.arange(self.in_channels, device=self.weight.device)
            input_permutations_1d = self.input_permutations + ar * self.in_channels
            weighted_perms.put_(input_permutations_1d, weights.view(-1,1).expand(self.directions,self.in_channels), accumulate=True)
            self.weight.grad = torch.mm(self.weight_noise.permute([0,2,3,1]).contiguous().view(-1,self.in_channels),
                                        weighted_perms).view(self.out_channels, *self.kernel_size,self.in_channels).permute([0,3,1,2])
        else:
            weighted_perms = torch.zeros((self.out_channels, self.out_channels), device=self.weight.device, dtype=self.weight.dtype)
            ar = torch.arange(self.out_channels, device=self.weight.device)
            output_permutations_1d = self.output_permutations + ar * self.out_channels
            weighted_perms.put_(output_permutations_1d,weights.view(-1,1).expand(self.directions, self.out_channels), accumulate=True)
            self.weight.grad = torch.mm(weighted_perms, self.weight_noise.view(self.out_channels,-1)).view_as(self.weight)
        if self.bias is not None:
            self.bias.grad = weights @ self.bias_noise


        self.free_memory()


class SparsePerturbedLinear(nn.Linear, SparsePerturbed):

    def __init__(self, in_features, out_features, directions, bias=True, sparsity="auto"):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SparsePerturbed.__init__(self, in_features, out_features, directions, bias=True, sparsity=sparsity)

    def allocate_weight(self):
        self.weight_noise = torch.empty((self.directions, self.out_degree, self.sparsity), device=self.weight.device, dtype=self.weight.dtype)
        self.selections = torch.empty((self.directions, self.sparsity), device=self.weight.device, dtype=torch.long)

    def forward(self, input):
        unperturbed = F.linear(input, self.weight, self.bias)
        if self.perturbed_flag:
            selected_input = torch.gather(input, 1, self.selections)
            perturbations = torch.bmm(selected_input.view(self.directions, 1, self.sparsity),
                                      self.weight_noise.permute([0, 2, 1])).view_as(unperturbed)
            add = unperturbed + perturbations
            if self.bias is not None:
                add += self.bias_noise
            return add
        return unperturbed

    def set_grad(self, weights):
        raise NotImplementedError

