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

    #options["direct"]=True for direct
    def __init__(self, directions, antithetic=False, options={}):
        self.directions = directions
        self.perturbed_flag = False
        self.noise_scale = None
        self.seed = None
        self.antithetic = antithetic
        self.options = options
        self.free_memory()

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
        if self.options.get("direct"):
            if self.antithetic:
                self.weight_noise[:self.directions // 2].normal_(std=self.noise_scale, generator=gen)
                self.weight_noise[self.directions // 2:] = -self.weight_noise[:self.directions // 2]
                self.weight_noise += self.weight
                if self.bias is not None:
                    self.bias_noise[:self.directions // 2].normal_(std=self.noise_scale, generator=gen)
                    self.bias_noise[self.directions // 2:] = -self.bias_noise[:self.directions // 2]
                    self.bias_noise += self.bias
            else:
                self.weight_noise.normal_(std=self.noise_scale, generator=gen)
                self.weight_noise += self.weight
                if self.bias is not None:
                    self.bias_noise.normal_(std=self.noise_scale, generator=gen)
                    self.bias_noise += self.bias
        else:
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

    def weight_grad(self, weights):
        if self.options.get("direct"):
            if self.antithetic:
                weight_noise = self.weight_noise[self.directions // 2:] - self.weight
            else:
                weight_noise = self.weight_noise - self.weight
        else:
            weight_noise = self.weight_noise
        return (weights @ weight_noise.view(self.directions, -1)).view(*self.weight.size())

    def update(self, weights, l1=0, l2=0):
        grad = self.weight_grad(weights)
        if l1:
            grad += l1 * torch.sign(self.weight)
        if l2:
            grad += l2 * self.weight
        self.weight.grad = grad if self.weight.grad is None else self.weight.grad + grad
        if self.bias is not None:
            if self.options.get("direct"):
                if self.antithetic:
                    bias_noise = self.bias_noise[self.directions // 2:] - self.bias
                else:
                    bias_noise = self.bias_noise - self.bias
            else:
                bias_noise = self.bias_noise
            bias_grad = (weights @ bias_noise.view(self.directions, -1)).view(*self.bias.size())
            self.bias.grad = bias_grad if self.bias.grad is None else self.bias.grad + bias_grad
            return grad, bias_grad
        return grad



class Permuted(Perturbed):
    #option["allow_repeats"] for faster sampling, as pytorch currently doesn't have parallel sampling for permutations.
    #it allows an input or output to be sampled with replacement.
    def __init__(self,  in_degree, out_degree, directions, antithetic=False, options={}, permutation="auto",
                 in_sparsity=0, out_sparsity=0):
        Perturbed.__init__(self, directions, antithetic, options)
        if permutation == "auto":
            if in_sparsity:
                if out_sparsity:
                    permutation = "both"
                else:
                    permutation = "in"
            elif out_sparsity:
                permutation = "out"
            elif 1 < in_degree < 32 and 1 < out_degree < 32:
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
            raise NotImplementedError("Permutation setting not recognized")
        self.in_degree = in_degree
        self.out_degree = out_degree
        self.in_sparsity = max(1, int(in_sparsity * self.in_degree)) if isinstance(in_sparsity, float) else in_sparsity
        if self.in_sparsity:
            self.permute_inputs = True
        else:
            self.in_sparsity = self.in_degree
        self.out_sparsity = max(1, int(out_sparsity * self.out_degree)) if isinstance(out_sparsity, float) else out_sparsity
        if self.out_sparsity:
            self.permute_outputs = True
        else:
            self.out_sparsity = self.out_degree
        if options.get("combined") and self.permute_inputs and self.permute_outputs:
            raise NotImplementedError("Can't do combined multiplication with both input and output permutations")

    def allocate_weight(self):
        self.weight_noise = torch.empty(self.out_sparsity, self.in_sparsity, *self.weight.shape[2:],
                                        device=self.weight.device, dtype=self.weight.dtype)

    def free_memory(self):
        Perturbed.free_memory(self)
        self.input_permutations = None
        self.output_permutations = None

    def set_noise(self, noise_scale=None):
        if self.seed is None:
            self.set_seed()
        if noise_scale is not None:
            self.set_noise_scale(noise_scale)
        if self.weight_noise is None:
            self.allocate_memory()
        gen = torch.cuda.manual_seed(self.seed)
        rescale = (self.out_degree * self.in_degree / self.out_sparsity / self.in_sparsity) ** .5
        self.weight_noise.normal_(std=self.noise_scale * rescale, generator=gen)
        if self.bias is not None:
            self.bias_noise.normal_(std=self.noise_scale, generator=gen)
        if self.options.get("allow_repeats"):
            if self.permute_outputs:
                self.output_permutations = torch.randint(self.out_degree,
                     (self.directions, self.out_degree), dtype=torch.short, generator=gen, device=self.weight.device)
            if self.permute_inputs:
                self.input_permutations = torch.randint(self.in_degree,
                     (self.directions, self.in_sparsity), dtype=torch.short, generator=gen, device=self.weight.device)
        else:
            gen = torch.manual_seed(self.seed)
            if self.permute_outputs:
                self.output_permutations = torch.empty(self.directions, self.out_degree, dtype=torch.short)
                for i in range(self.directions):
                    torch.randperm(self.out_degree, out=self.output_permutations[i], generator=gen)
                self.output_permutations = self.output_permutations.to(self.weight.device)
            if self.permute_inputs:
                self.input_permutations = torch.empty(self.directions, self.in_degree, dtype=torch.short)
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

    def get_permutation_weights(self, weights):
        if self.permute_inputs and self.permute_outputs:
            inverse_out = torch.argsort(self.output_permutations, dim=1)[:,:self.out_sparsity]
            sp_size = self.out_sparsity * self.in_sparsity
            mat_size = self.out_degree * self.in_degree
            permutations = self.input_permutations.view(self.directions, 1, self.in_sparsity) + \
                           (inverse_out * self.in_sparsity).view(self.directions, self.out_sparsity, 1)
            ar = torch.arange(sp_size, device=self.weight.device)
            permutations_1d = permutations.view(self.directions, sp_size) + ar * mat_size
            weighted_perms = torch.zeros((sp_size, mat_size), device=self.weight.device, dtype=self.weight.dtype)
            weighted_perms.put_(permutations_1d, weights.view(-1, 1).expand(self.directions, sp_size), accumulate=True)
            weighted_perms = weighted_perms.t()
        elif self.permute_inputs:
            weighted_perms = torch.zeros((self.in_sparsity, self.in_degree), device=self.weight.device, dtype=self.weight.dtype)
            ar = torch.arange(self.in_sparsity, device=self.weight.device)
            input_permutations_1d = self.input_permutations + ar * self.in_degree
            weighted_perms.put_(input_permutations_1d, weights.view(-1,1).expand(self.directions, self.in_sparsity), accumulate=True)
            # for i in range(self.in_sparsity):
            #     weighted_perms[i].put_(self.input_permutations[:,i], weights, accumulate=True)
        else:
            weighted_perms = torch.zeros((self.out_degree, self.out_degree), device=self.weight.device, dtype=self.weight.dtype)
            ar = torch.arange(self.out_degree, device=self.weight.device)
            output_permutations_1d = self.output_permutations + ar * self.out_degree
            weighted_perms.put_(output_permutations_1d,weights.view(-1,1).expand(self.directions, self.out_degree), accumulate=True)
            weighted_perms = weighted_perms[:, :self.out_sparsity]
        return weighted_perms

class Synthetic(Perturbed):

    def __init__(self, in_degree, out_degree, directions, antithetic=False, options={},
                 flip="auto", in_sparsity=0, out_sparsity=0):
        Perturbed.__init__(self, directions, antithetic, options)
        if flip == "auto":
            if 1 < in_degree < 32 and 1 < out_degree < 32:
                flip = "both"
            elif in_degree > out_degree:
                flip = "in"
            else:
                flip = "out"
        if flip == "both":
            self.flip_inputs = True
            self.flip_outputs = True
        elif flip == "in":
            self.flip_inputs = True
            self.flip_outputs = False
        elif flip == "out":
            self.flip_inputs = False
            self.flip_outputs = True
        else:
            raise NotImplementedError("Flip setting not recognized")
        self.in_degree = in_degree
        self.out_degree = out_degree
        if in_sparsity or out_sparsity:
            raise NotImplementedError("Sparsity is not efficient for synthetic sampling, use permuted sampling instead")
        self.in_sparsity = self.in_degree
        self.out_sparsity = self.out_degree
        if options.get("combined") and self.flip_inputs and self.flip_outputs:
            raise NotImplementedError("Can't do combined multiplication with both input and output flips")

    def allocate_weight(self):
        self.weight_noise = torch.empty(self.out_sparsity, self.in_sparsity, *self.weight.shape[2:],
                                        device=self.weight.device, dtype=self.weight.dtype)

    def free_memory(self):
        Perturbed.free_memory(self)
        self.input_flips = None
        self.output_flips = None


    def set_noise(self, noise_scale=None):
        if self.seed is None:
            self.set_seed()
        if noise_scale is not None:
            self.set_noise_scale(noise_scale)
        if self.weight_noise is None:
            self.allocate_memory()
        gen = torch.cuda.manual_seed(self.seed)
        rescale = (self.out_degree * self.in_degree / self.out_sparsity / self.in_sparsity) ** .5
        self.weight_noise.normal_(std=self.noise_scale * rescale, generator=gen)
        if self.bias is not None:
            self.bias_noise.normal_(std=self.noise_scale, generator=gen)
        if self.flip_outputs:
            self.output_flips = 2 * torch.randint(2, size=(self.directions, self.out_degree), dtype=torch.int8,
                                                  generator=gen, device=self.weight.device) - 1
        if self.flip_inputs:
            self.input_flips = 2 * torch.randint(2, size=(self.directions, self.in_degree), dtype=torch.int8,
                                                 generator=gen, device=self.weight.device) - 1

    def get_flipped_weights(self, weights):
        if self.flip_inputs and self.flip_outputs:
            return torch.einsum("d,da,db->ab", weights, self.output_flips, self.input_flips * weights.unsqueeze(1))
        elif self.flip_inputs:
            return (self.input_flips * weights.unsqueeze(1)).sum(dim=0)
        else:
            return (self.output_flips * weights.unsqueeze(1)).sum(dim=0)

    def weight_grad(self, weights):
        if self.flip_inputs and self.flip_outputs:
            flipped_weights = torch.einsum("d,da,db->ab", weights, self.output_flips, self.input_flips * weights.unsqueeze(1))
            for _ in range(len(self.weight.shape) - 2):
                flipped_weights = flipped_weights.unsqueeze(-1)
        elif self.flip_inputs:
            flipped_weights = (self.input_flips * weights.unsqueeze(1)).sum(dim=0).view(1, -1, *[1] * (len(self.weight.shape) - 2))
        else:
            flipped_weights = (self.output_flips * weights.unsqueeze(1)).sum(dim=0).view(-1, *[1] * (len(self.weight.shape) - 1))
        return flipped_weights * self.weight_noise

class PerturbedLinear(nn.Linear, Perturbed):

    def __init__(self, in_features, out_features, directions, bias=True, antithetic=False, options={}):
        nn.Linear.__init__(self, in_features, out_features, bias)
        Perturbed.__init__(self, directions, antithetic, options)

    def forward(self, input):
        if self.options.get("direct") and self.perturbed_flag:
            input_by_direction = input.view(-1, self.directions, self.in_features).permute([1, 0, 2])
            if self.bias is not None:
                return torch.baddbmm(self.bias_noise.view(self.directions, 1, self.out_features),
                                     input_by_direction,
                                     self.weight_noise.permute([0, 2, 1])).permute([1, 0, 2])
            else:
                return torch.bmm(input_by_direction,
                                 self.weight_noise.permute([0, 2, 1])).permute([1, 0, 2])
        else:
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
                if self.antithetic:
                    perturbations[(repeat_size + 1) // 2:] *= -1
                add = (unperturbed.view_as(perturbations) + perturbations).view_as(unperturbed)
                return add
            return unperturbed


class PerturbedConv2d(nn.Conv2d, Perturbed):

    def __init__(self, in_channels, out_channels, kernel_size, directions, stride=1, 
                 padding=0, dilation=1, groups=1, 
                 bias=True, antithetic=False, options={}, padding_mode='zeros'):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        Perturbed.__init__(self, directions, antithetic, options)



    #based on https://github.com/pytorch/pytorch/issues/17983
    def forward(self, input):
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._padding_repeated_twice, mode=self.padding_mode)
            padding = module_utils._pair(0)
        else:
            padding = self.padding
        if self.options.get("direct") and self.perturbed_flag:
            bias_noise = self.bias_noise.view(-1) if self.bias is not None else None
            input_view = input.view(-1, self.directions * self.in_channels, *input.size()[-2:])
            repeat_size = input_view.size(0)
            torch.cuda.synchronize()
            perturbations = F.conv2d(input_view, self.weight_noise.view(-1, *self.weight.size()[1:]), None, self.stride,
                                     padding, self.dilation, self.groups * self.directions)
            torch.cuda.synchronize()
            return perturbations.view(repeat_size, self.directions, self.out_channels, *perturbations.shape[-2:])
        else:
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
                if self.antithetic:
                    perturbations[(repeat_size + 1) // 2:] *= -1
                add = unperturbed + perturbations.view_as(unperturbed)
                return add
            return unperturbed

class PerturbedAffine(nn.Module, Perturbed):

    def __init__(self, normalized_shape, directions, antithetic=False, options={"direct": True}):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        if not options.get("direct"):
            raise NotImplementedError("Only direct is implemented")
        Perturbed.__init__(self, directions, antithetic=antithetic, options=options)

    def forward(self, input):
        if self.perturbed_flag:
            weight = self.weight_noise
            bias = self.bias_noise
        else:
            weight = self.weight
            bias = self.bias
        return input * weight + bias


class PermutedLinear(nn.Linear, Permuted):

    def __init__(self, in_features, out_features, directions, bias=True, antithetic=False, options={},
                 permutation="auto", in_sparsity=0, out_sparsity=0):
        nn.Linear.__init__(self, in_features, out_features, bias)
        Permuted.__init__(self, in_features, out_features, directions,
                          antithetic=antithetic, options=options,
                          permutation=permutation, in_sparsity=in_sparsity, out_sparsity=out_sparsity)

    def forward(self, input):
        if self.options.get("combined") and self.perturbed_flag:
            if self.permute_inputs:
                input_view = input.view(-1, self.directions * self.in_features)
                permuted_input = self.apply_input_permutation(input_view).view(-1, self.in_sparsity)
                combined_input = torch.cat([input.view(-1, self.in_features), permuted_input], dim=1)
                combined_weights = torch.cat([self.weight, self.weight_noise], dim=1)
                combined_bias = self.bias
                return F.linear(combined_input, combined_weights, combined_bias)
            elif self.permute_outputs:
                input_view = input.view(-1, self.in_features)
                repeat_size = input_view.size(0) // self.directions
                combined_weights = torch.cat([self.weight, self.weight_noise], dim=0)
                if self.out_sparsity < self.out_degree or True:
                    combined_output = torch.zeros((repeat_size, self.directions * self.out_features, 2), device=input.device, dtype=input.dtype)
                    torch.mm(input_view, combined_weights.t(), out=combined_output.view(-1, 2 * self.out_features)[:, :self.out_features + self.out_sparsity])
                else:
                    combined_output = torch.mm(input_view, combined_weights.t())


                perturbations = combined_output[:, :, 1].view(repeat_size, self.directions * self.out_features)
                permuted_output = self.apply_output_permutation(perturbations).view(repeat_size,
                                                                                    self.directions, self.out_features)
                if self.bias is not None:
                    permuted_output += self.bias_noise
                if self.antithetic:
                    permuted_output[(repeat_size + 1) // 2:] *= -1
                add = combined_output[:, :, 0].view(-1, self.out_features) + permuted_output.view(-1, self.out_features)
                if self.bias is not None:
                    add += self.bias
                return add
        else:
            unperturbed = F.linear(input, self.weight, self.bias)
            if self.perturbed_flag:
                input_view = input.view(-1, self.directions * self.in_features)
                repeat_size = input_view.size(0)
                permuted_input = self.apply_input_permutation(input_view).view(-1, self.in_sparsity)
                if self.out_sparsity < self.out_degree:
                    perturbations = torch.zeros_like(unperturbed)
                    torch.mm(permuted_input, self.weight_noise.t(), out=perturbations.view(-1, self.out_features)[:, :self.out_sparsity])
                    perturbations = perturbations.view(repeat_size,
                                                              self.directions * self.out_features)
                else:
                    perturbations = torch.mm(permuted_input, self.weight_noise.t()).view(repeat_size,
                                                                                         self.directions * self.out_features)
                permuted_output = self.apply_output_permutation(perturbations).view(repeat_size, self.directions, self.out_features)
                if self.bias is not None:
                    permuted_output += self.bias_noise
                if self.antithetic:
                    permuted_output[(repeat_size + 1) // 2:] *= -1
                add = unperturbed + permuted_output.view_as(unperturbed)
                return add
            return unperturbed


    def weight_grad(self, weights):
        permutation_weights = self.get_permutation_weights(weights)
        if self.permute_inputs and self.permute_outputs:
            return torch.mm(permutation_weights, self.weight_noise.view(-1, 1)).reshape(self.out_features, self.in_features)
        elif self.permute_inputs:
            return torch.mm(self.weight_noise, permutation_weights)
        else:
            return torch.mm(permutation_weights, self.weight_noise)


class PermutedConv2d(nn.Conv2d, Permuted):

    def __init__(self, in_channels, out_channels, kernel_size, directions, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, antithetic=False, options={},
                 padding_mode='zeros', permutation="auto", in_sparsity=0, out_sparsity=0):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        Permuted.__init__(self, in_channels, out_channels, directions,
                          antithetic=antithetic, options=options,
                          permutation=permutation, in_sparsity=in_sparsity, out_sparsity=out_sparsity)

    def forward(self, input):
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._padding_repeated_twice, mode=self.padding_mode)
            padding = module_utils._pair(0)
        else:
            padding = self.padding
        if self.options.get("combined") and self.perturbed_flag:
            if self.permute_inputs:
                input_dims = input.shape[-2:]
                input_view = input.view(-1, self.directions * self.in_channels, *input_dims)
                repeat_size = input_view.size(0)
                permuted_input = self.apply_input_permutation(input_view).view(-1, self.in_sparsity, *input_dims)
                if self.antithetic:
                    permuted_input.view(repeat_size, self.directions, self.in_sparsity, *input_dims)[(repeat_size + 1) // 2:] *= -1
                combined_input = torch.cat([input.view(-1, self.in_channels, *input_dims), permuted_input], dim=1)
                combined_weights = torch.cat([self.weight, self.weight_noise], dim=1)
                result = F.conv2d(combined_input, combined_weights, self.bias, self.stride,
                         padding, self.dilation, self.groups)
                if self.bias is not None:
                    if self.antithetic:
                        result.view(repeat_size, self.directions, *result.shape[1:])[:(repeat_size + 1) // 2] += self.bias_noise.unsqueeze(-1).unsqueeze(-1)
                        result.view(repeat_size, self.directions, *result.shape[1:])[(repeat_size + 1) // 2:] -= self.bias_noise.unsqueeze(-1).unsqueeze(-1)
                    else:
                        return result + self.bias_noise.unsqueeze(-1).unsqueeze(-1)
                return result
        else:
            unperturbed = F.conv2d(input, self.weight, self.bias, self.stride,
                                   padding, self.dilation, self.groups)
            if self.perturbed_flag:
                input_dims = input.shape[-2:]
                output_dims = unperturbed.shape[-2:]
                input_view = input.view(-1, self.directions * self.in_channels, *input_dims)
                repeat_size = input_view.size(0)

                permuted_input = self.apply_input_permutation(input_view).view(-1, self.in_sparsity, *input_dims)
                if self.out_sparsity < self.out_degree:
                    perturbations = torch.zeros_like(unperturbed)
                    perturbations.view(-1, self.out_channels, *output_dims)[:, :self.out_sparsity] = \
                        F.conv2d(permuted_input, self.weight_noise, None, self.stride, padding, self.dilation, self.groups)
                else:
                    perturbations = F.conv2d(permuted_input, self.weight_noise, None, self.stride,
                                             padding, self.dilation, self.groups)


                # permuted_input = self.apply_input_permutation(input_view).view_as(input)
                # print(input.shape,permuted_input.shape,input_view.shape)
                # perturbations = F.conv2d(permuted_input, self.weight_noise, None, self.stride,
                #                          padding, self.dilation, self.groups)
                permuted_output = self.apply_output_permutation(perturbations.view(repeat_size, self.directions * self.out_channels,
                                                                         *perturbations.shape[-2:])).view_as(unperturbed)
                if self.bias is not None:
                    permuted_output += self.bias_noise.unsqueeze(-1).unsqueeze(-1)
                if self.antithetic:
                    permuted_output.view(repeat_size, -1)[(repeat_size + 1) // 2:] *= -1
                add = unperturbed + permuted_output
                return add
            return unperturbed

    def weight_grad(self, weights):
        permutation_weights = self.get_permutation_weights(weights)
        if self.permute_inputs and self.permute_outputs:
            sp_size = self.out_sparsity * self.in_sparsity
            return torch.mm(permutation_weights, self.weight_noise.view(sp_size, -1)).view_as(self.weight)
        elif self.permute_inputs:
            return torch.mm(self.weight_noise.permute([0,2,3,1]).contiguous().view(-1,self.in_sparsity),
                                        permutation_weights).view(self.out_channels, *self.kernel_size,self.in_channels).permute([0,3,1,2])
        else:
            return torch.mm(permutation_weights, self.weight_noise.view(self.out_sparsity,-1)).view_as(self.weight)

class SyntheticLinear(nn.Linear, Synthetic):

    def __init__(self, in_features, out_features, directions, bias=True, antithetic=False, options={},
                 flip="auto", in_sparsity=0, out_sparsity=0):
        nn.Linear.__init__(self, in_features, out_features, bias)
        Synthetic.__init__(self, in_features, out_features, directions,
                           antithetic=antithetic, options=options,
                           flip=flip, in_sparsity=in_sparsity, out_sparsity=out_sparsity)


    def forward(self, input):
        if self.options.get("combined") and self.perturbed_flag:
            pass
        else:
            unperturbed = F.linear(input, self.weight, self.bias)
            if self.perturbed_flag:
                input_view = input.view(-1, self.directions, self.in_features)
                repeat_size = input_view.size(0)
                flipped_input = input_view * self.input_flips if self.flip_inputs else input_view
                perturbations = flipped_input @ self.weight_noise.t()
                flipped_output = (perturbations * self.output_flips if self.flip_outputs else perturbations)
                if self.bias is not None:
                    flipped_output += self.bias_noise
                if self.antithetic:
                    flipped_output[(repeat_size + 1) // 2:] *= -1
                add = unperturbed + flipped_output.view_as(unperturbed)
                return add
            return unperturbed


class SyntheticConv2d(nn.Conv2d, Synthetic):

    def __init__(self, in_channels, out_channels, kernel_size, directions, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, antithetic=False, options={},
                 padding_mode='zeros', flip="auto", in_sparsity=0, out_sparsity=0):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        Synthetic.__init__(self, in_channels, out_channels, directions,
                           antithetic=antithetic, options=options,
                           flip=flip, in_sparsity=in_sparsity, out_sparsity=out_sparsity)

    def forward(self, input):
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._padding_repeated_twice, mode=self.padding_mode)
            padding = module_utils._pair(0)
        else:
            padding = self.padding
        if self.options.get("combined") and self.perturbed_flag:
            pass
        else:
            unperturbed = F.conv2d(input, self.weight, self.bias, self.stride,
                                   padding, self.dilation, self.groups)
            if self.perturbed_flag:
                input_dims = input.shape[-2:]
                output_dims = unperturbed.shape[-2:]
                input_view = input.view(-1, self.directions, self.in_channels, *input_dims)
                repeat_size = input_view.size(0)
                flipped_input = input_view * self.input_flips.unsqueeze(2).unsqueeze(3) if self.flip_inputs else input_view
                perturbations = F.conv2d(flipped_input.view(-1, self.in_sparsity, *input_dims), self.weight_noise, None, self.stride,
                                             padding, self.dilation, self.groups)
                flipped_output = (perturbations * self.output_flips.unsqueeze(2).unsqueeze(3) if self.flip_outputs else perturbations)
                if self.bias is not None:
                    flipped_output += self.bias_noise.unsqueeze(-1).unsqueeze(-1)
                if self.antithetic:
                    flipped_output.view(repeat_size, -1)[(repeat_size + 1) // 2:] *= -1
                add = unperturbed + flipped_output
                return add
            return unperturbed

