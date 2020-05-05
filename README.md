# NESgpu
Optimized PyTorch GPU implementation of Natural Evolution Strategies/Augmented Random Search

NESgpu defines the Perturbed class, which streamlines the noise sampling and weight update process for NES and similar training algorithms. It directly allows for batch training, obtaining performance gains of **2-100x** (see below) over non-batched implementations.

## Currently Defined Layers

PerturbedLinear/PerturbedConv2d is the "default" implementation which adds Gaussian noise to its weights. Large amounts of memory are needed to store the Gaussian noise.

SparsePerturbedLinear is a test demonstration of sparse Gaussian noise. Some basic tests on MNIST show some bad effects with extreme sparsity.

PermutedLinear/PermutedConv2d reuses a single Gaussian noise vector in a shuffled manner for all population members. It is fast and has a low memory overhead. It introduces correlation between members, but in a manner which should have no practical impact. It also has sparse options.


See https://github.com/jamesal1/NESgpu/wiki/Explanation-of-Design-Decisions for further discussion of the optimizations.

## Forward Pass Performance
Experiments were done on a GeForce 1080 Ti, with a batch size of 1024. The reported times are the median time of a 100 runs. The format for the column headers are (out dimension, in dimension) for the dense layers and (out_channels, in channels, filter_size, input_size) for the convolutional layers. 

The base time is the time for the layer to compute a forward pass in evaluation mode; it represents an upper bound on the performance of the layer in training mode.

The naive time is calculated by running the layer in evaluation mode with a batch size of 1, repeated 1024 times, to simulate the performance of an implementation that samples each batch element sequentially.
### Time (ms, lower is better)
|                                |   (256, 256) |   (512, 512) |   (1024, 1024) |
|:-------------------------------|-------------:|-------------:|---------------:|
| PerturbedLinear                |        0.938 |        3.231 |         12.984 |
| SparsePerturbedLinear (k=1)    |        0.223 |        0.377 |          0.607 |
| SparsePerturbedLinear (k=2)    |        0.302 |        0.505 |          1.097 |
| PermutedLinear                 |        0.216 |        0.321 |          0.786 |
| PermutedLinear(in_sparsity=.5) |        0.214 |        0.269 |          0.618 |
| PermutedLinear(in_sparsity=.1) |        0.216 |        0.276 |          0.484 |
| Base                           |        0.063 |        0.109 |          0.289 |
| Naive                          |       24.484 |       23.921 |         24.065 |
### Speed multiplier vs. naive (higher is better)
|                                |   (256, 256) |   (512, 512) |   (1024, 1024) |
|:-------------------------------|-------------:|-------------:|---------------:|
| PerturbedLinear                |        26.1  |         7.4  |           1.85 |
| SparsePerturbedLinear (k=1)    |       109.6  |        63.42 |          39.63 |
| SparsePerturbedLinear (k=2)    |        81.18 |        47.33 |          21.94 |
| PermutedLinear                 |       113.6  |        74.54 |          30.61 |
| PermutedLinear(in_sparsity=.5) |       114.36 |        89.02 |          38.96 |
| PermutedLinear(in_sparsity=.1) |       113.47 |        86.64 |          49.67 |
| Base                           |       387.52 |       218.58 |          83.14 |
| Naive                          |         1    |         1    |           1    |
### Time multiplier vs. base (lower is better)
|                                |   (256, 256) |   (512, 512) |   (1024, 1024) |
|:-------------------------------|-------------:|-------------:|---------------:|
| PerturbedLinear                |        14.85 |        29.52 |          44.86 |
| SparsePerturbedLinear (k=1)    |         3.54 |         3.45 |           2.1  |
| SparsePerturbedLinear (k=2)    |         4.77 |         4.62 |           3.79 |
| PermutedLinear                 |         3.41 |         2.93 |           2.72 |
| PermutedLinear(in_sparsity=.5) |         3.39 |         2.46 |           2.13 |
| PermutedLinear(in_sparsity=.1) |         3.42 |         2.52 |           1.67 |
| Base                           |         1    |         1    |           1    |
| Naive                          |       387.52 |       218.58 |          83.14 |


### Time (ms, lower is better)
|                                       |   (16, 16, (3, 3), (64, 64)) |   (32, 32, (3, 3), (64, 64)) |   (64, 64, (3, 3), (32, 32)) |   (32, 32, (1, 1), (32, 32)) |   (1024, 1024, (1, 1), (1, 1)) |
|:--------------------------------------|-----------------------------:|-----------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|
| PerturbedConv2d                       |                       22.623 |                       37.883 |                       26.877 |                       10.466 |                         83.89  |
| PermutedConv2d                        |                       20.37  |                       40.625 |                       20.125 |                        6.205 |                          1.565 |
| PermutedConv2d(out_sparsity=.5) |                       21.622 |                       42.852 |                       19.034 |                        6.773 |                          1.218 |
| PermutedConv2d(out_sparsity=.1) |                       20.906 |                       40.743 |                       17.552 |                        6.447 |                          1.178 |
| Base                                  |                        8.32  |                       16.273 |                        8.017 |                        1.834 |                          0.719 |
| Naive                                 |                       46.983 |                       44.928 |                       45.732 |                       41.82  |                        146.832 |
### Speed multiplier vs. naive (higher is better)
|                                       |   (16, 16, (3, 3), (64, 64)) |   (32, 32, (3, 3), (64, 64)) |   (64, 64, (3, 3), (32, 32)) |   (32, 32, (1, 1), (32, 32)) |   (1024, 1024, (1, 1), (1, 1)) |
|:--------------------------------------|-----------------------------:|-----------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|
| PerturbedConv2d                       |                         2.08 |                         1.19 |                         1.7  |                         4    |                           1.75 |
| PermutedConv2d                        |                         2.31 |                         1.11 |                         2.27 |                         6.74 |                          93.85 |
| PermutedConv2d(out_sparsity=.5) |                         2.17 |                         1.05 |                         2.4  |                         6.17 |                         120.52 |
| PermutedConv2d(out_sparsity=.1) |                         2.25 |                         1.1  |                         2.61 |                         6.49 |                         124.64 |
| Base                                  |                         5.65 |                         2.76 |                         5.7  |                        22.8  |                         204.2  |
| Naive                                 |                         1    |                         1    |                         1    |                         1    |                           1    |
### Time multiplier vs. base (lower is better)
|                                       |   (16, 16, (3, 3), (64, 64)) |   (32, 32, (3, 3), (64, 64)) |   (64, 64, (3, 3), (32, 32)) |   (32, 32, (1, 1), (32, 32)) |   (1024, 1024, (1, 1), (1, 1)) |
|:--------------------------------------|-----------------------------:|-----------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|
| PerturbedConv2d                       |                         2.72 |                         2.33 |                         3.35 |                         5.71 |                         116.66 |
| PermutedConv2d                        |                         2.45 |                         2.5  |                         2.51 |                         3.38 |                           2.18 |
| PermutedConv2d(out_sparsity=.5) |                         2.6  |                         2.63 |                         2.37 |                         3.69 |                           1.69 |
| PermutedConv2d(out_sparsity=.1) |                         2.51 |                         2.5  |                         2.19 |                         3.52 |                           1.64 |
| Base                                  |                         1    |                         1    |                         1    |                         1    |                           1    |
| Naive                                 |                         5.65 |                         2.76 |                         5.7  |                        22.8  |                         204.2  |



To summarize, there are huge gains for dense layers. For certain convolutional layers that are already highly parallel, there are only modest gains versus the naive implementation.

## How to use
All classes/layers can be found in modules.py.

1. Replace the paramaterized layers in the base model with perturbed versions (i.e. Linear -> PerturbedLinear), passing in the population size using the directions parameter. PermutedLinear and PermutedConv2d are the most recommended to use, as they are fast and don't use large amounts of memory.

2. Create the wrapper as perturbed_model = PerturbedModel(base_model, directions). To run in evaluation mode, call base_model.my_function(). To run in training mode, call perturbed_model.my_function(), which will set the layers to training mode and then call base_model.my_function().

3. For each training iteration:

    3.1 Initialize noise tensors by calling perturbed_model.set_noise().
    
    3.2 Using torch.no_grad(), calculate the training loss. The batch size should be an even multiple of the population size for antithetic sampling, in which case the input should be in the shape of (repeat_size, directions, ...), and the second half of the repeats will use the negated noise vector. The batch size can also be set to be equal to the population size, in which case antithetic sampling will not be used.
    
    3.3 Calculate the update weights as desired, then call perturbed_model.set_grad(weights), then step the optimizer.
    
4. Before saving the model, call perturbed_model.free_memory() to delete all noise tensors.
