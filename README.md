# NESgpu
Optimized PyTorch GPU implementation of Natural Evolution Strategies/Augmented Random Search

See https://github.com/jamesal1/NESgpu/wiki/Explanation-of-Design-Decisions for an explanation of the optimizations.

## How to use

1. Replace the paramaterized layers in the base model with perturbed versions (i.e. Linear -> PerturbedLinear), passing in the population size using the directions parameter.

2. Initialize the wrapper as perturbed_model = PerturbedModel(base_model, directions). To run in evaluation mode, call base_model.function(). To run in training mode, call perturbed_model.function(), which will set the layers to training mode and then call base_model.function().

3. For each training iteration:

    3.1 Initialize noise tensors by calling perturbed_model.set_noise().
    
    3.2 Using torch.no_grad(), calculate the training loss. The batch size should be an even multiple of the population size for antithetic sampling, in which case the input should be in the shape of (repeat_size, directions, ...), and the second half of the repeats will use the negated noise vector. The batch size can also be set to be equal to the population size, in which case antithetic sampling will not be used.
    
    3.3 Calculate the update weights as desired, then call perturbed_model.set_grad(weights), then step the optimizer.
    
4. Before saving the model, call perturbed_model.free_memory() to delete all noise tensors.




## Forward Pass Performance
Experiments were done on a GeForce 1080 Ti, with a batch size of 1024. The format for the column headers are (out dimension, in dimension) for the dense layers and (out_channels, in channels, filter_size, input_size) for the convolutional layers. 

The base time is the time for the layer to compute a forward pass in evaluation mode; it represents an upper bound on the performance of the layer in training mode.

The naive time is calculated by running the layer in evaluation mode with a batch size of 1, repeated 1024 times, to simulate the performance of an implementation that samples each batch element sequentially.
### Time (ms, lower is better)
|                             |   (256, 256) |   (512, 512) |   (1024, 1024) |
|:----------------------------|-------------:|-------------:|---------------:|
| PerturbedLinear             |        0.863 |        3.229 |         12.252 |
| SparsePerturbedLinear (k=1) |        0.188 |        0.396 |          0.641 |
| SparsePerturbedLinear (k=2) |        0.226 |        0.483 |          1.041 |
| PermutedLinear              |        0.222 |        0.348 |          0.867 |
| Base                        |        0.051 |        0.109 |          0.306 |
| Naive                       |       22.282 |       22.42  |         23.259 |

### Speed multiplier vs. naive (higher is better)
|                             |   (256, 256) |   (512, 512) |   (1024, 1024) |
|:----------------------------|-------------:|-------------:|---------------:|
| PerturbedLinear             |        25.82 |         6.94 |           1.9  |
| SparsePerturbedLinear (k=1) |       118.45 |        56.68 |          36.28 |
| SparsePerturbedLinear (k=2) |        98.38 |        46.42 |          22.34 |
| PermutedLinear              |       100.28 |        64.41 |          26.82 |
| Base                        |       434.69 |       205.32 |          75.98 |
| Naive                       |         1    |         1    |           1    |

### Time multiplier vs. base (lower is better)
|                             |   (256, 256) |   (512, 512) |   (1024, 1024) |
|:----------------------------|-------------:|-------------:|---------------:|
| PerturbedLinear             |        16.83 |        29.57 |          40.02 |
| SparsePerturbedLinear (k=1) |         3.67 |         3.62 |           2.09 |
| SparsePerturbedLinear (k=2) |         4.42 |         4.42 |           3.4  |
| PermutedLinear              |         4.33 |         3.19 |           2.83 |
| Base                        |         1    |         1    |           1    |
| Naive                       |       434.69 |       205.32 |          75.98 |
### Time (ms, lower is better)
|                 |   (16, 16, (3, 3), (64, 64)) |   (32, 32, (3, 3), (64, 64)) |   (64, 64, (3, 3), (32, 32)) |   (32, 32, (1, 1), (32, 32)) |   (1024, 1024, (1, 1), (1, 1)) |
|:----------------|-----------------------------:|-----------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|
| PerturbedConv2d |                       20.251 |                       33.983 |                       24.37  |                        9.73  |                         85.235 |
| PermutedConv2d  |                       17.947 |                       35.562 |                       18.163 |                        5.604 |                          1.362 |
| Base            |                        7.01  |                       13.805 |                        7.269 |                        1.734 |                          0.614 |
| Naive           |                       43.68  |                       42.905 |                       42.851 |                       38.517 |                        161.733 |
### Speed multiplier vs. naive (higher is better)
|                 |   (16, 16, (3, 3), (64, 64)) |   (32, 32, (3, 3), (64, 64)) |   (64, 64, (3, 3), (32, 32)) |   (32, 32, (1, 1), (32, 32)) |   (1024, 1024, (1, 1), (1, 1)) |
|:----------------|-----------------------------:|-----------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|
| PerturbedConv2d |                         2.16 |                         1.26 |                         1.76 |                         3.96 |                           1.9  |
| PermutedConv2d  |                         2.43 |                         1.21 |                         2.36 |                         6.87 |                         118.78 |
| Base            |                         6.23 |                         3.11 |                         5.9  |                        22.21 |                         263.24 |
| Naive           |                         1    |                         1    |                         1    |                         1    |                           1    |

### Time multiplier vs. base (lower is better)
|                 |   (16, 16, (3, 3), (64, 64)) |   (32, 32, (3, 3), (64, 64)) |   (64, 64, (3, 3), (32, 32)) |   (32, 32, (1, 1), (32, 32)) |   (1024, 1024, (1, 1), (1, 1)) |
|:----------------|-----------------------------:|-----------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|
| PerturbedConv2d |                         2.89 |                         2.46 |                         3.35 |                         5.61 |                         138.73 |
| PermutedConv2d  |                         2.56 |                         2.58 |                         2.5  |                         3.23 |                           2.22 |
| Base            |                         1    |                         1    |                         1    |                         1    |                           1    |
| Naive           |                         6.23 |                         3.11 |                         5.9  |                        22.21 |                         263.24 |
