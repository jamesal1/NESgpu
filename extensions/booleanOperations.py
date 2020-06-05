import torch
import numpy as np

import cupy as cp


with open("extensions/booleanOperations.cu") as f:

    batch_im2col_kernel = cp.RawKernel(f.read(), "batch_im2col_kernel")

def next_pow2_clip(v, cap):
    if (v > cap // 2):
        return cap
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v += 1
    return v

def ceil_div(a, b):
    return (a + b - 1) // b



def tensor_cupy_type(tensor):
    dtype = tensor.dtype
    if dtype == torch.int64:
        return cp.int64
    if dtype == torch.int32:
        return cp.int32
    raise NotImplemented

def batch_im2col(input, filterx, filtery, padx, pady, stridex, stridey):
    h = (input.size(1) - filterx + 2 * padx) // stridex + 1
    w = (input.size(2) - filtery + 2 * pady) // stridey + 1
    return torch.zeros((input.size(0), h * w, filterx * filtery * input.size(3)), dtype=input.dtype, device=input.device)

    output = cp.zeros((input.size(0), h * w, filterx * filtery * input.size(3)), dtype=tensor_cupy_type(input))
    threadsx = next_pow2_clip(output.shape[2], 256)
    threadsy = next_pow2_clip(output.shape[1], 256 // threadsx)
    threadsz = 1
    block = (threadsx, threadsy, threadsz)
    grid = ceil_div(output.shape[2], threadsx), ceil_div(output.shape[1], threadsy), ceil_div(output.shape[0], threadsz)
    batch_im2col_kernel(grid, block, args=[
        output,
        input.data_ptr(),
        *(np.int32(x) for x in output.shape),
        *(np.int32(x) for x in input.shape[1:]),
        np.int32(filterx), np.int32(filtery), np.int32(padx), np.int32(pady), np.int32(stridex), np.int32(stridey)
    ])
    return torch.as_tensor(output, device=input.device)
