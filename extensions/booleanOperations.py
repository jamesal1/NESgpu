import torch
from collections import namedtuple
Stream = namedtuple('Stream', ['ptr'])
torch_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
import numpy as np

import cupy as cp
import cupy.cuda.texture as cptex
import cupy.cuda.runtime as cprun
def long_to_int(st):
    return st.replace("long", "int").replace("popcll", "popc").replace("ELEMENT_SIZE 64", "ELEMENT_SIZE 32")

class KernelWrapper():

    def __init__(self, raw, name, backend="nvrtc"):
        self.raw = raw
        self.name = name
        self.backend = backend
        self.saved = {}

    def __call__(self, *args, **kwargs):
        req = tuple(args)
        if req not in self.saved:
            text = self.raw
            if args[0] == torch.int32:
                text = long_to_int(text)
            elif args[0] != torch.int64:
                raise NotImplemented
            for a, b in args[1:]:
                if isinstance(b,int):
                    text = text.replace("{} {}".format(a,"X"),"{} {}".format(a,b))
                else:
                    text = text.replace(a,b)

            self.saved[req] = cp.RawKernel(text, self.name, backend=self.backend, options=("-lineinfo",))
        return self.saved[req]

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self(*item)
        return self(item)

kernels = {}



for file in "batch_im2col,bmm,bmm_act,pack,int8pack,weighted_sum,max_pool2d".split(","):
    with open("extensions/src/{}.cu".format(file)) as f:
        text = f.read()
        kernels[file] = KernelWrapper(text, file+"_kernel")

for file in "sample_bits".split(","):
    with open("extensions/src/{}.cu".format(file)) as f:
        text = f.read()
        kernels[file] = KernelWrapper(text, file+"_kernel","nvcc")

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
    dtype = tensor.dtype if isinstance(tensor, torch.Tensor) else tensor
    if dtype == torch.int64:
        return cp.int64
    if dtype == torch.int32:
        return cp.int32
    raise NotImplemented

def batch_im2col(input, filterx, filtery, padx, pady, stridex, stridey, REPEAT = 128):
    h = (input.size(1) - filterx + 2 * padx) // stridex + 1
    w = (input.size(2) - filtery + 2 * pady) // stridey + 1
    # return torch.zeros((input.size(0), h * w, filterx * filtery * input.size(3)), dtype=input.dtype, device=input.device)

    output = cp.empty((input.size(0), h * w, filterx * filtery * input.size(3)), dtype=tensor_cupy_type(input))
    threadsx = next_pow2_clip(output.shape[2], 512)
    threadsy = next_pow2_clip(output.shape[1], 512 // threadsx)
    threadsz = 1
    block = (threadsx, threadsy, threadsz)
    grid = ceil_div(output.shape[2], threadsx), ceil_div(output.shape[1], threadsy), ceil_div(output.shape[0], REPEAT * threadsz)
    # grid = ceil_div(output.shape[2], threadsx), ceil_div(output.shape[1], REPEAT * threadsy), ceil_div(output.shape[0], threadsz)
    kernels["batch_im2col"][input.dtype,("REPEAT",REPEAT)](grid, block, args=[
        output,
        # cp.asarray(input),
        input.data_ptr(),
        *output.shape,
        *input.shape[1:],
        filterx, filtery, padx, pady, stridex, stridey
    ],stream=torch_stream)
    return torch.as_tensor(output, device=input.device)




def bmm(A,B):
    assert(len(A.shape) == 3)
    assert(len(B.shape) == 3)
    assert(A.size(0) == B.size(0))
    assert(A.size(2) == B.size(1))
    BLOCK_SIZE = 8
    MULT_A = 4
    MULT_B = 4
    C = cp.empty((A.size(0), A.size(1), B.size(2)), dtype=cp.int32)
    threadsx = BLOCK_SIZE
    threadsy = BLOCK_SIZE
    threadsz = 1
    block = threadsx, threadsy, threadsz
    grid = ceil_div(C.shape[2], threadsx * MULT_B), ceil_div(C.shape[1], threadsy * MULT_A), ceil_div(C.shape[0], threadsz)
    kernels["bmm"][A.dtype,("BLOCK_SIZE",BLOCK_SIZE),("MULT_A", MULT_A),("MULT_B", MULT_B)](grid, block, args=[
        C,
        A.data_ptr(),
        B.data_ptr(),
        *C.shape[1:],
        *A.shape[1:],
        *B.shape[1:]
    ],stream=torch_stream)
    return torch.as_tensor(C, device=A.device)

def bmm_act(A,B, threshold):
    assert(len(A.shape) == 3)
    assert(len(B.shape) == 3)
    assert(len(threshold.shape) == 2)
    assert(A.size(0) == B.size(0) == threshold.size(0))
    assert(A.size(2) == B.size(1))
    assert(threshold.size(1) == B.size(2))
    assert(threshold.dtype == torch.int32)
    BLOCK_SIZE = 8
    MULT_A = 4
    MULT_B = 4
    C = cp.empty((A.size(0), A.size(1), B.size(2)), dtype=cp.bool)
    threadsx = BLOCK_SIZE
    threadsy = BLOCK_SIZE
    threadsz = 1
    block = threadsx, threadsy, threadsz
    grid = ceil_div(C.shape[2], threadsx * MULT_B), ceil_div(C.shape[1], threadsy * MULT_A), ceil_div(C.shape[0], threadsz)
    kernels["bmm_act"][A.dtype,("BLOCK_SIZE",BLOCK_SIZE),("MULT_A", MULT_A),("MULT_B", MULT_B)](grid, block, args=[
        C,
        A.data_ptr(),
        B.data_ptr(),
        threshold.data_ptr(),
        *C.shape[1:],
        *A.shape[1:],
        *B.shape[1:]
    ],stream=torch_stream)
    # cp.cuda.stream.get_current_stream().synchronize()
    return torch.as_tensor(C, device=A.device)



def conv(input, filter, filterx, filtery, padx, pady, stridex, stridey):
    h = (input.size(1) - filterx + 2 * padx) // stridex + 1
    w = (input.size(2) - filtery + 2 * pady) // stridey + 1
    cols = batch_im2col(input, filterx, filtery, padx, pady, stridex, stridey)
    return bmm(cols, filter).view(input.size(0),h,w, -1)

def conv_act(input, filter, threshold, filterx, filtery, padx, pady, stridex, stridey):
    h = (input.size(1) - filterx + 2 * padx) // stridex + 1
    w = (input.size(2) - filtery + 2 * pady) // stridey + 1
    cols = batch_im2col(input, filterx, filtery, padx, pady, stridex, stridey)
    return bmm_act(cols, filter, threshold).view(input.size(0),h,w, -1)


def pack(input, dtype=torch.int32):
    assert(len(input.shape)==2)
    BLOCK_SIZE = 16 ** 2
    ret_size1 = ceil_div(input.shape[1], torch.iinfo(dtype).bits)
    threadsx = next_pow2_clip(ret_size1, BLOCK_SIZE)
    threadsy = BLOCK_SIZE // threadsx
    block = (threadsx, threadsy, 1)
    grid = (ceil_div(ret_size1, threadsx), ceil_div(input.size(0), threadsy))
    ret = cp.empty((input.size(0),ret_size1), dtype=tensor_cupy_type(dtype))
    kernels["pack"][dtype](grid, block, args=[
        ret, input.data_ptr(), *ret.shape, input.shape[1]
    ],stream=torch_stream)
    return torch.as_tensor(ret, device=input.device)

def int8pack(input, dtype=torch.int32):
    assert(len(input.shape) == 2)
    BLOCK_SIZE = 16 ** 2
    ret_size1 = ceil_div(input.shape[1], torch.iinfo(dtype).bits // 8)
    threadsx = next_pow2_clip(ret_size1, BLOCK_SIZE)
    threadsy = BLOCK_SIZE // threadsx
    block = (threadsx, threadsy, 1)
    grid = (ceil_div(ret_size1, threadsx), ceil_div(input.size(0), threadsy))
    ret = cp.empty((input.size(0), ret_size1), dtype=tensor_cupy_type(dtype))
    kernels["int8pack"][dtype](grid, block, args=[
        ret, input.data_ptr(), *ret.shape, input.shape[1]
    ],stream=torch_stream)
    return torch.as_tensor(ret, device=input.device)

def sample_bits(p, n, dtype, seed):
    assert(len(p.shape) == 2)
    BLOCK_SIZE = 128
    ret_size2 = ceil_div(p.size(1), torch.iinfo(dtype).bits)
    ret = cp.empty((n, p.size(0), ret_size2), dtype=tensor_cupy_type(dtype))
    threadsx = BLOCK_SIZE
    threadsy = 1
    block = (threadsx, threadsy, 1)
    grid = (ceil_div(ret_size2, threadsx), ceil_div(p.shape[0], threadsy), 1)
    kernels["sample_bits"][dtype, ("BLOCK_SIZE", BLOCK_SIZE)](grid, block, args=[
        ret, p.data_ptr(), *ret.shape, p.shape[1], seed
    ],stream=torch_stream)
    return torch.as_tensor(ret, device=p.device)


def weighted_sum(input, weights, zbits):
    assert(len(input.shape)==3)
    assert(input.size(0)==weights.size(0))
    BLOCK_SIZE = 64
    ret = cp.empty((input.size(1), zbits), dtype=cp.float16)
    threadsx = BLOCK_SIZE
    block = (threadsx, 1, 1)
    grid = (ceil_div(zbits, threadsx), input.size(1), 1)
    kernels["weighted_sum"][input.dtype](grid, block, args=[
        ret, input.data_ptr(), weights.data_ptr(), *ret.shape, *input.shape
    ],stream=torch_stream)
    return torch.as_tensor(ret, device=input.device)


def max_pool2d(input, filterx, filtery, padx, pady, stridex, stridey, REPEAT=128):
    assert(len(input.shape) == 4)
    h = (input.size(1) - filterx + 2 * padx) // stridex + 1
    w = (input.size(2) - filtery + 2 * pady) // stridey + 1

    output = cp.empty((input.size(0), h * w, input.size(3)), dtype=tensor_cupy_type(input))
    threadsx = next_pow2_clip(output.shape[2], 128)
    threadsy = next_pow2_clip(output.shape[1], 128 // threadsx)
    threadsz = 1
    block = (threadsx, threadsy, threadsz)
    grid = ceil_div(output.shape[2], threadsx), ceil_div(output.shape[1], threadsy), ceil_div(output.shape[0], REPEAT * threadsz)
    # grid = ceil_div(output.shape[2], threadsx), ceil_div(output.shape[1], REPEAT * threadsy), ceil_div(output.shape[0], threadsz)
    kernels["max_pool2d"][input.dtype,("REPEAT", REPEAT)](grid, block, args=[
        output,
        # cp.asarray(input),
        input.data_ptr(),
        *output.shape,
        *input.shape[1:],
        filterx, filtery, padx, pady, stridex, stridey
    ],stream=torch_stream)
    return torch.as_tensor(output, device=input.device).view(input.size(0),h,w,input.size(3))