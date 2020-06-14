import torch
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
                text = text.replace(a,b)

            self.saved[req] = cp.RawKernel(text, self.name, backend=self.backend)
        return self.saved[req]

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self(*item)
        return self(item)

kernels = {}



for file in "batch_im2col,batch_im2col_input,batch_conv2d,bmm,bmmT,pack,weighted_sum".split(","):
    with open("extensions/src/{}.cu".format(file)) as f:
        text = f.read()
        kernels[file] = KernelWrapper(text, file+"_kernel")

for file in "sample_bits".split(","):
    with open("extensions/src/{}.cu".format(file)) as f:
        text = f.read()
        kernels[file] = KernelWrapper(text, file+"_kernel","nvcc")
for file in "texture_batch_im2col,texture_bmm".split(","):
    with open("extensions/src/{}.cu".format(file)) as f:
        text = f.read()
        kernels[file] = {torch.int32: cp.RawKernel(text, file+"_kernel")}
# for file in "sample_bits".split(","): #slower, probably because of some compiler flag
#     f = "extensions/src/{}.ptx".format(file)
#     # f = "extensions/src/{}.cubin".format(file)
#     kernels[file] = {torch.int64: cp.RawModule(path=f).get_function(file+"_kernel")}

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
    kernels["batch_im2col"][input.dtype,("REPEAT 1", "REPEAT {}".format(REPEAT))](grid, block, args=[
        output,
        # cp.asarray(input),
        input.data_ptr(),
        *output.shape,
        *input.shape[1:],
        filterx, filtery, padx, pady, stridex, stridey
    ])
    return torch.as_tensor(output, device=input.device)

def batch_im2col_input(input, filterx, filtery, padx, pady, stridex, stridey):
    REPEAT = 1
    h = (input.size(1) - filterx + 2 * padx) // stridex + 1
    w = (input.size(2) - filtery + 2 * pady) // stridey + 1
    xy = input.shape[1] * input.shape[2]
    output = cp.zeros((input.size(0), h * w, filterx * filtery * input.size(3)), dtype=tensor_cupy_type(input))
    threadsx = next_pow2_clip(input.shape[3], 256)
    threadsy = next_pow2_clip(xy, 256 // threadsx)
    threadsz = 1
    block = (threadsx, threadsy, threadsz)
    grid = ceil_div(input.shape[3], threadsx), ceil_div(xy, threadsy ), ceil_div(output.shape[0], REPEAT * threadsz)
    kernels["batch_im2col_input"][input.dtype](grid, block, args=[
        output,
        # cp.asarray(input),
        input.data_ptr(),
        *output.shape,
        *input.shape[1:],
        filterx, filtery, padx, pady, stridex, stridey
    ])
    return torch.as_tensor(output, device=input.device)

def texture_batch_im2col(input, filterx, filtery, padx, pady, stridex, stridey):
    h = (input.size(1) - filterx + 2 * padx) // stridex + 1
    w = (input.size(2) - filtery + 2 * pady) // stridey + 1

    output = cp.zeros((input.size(0), h * w, filterx * filtery * input.size(3)), dtype=tensor_cupy_type(input))
    threadsx = next_pow2_clip(output.shape[2], 256)
    threadsy = next_pow2_clip(output.shape[1], 256 // threadsx)
    threadsz = 1
    block = (threadsx, threadsy, threadsz)
    grid = ceil_div(output.shape[2], threadsx), ceil_div(output.shape[1], threadsy), ceil_div(output.shape[0], threadsz)
    channels = cptex.ChannelFormatDescriptor(32,0,0,0,cprun.cudaChannelFormatKindSigned)
    # cuArr = cptex.CUDAarray(channels, input.size(3), input.size(2), input.size(1) * input.size(0))
    # cuArr.copy_from(cp.asarray(input.view(-1, input.size(2), input.size(3))))
    # resDesc = cptex.ResourceDescriptor(cprun.cudaResourceTypeArray, cuArr=cuArr)
    arr = cp.asarray(input)
    resDesc = cptex.ResourceDescriptor(cprun.cudaResourceTypeLinear, arr=arr, chDesc=channels, sizeInBytes=arr.size*arr.dtype.itemsize)
    texDesc = cptex.TextureDescriptor(addressModes=(cprun.cudaAddressModeBorder,) * 1)
    tex = cptex.TextureObject(resDesc,texDesc)
    kernels["texture_batch_im2col"][input.dtype](grid, block, args=[
        output,
        tex,
        *output.shape,
        *input.shape[1:],
        filterx, filtery, padx, pady, stridex, stridey
    ])
    return torch.as_tensor(output, device=input.device)

def batch_conv2d(input, filter, padx, pady, stridex, stridey):
    h = (input.size(1) - filter.size(2) + 2 * padx) // stridex + 1
    w = (input.size(2) - filter.size(3) + 2 * pady) // stridey + 1
    output = cp.zeros((filter.size(0), h, w, filter.size(1)), dtype=cp.int32)
    hw = h * w
    bo = filter.size(0) * filter.size(1)

    channels = cptex.ChannelFormatDescriptor(32, 0, 0, 0, cprun.cudaChannelFormatKindSigned)
    cuArr = cptex.CUDAarray(channels, input.size(3), input.size(2), input.size(1) * input.size(0))
    cuArr.copy_from(cp.asarray(input.view(-1, input.size(2), input.size(3))))
    resDesc = cptex.ResourceDescriptor(cprun.cudaResourceTypeArray, cuArr=cuArr)
    # arr = cp.asarray(input)
    # resDesc = cptex.ResourceDescriptor(cprun.cudaResourceTypeLinear, arr=arr, chDesc=channels, sizeInBytes=arr.size*arr.dtype.itemsize)
    texDesc = cptex.TextureDescriptor(addressModes=(cprun.cudaAddressModeBorder,) * 1)
    tex = cptex.TextureObject(resDesc, texDesc)


    threadsx = next_pow2_clip(hw, 256)
    threadsy = next_pow2_clip(bo, 256 / threadsx)
    block = (threadsx, threadsy, 1)
    grid = (ceil_div(hw, threadsx),ceil_div(bo, threadsy), 1)
    kernels["batch_conv2d"][input.dtype](grid, block, args=[
        output,
        # input.data_ptr(),
        # cp.asarray(input),
        tex,
        filter.data_ptr(),
        *output.shape,
        *input.shape[1:],
        *filter.shape[1:],
         padx, pady, stridex, stridey
    ])
    return torch.as_tensor(output, device=input.device)

def bmm(A,B):
    BLOCK_SIZE = 16
    MULT_A = 4
    MULT_B = 4
    C = cp.empty((A.size(0), A.size(1), B.size(2)), dtype=cp.int32)
    threadsx = BLOCK_SIZE
    threadsy = BLOCK_SIZE
    threadsz = 1
    block = threadsx, threadsy, threadsz
    grid = ceil_div(C.shape[2], threadsx * MULT_B), ceil_div(C.shape[1], threadsy * MULT_A), ceil_div(C.shape[0], threadsz)
    kernels["bmm"][A.dtype](grid, block, args=[
        C,
        A.data_ptr(),
        B.data_ptr(),
        *C.shape[1:],
        *A.shape[1:],
        *B.shape[1:]
    ])
    return torch.as_tensor(C, device=A.device)

def bmmT(A,B):
    BLOCK_SIZE = 16
    MULT_A = 4
    MULT_B = 4
    C = cp.empty((A.size(0), A.size(1), B.size(1)), dtype=cp.int32)
    threadsx = BLOCK_SIZE
    threadsy = BLOCK_SIZE
    threadsz = 1
    block = threadsx, threadsy, threadsz
    grid = ceil_div(C.shape[2], threadsx * MULT_B), ceil_div(C.shape[1], threadsy * MULT_A), ceil_div(C.shape[0], threadsz)
    kernels["bmmT"][A.dtype](grid, block, args=[
        C,
        A.data_ptr(),
        B.data_ptr(),
        *C.shape[1:],
        *A.shape[1:],
        *B.shape[1:]
    ])
    return torch.as_tensor(C, device=A.device)

def texture_bmm(A,B):
    BLOCK_SIZE = 8
    C = cp.zeros((A.size(0), A.size(1), B.size(2)), dtype=cp.int32)
    channels = cptex.ChannelFormatDescriptor(32,0,0,0,cprun.cudaChannelFormatKindSigned)
    # cuArrA = cptex.CUDAarray(channels, A.size(2), A.size(1), A.size(0))
    # cuArrB = cptex.CUDAarray(channels, B.size(2), B.size(1), B.size(0))
    # cuArrA.copy_from(cp.asarray(A))
    # cuArrB.copy_from(cp.asarray(B))
    # resDescA = cptex.ResourceDescriptor(cprun.cudaResourceTypeArray, cuArr=cuArrA)
    # resDescB = cptex.ResourceDescriptor(cprun.cudaResourceTypeArray, cuArr=cuArrB)
    arrA = cp.asarray(A)
    arrB = cp.asarray(B)
    resDescA = cptex.ResourceDescriptor(cprun.cudaResourceTypeLinear, arr=arrA, chDesc=channels, sizeInBytes=arrA.size*arrA.dtype.itemsize)
    resDescB = cptex.ResourceDescriptor(cprun.cudaResourceTypeLinear, arr=arrB, chDesc=channels, sizeInBytes=arrB.size*arrB.dtype.itemsize)
    texDesc = cptex.TextureDescriptor(addressModes=(cprun.cudaAddressModeBorder,) * 1)
    texA = cptex.TextureObject(resDescA, texDesc)
    texB = cptex.TextureObject(resDescB, texDesc)
    threadsx = BLOCK_SIZE
    threadsy = BLOCK_SIZE
    threadsz = 1
    block = threadsx, threadsy, threadsz
    grid = ceil_div(C.shape[2], threadsx) , ceil_div(C.shape[1], threadsy), ceil_div(C.shape[0], threadsz)
    kernels["texture_bmm"][A.dtype](grid, block, args=[
        C,
        texA,
        texB,
        *C.shape[1:],
        *A.shape[1:],
        *B.shape[1:]
    ])

def conv_bmm(input, filter, filterx, filtery, padx, pady, stridex, stridey, ):
    cols = batch_im2col(input, filterx, filtery, padx, pady, stridex, stridey)
    return bmm(cols, filter)

def conv_bmmT(input, filter, filterx, filtery, padx, pady, stridex, stridey):
    cols = batch_im2col(input, filterx, filtery, padx, pady, stridex, stridey)
    return bmmT(cols, filter)

def texture_conv_bmm(input, filter, filterx, filtery, padx, pady, stridex, stridey):
    cols = batch_im2col(input, filterx, filtery, padx, pady, stridex, stridey)
    return texture_bmm(cols, filter)

def pack(input, dtype=torch.int32):
    BLOCK_SIZE = 16 ** 2
    ret_size1 = ceil_div(input.shape[1], torch.iinfo(dtype).bits)
    threadsx = next_pow2_clip(ret_size1, BLOCK_SIZE)
    threadsy = BLOCK_SIZE // threadsx
    block = (threadsx, threadsy, 1)
    grid = (ceil_div(ret_size1, threadsx), ceil_div(input.size(0), threadsy))
    ret = cp.empty((input.size(0),ret_size1), dtype=tensor_cupy_type(dtype))
    kernels["pack"][dtype](grid, block, args=[
        ret, input.data_ptr(), *ret.shape, input.shape[1]
    ])
    return torch.as_tensor(ret, device=input.device)

# def sample_bits(p, n, dtype, seed):
#     ret_size2 = ceil_div(p.size(1), torch.iinfo(dtype).bits)
#     ret = cp.empty((n, p.size(0), ret_size2), dtype=tensor_cupy_type(dtype))
#     threadsx = next_pow2_clip(ret_size2, 128)
#     threadsy = next_pow2_clip(p.shape[0], 128 // threadsx)
#     block = (threadsx, threadsy, 1)
#     grid = (ceil_div(ret_size2, threadsx), ceil_div(p.shape[0], threadsy), 1)
#     kernels["sample_bits"][dtype](grid, block, args=[
#         ret, p.data_ptr(), *ret.shape, p.shape[1], seed
#     ])
#     return torch.as_tensor(ret, device=p.device)

def sample_bits(p, n, dtype, seed):
    BLOCK_SIZE = 128
    ret_size2 = ceil_div(p.size(1), torch.iinfo(dtype).bits)
    ret = cp.empty((n, p.size(0), ret_size2), dtype=tensor_cupy_type(dtype))
    threadsx = BLOCK_SIZE
    block = (threadsx, 1, 1)
    grid = (ceil_div(n, threadsx), p.size(0), ret_size2)
    kernels["sample_bits"][dtype](grid, block, args=[
        ret, p.data_ptr(), *ret.shape, p.shape[1], seed
    ])
    return torch.as_tensor(ret, device=p.device)

def weighted_sum(input, weights, zbits):
    BLOCK_SIZE = 64
    ret = cp.empty((input.size(1), zbits), dtype=cp.float16)
    threadsx = BLOCK_SIZE
    block = (threadsx, 1, 1)
    grid = (ceil_div(zbits, threadsx), input.size(1), 1)
    kernels["weighted_sum"][input.dtype](grid, block, args=[
        ret, input.data_ptr(), weights.data_ptr(), *ret.shape, *input.shape
    ])
    return torch.as_tensor(ret, device=input.device)