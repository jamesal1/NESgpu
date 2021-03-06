import torch
from torch.utils.cpp_extension import load
import time
import torch.nn.functional as F
from extensions import booleanOperations
def time_function(fun, *args):
    torch.cuda.synchronize()
    start = time.time()
    fun(*args)
    torch.cuda.synchronize()
    res = time.time() - start
    return round(res * 1000, 2)


boolop_cuda = load(name="boolop_cuda", sources=["extensions/booleanOperations.cpp","extensions/booleanOperationsCuda.cu"])
boolutil_cuda = load(name="boolutil_cuda", sources=["extensions/booleanUtilities.cpp","extensions/booleanUtilitiesCuda.cu"])


def pack(input, dtype=torch.int32):
    bits = torch.iinfo(dtype).bits
    return boolutil_cuda.pack(input,  bits)

def pack_cupy(input, dtype=torch.int32):
    return booleanOperations.pack(input, dtype)

def bmm_test(packA, packB):
    return boolop_cuda.binary_bmm(packA, packB)

def seeded_bmv_test(P, packB):
    return boolop_cuda.binary_seeded_bmv(P, packB, 999)

def pack_test(A, dtype):
    packA = pack(A, dtype=dtype)

def pack32_test(A):
    packA = pack(A,dtype=torch.int32)

def pack8_test(A):
    packA = boolutil_cuda.pack8(A)

def unpack_test(packA):
    boolutil_cuda.unpack(packA)

def sample(P, batch_dim, dtype=torch.int32):
    bits = torch.iinfo(dtype).bits
    return boolutil_cuda.sample_bits(P, batch_dim, bits, 999)

def sample_cupy(P, batch_dim, dtype=torch.int32):
    return booleanOperations.sample_bits(P, batch_dim, dtype, 999)

def naive_sample(size):
    return torch.randint(2, size=size, device="cuda", dtype=torch.bool)

def naive_bmm(A,B):
    return torch.bmm(A, B)

def weighted_sum(packA, ws, inner_dim):
    return boolutil_cuda.binary_weighted_sum(packA, ws, inner_dim)


def conv_act(packed_input, packed_filter, thresh, padding=1, stride=1):
    packed_filter_reordered = packed_filter.permute([0, 2, 3, 4, 1]).view(packed_filter.size(0), -1, packed_filter.size(1)).contiguous()
    return booleanOperations.conv_act(packed_input, packed_filter_reordered, thresh, packed_filter.size(2), packed_filter.size(3), padding, padding, stride, stride)

def conv_bmm_cupy(packed_input, packed_filter, padding=1, stride=1):
    packed_filter_reordered = packed_filter.permute([0, 2, 3, 4, 1]).view(packed_filter.size(0), -1, packed_filter.size(1)).contiguous()
    return booleanOperations.conv(packed_input, packed_filter_reordered, packed_filter.size(2), packed_filter.size(3), padding, padding, stride, stride)



def compare(input, bias):
    return input > bias


def reorder(packed_input, packed_filter, padding=1, stride=1):
    packed_filter_reordered = packed_filter.permute([0, 2, 3, 4, 1]).view(packed_filter.size(0), -1, packed_filter.size(1)).contiguous()

def reorder_input(packed_input, packed_filter, padding=1, stride=1):
    packed_input_reordered = packed_input.permute([0,3,1,2]).contiguous()



def conv_cupy(packed_input, packed_filter, padding=1, stride=1):
    return booleanOperations.batch_conv2d(packed_input, packed_filter, padding, padding, stride, stride)


def im2col(packed_input, packed_filter, padding=1, stride=1):
    return boolop_cuda.batch_im2col(packed_input, packed_filter.size(2), packed_filter.size(3), padding, padding, stride, stride)

def im2colcupy(packed_input, packed_filter, padding=1, stride=1):
    return booleanOperations.batch_im2col(packed_input, packed_filter.size(2), packed_filter.size(3), padding, padding, stride, stride)


def maxpool2d_naive(input, filter=2, padding=0, stride=2):
    return F.max_pool2d(input.permute([0, 3, 1, 2]).type(torch.float16), (filter, filter),(stride, stride), (padding, padding), (1, 1)).permute([0,2,3,1]).type(torch.bool)

def maxpool2d(packed_input, filter=2, padding=0, stride=2):
    return booleanOperations.max_pool2d(packed_input, filter, filter, padding, padding, stride, stride)


def conv_naive(naive_input, naive_filter, batch_dim, padding=1, stride=1):
    return torch.conv2d(torch.nn.functional.pad(naive_input,(padding,) * 4, value=-1), naive_filter, stride=(stride, stride), groups=batch_dim)

def convert_tensor(t):
    import cupy
    return torch.as_tensor(cupy.asarray(t.data_ptr()), device=t.device)

def check_result():
    # batch_dim = 128
    # out_dim = 65
    # in_dim = 64
    # input_size = 32
    batch_dim = 1024
    out_dim = 64
    in_dim = 1
    input_size = 10
    # batch_dim = 1
    # out_dim = 3
    # in_dim = 3
    # input_size = 5
    dtype = torch.int32
    for filter_size in [1, 3, 7]:
        for padding in range(3):
            for stride in range(1,4):
                print(filter_size, padding, stride)
                in_length = (filter_size ** 2 * in_dim)
                input = torch.randint(2, size=(batch_dim * input_size ** 2, in_dim), device="cuda", dtype=torch.bool)
                input8 = torch.randint(255,size=(batch_dim * input_size ** 2, in_dim // 8), device="cuda", dtype=torch.uint8)
                filter = torch.randint(2, size=(batch_dim * out_dim * filter_size ** 2, in_dim), device="cuda", dtype=torch.bool)
                thresh = torch.randint(in_length, size=(batch_dim,out_dim), device="cuda", dtype=torch.int32) - in_length // 2
                # thresh = (filter_size ** 2 * in_dim // 2) * torch.ones(size=(batch_dim,out_dim), device="cuda", dtype=torch.int32)
                print(thresh)
                thresh += in_length

                packed_input = pack(input, dtype).view(batch_dim, input_size, input_size, -1)
                packed_filter = pack(filter, dtype).view(batch_dim, out_dim, filter_size, filter_size, -1)

                mp = maxpool2d(packed_input)
                mpn = maxpool2d_naive(input.view(batch_dim,input_size,input_size,in_dim))
                pmpn = booleanOperations.pack(mpn.view(-1,mpn.size(3))).view_as(mp)
                print("maxpool",torch.allclose(mp, pmpn))

                # i8 = booleanOperations.int8pack(input8, dtype)
                # i8_test = booleanOperations.pack(boolutil_cuda.unpack(input8), dtype)
                # print("int8pack",torch.allclose(i8, i8_test))

                cuda_act = conv_act(packed_input, packed_filter, thresh, padding=padding, stride=stride)
                cuda_res = conv_bmm_cupy(packed_input, packed_filter, padding=padding, stride=stride)
                # cuda_res = booleanOperations.conv(packed_input, packed_filter_reordered, filter_size, filter_size, padding, padding, stride, stride)
                naive_input = (2 * (input.type(torch.float16) - .5)).view(batch_dim, input_size, input_size, in_dim)\
                    .permute([0, 3, 1, 2]).contiguous().view(1, batch_dim * in_dim, input_size, input_size)
                naive_filter = (-2 * (filter.type(torch.float16) - .5)).view(batch_dim , out_dim, filter_size, filter_size, in_dim) \
                    .permute([0, 1, 4, 2, 3]).contiguous().view(batch_dim * out_dim, in_dim, filter_size, filter_size)
                naive_res = torch.conv2d(torch.nn.functional.pad(naive_input,(padding,) * 4, value=-1), naive_filter, stride=(stride, stride), groups=batch_dim)
                naive_res = naive_res.view(batch_dim, out_dim, *naive_res.shape[-2:]).permute([0, 2, 3, 1])
                print(thresh.shape, cuda_res.shape, cuda_act.shape)
                cuda_res_act = 2 * cuda_res > thresh.unsqueeze(1).unsqueeze(1)
                # print(cuda_act)
                print("conv_act",torch.all(cuda_res_act == cuda_act))
                print("conv",torch.allclose((cuda_res * 2 - (filter_size ** 2 * in_dim)).type(torch.float16).view_as(naive_res), naive_res))
                # print(cuda_res)
                # print((cuda_res * 2 - (filter_size ** 2 * in_dim)), naive_res)
                # exit()


def speedtests():
    batch_dim = 2 ** 10
    out_dim = 2 ** 7
    inner_dim = 2 ** 13
    batch_dim = 2 ** 9
    out_dim = 784
    inner_dim = 784
    repeat = 5

    A = torch.randint(2, size=(batch_dim * out_dim, inner_dim), device="cuda", dtype=torch.bool)
    A2 = torch.randint(2, size=(batch_dim * out_dim * 4, inner_dim // 4), device="cuda", dtype=torch.bool)
    B = torch.randint(2, size=(batch_dim, inner_dim), device="cuda", dtype=torch.bool)
    dtype = torch.int32
    packA = pack(A, dtype)
    packB = pack(B, dtype)
    P = torch.rand(out_dim,inner_dim, device="cuda", dtype=torch.float16)
    ws = torch.rand(batch_dim, device="cuda", dtype=torch.float16)

    print("bmm_only")
    for i in range(repeat):
        print(time_function(bmm_test, packA.view(batch_dim, out_dim, -1), packB.view(batch_dim, -1, 1)))
    print("seeded_bmv")
    for i in range(repeat):
        print(time_function(seeded_bmv_test, P, packB))
    print("pack")
    for i in range(repeat):
        print(time_function(pack_test, A, dtype))
    print("pack32")
    for i in range(repeat):
        print(time_function(pack32_test, A))
    print("pack8")
    for i in range(repeat):
        print(time_function(pack8_test, A))
    print("pack (vector)")
    for i in range(repeat):
        print(time_function(pack_test, B, dtype))
    print("pack32 (vector)")
    for i in range(repeat):
        print(time_function(pack32_test, B))
    print("pack8 (vector)")
    for i in range(repeat):
        print(time_function(pack8_test, B))
    print("sample")
    for i in range(repeat):
        print(time_function(sample, P, batch_dim))
    print("sample cupy")
    for i in range(repeat):
        print(time_function(sample_cupy, P, batch_dim))
    print("sample naive")
    for i in range(repeat):
        print(time_function(naive_sample, (batch_dim, out_dim, inner_dim)))
    print("weighted sum")
    for i in range(repeat):
        print(time_function(weighted_sum, packA.view(batch_dim, out_dim, -1), ws, inner_dim))
    print("unpack")
    for i in range(repeat):
        print(time_function(unpack_test, packA))
    print("bmm naive")
    for i in range(repeat):
        print(time_function(naive_bmm, A.view(batch_dim, out_dim, inner_dim).type(torch.float16),
                            B.view(batch_dim, inner_dim, 1).type(torch.float16)))

def speed_conv():
    repeat = 3

    for batch_dim, out_dim, in_dim, filter_size, input_size, padding, stride in \
        [(2 ** 7, 64, 24, 7, 224, 3, 2),
        # (2 ** 8, 64, 64, 3, 56, 1, 1),
        # (2 ** 8, 128, 128, 3, 28, 1, 1),
        # (2 ** 8, 256, 256, 3, 14, 1, 1),
        (2 ** 7, 512, 512, 3, 7, 1, 1)]:
        # (2 ** 7, 128, 2048, 3, 7, 1, 1)]:
        print(batch_dim, out_dim, in_dim, filter_size, input_size, padding, stride)

        dtype = torch.int32
        dtype_naive = torch.float16
        input = torch.randint(2, size=(batch_dim * input_size ** 2, in_dim), device="cuda", dtype=torch.bool)
        input_int = torch.randint(2, size=(batch_dim, (input_size//2) ** 2, out_dim), device="cuda", dtype=torch.int32)
        input_bias = torch.randint(2, size=((input_size//2) ** 2, out_dim), device="cuda", dtype=torch.int32)
        filter = torch.randint(2, size=(batch_dim * out_dim * filter_size ** 2, in_dim), device="cuda", dtype=torch.bool)
        thresh = torch.randint(filter_size ** 2 * in_dim, size=(batch_dim,out_dim), device="cuda", dtype=torch.int32)
        thresh = (filter_size ** 2 * in_dim // 2) * torch.ones(size=(batch_dim,out_dim), device="cuda", dtype=torch.int32)
        packed_input = pack(input, dtype).view(batch_dim, input_size, input_size, -1)
        packed_filter = pack(filter, dtype).view(batch_dim, out_dim, filter_size, filter_size, -1)
        # print(torch.allclose(pack(input, dtype), pack_cupy(input, dtype)))
        #
        # print(packed_input.shape)
        # naive_input = (2 * (input.type(dtype_naive) - .5)).view(batch_dim, input_size, input_size, in_dim) \
        #     .permute([0, 3, 1, 2]).contiguous().view(1, batch_dim * in_dim, input_size, input_size)
        # naive_filter = (-2 * (filter.type(dtype_naive) - .5)).view(batch_dim , out_dim, filter_size, filter_size, in_dim) \
        #     .permute([0, 1, 4, 2, 3]).contiguous().view(batch_dim * out_dim, in_dim, filter_size, filter_size)
        # small_input = naive_input[:,::32]
        # small_filter = naive_filter[:,::32]

        # P = torch.rand(out_dim * filter_size ** 2,in_dim, device="cuda", dtype=torch.float16)
        P = torch.rand(filter_size ** 2, out_dim * in_dim, device="cuda", dtype=torch.float16)
        # P = torch.ones(out_dim * filter_size ** 2,in_dim, device="cuda", dtype=torch.float16)
        ws = torch.rand(batch_dim, device="cuda", dtype=torch.float16)

        # print(torch.allclose(sample(P, batch_dim), sample_cupy(P, batch_dim)))
        # print(im2col(packed_input, packed_filter).shape)
        # cols = im2col(packed_input, packed_filter)
        cols = im2colcupy(packed_input, packed_filter)
        # print(im2coltexture(packed_input,packed_filter))
        # print(im2colcupy(packed_input,packed_filter))
        # print(torch.allclose(im2col(packed_input, packed_filter),im2colcupy(packed_input,packed_filter)))
        # print(torch.allclose(im2colinput(packed_input, packed_filter),im2colcupy(packed_input,packed_filter)))
        # print(torch.allclose(im2col(packed_input, packed_filter),im2coltexture(packed_input,packed_filter)))
        # print(torch.allclose(conv_cupy(packed_input, packed_filter),conv_old(packed_input,packed_filter)))
        # print(torch.allclose(conv_cupy(packed_input, packed_filter),conv_old(packed_input,packed_filter)))
        # print(torch.allclose(conv_bmm_cupy(packed_input, packed_filter),conv_old(packed_input,packed_filter)))
        # print(torch.allclose(conv_bmmT_cupy(packed_input, packed_filter),conv_old(packed_input,packed_filter)))
        print(torch.allclose(sample(P, batch_dim, dtype),sample_cupy(P, batch_dim, dtype)))
        # print(sample(P, batch_dim, dtype).sum(),sample_cupy(P, batch_dim, dtype).sum())
        # print(sample(P, batch_dim, dtype))
        # print(sample_cupy(P, batch_dim, dtype))
        # print(sample(P, batch_dim, dtype).nelement())
        # exit()
        # print(torch.allclose(booleanOperations.weighted_sum(packed_filter.view(batch_dim, -1, packed_filter.size(-1)), ws, in_dim).abs(),
        #                      weighted_sum(packed_filter.view(batch_dim, -1, packed_filter.size(-1)), ws, in_dim).abs(), rtol=1e-2))
        print("convert")
        for i in range(repeat):
            print(time_function(convert_tensor, cols))
        # print("im2col")
        # for i in range(repeat):
        #     print(time_function(im2col, packed_input, packed_filter, padding, stride))
        print("im2colcupy")
        for i in range(repeat):
            print(time_function(im2colcupy, packed_input, packed_filter, padding, stride))
        print("reorder")
        for i in range(repeat):
            print(time_function(reorder, packed_input, packed_filter, padding, stride))
        print("reorder")
        for i in range(repeat):
            print(time_function(reorder_input, packed_input, packed_filter, padding, stride))
        # print("conv")
        # for i in range(repeat):
        #     print(time_function(conv, packed_input, packed_filter, padding, stride))
        print("conv_bmm_cupy")
        for i in range(repeat):
            print(time_function(conv_bmm_cupy, packed_input, packed_filter, padding, stride))
        print("conv_act")
        for i in range(repeat):
            print(time_function(conv_act, packed_input, packed_filter, thresh, padding, stride))
        # print("conv naive")
        # for i in range(repeat):
        #     print(time_function(conv_naive, naive_input, naive_filter, batch_dim, padding, stride))
        # print("conv naive small")
        # for i in range(repeat):
        #     print(time_function(conv_naive, small_input, small_filter, batch_dim, padding, stride))
        # pack_dtype = torch.int32
        print("pack (vector)")
        for i in range(repeat):
            print(time_function(pack_cupy, input, dtype))
        print("sample cupy")
        for i in range(repeat):
            print(time_function(sample_cupy, P, batch_dim, dtype))
        print("weighted sum_cupy")
        for i in range(repeat):
            print(time_function(booleanOperations.weighted_sum, packed_filter.view(batch_dim, -1, packed_filter.size(-1)), ws, in_dim))
        print("compare")
        for i in range(repeat):
            print(time_function(compare,input_int, input_bias))
        print("maxpool2d")
        for i in range(repeat):
            print(time_function(maxpool2d,packed_input))
        print("maxpool2d naive")
        for i in range(repeat):
            print(time_function(maxpool2d_naive,input.view(batch_dim,input_size,input_size,in_dim)))


# check_result()
# speedtests()
speed_conv()