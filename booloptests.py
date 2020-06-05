import torch
from torch.utils.cpp_extension import load
import time
from extensions import booleanOperations
def time_function(fun, *args):
    torch.cuda.synchronize()
    start = time.time()
    fun(*args)
    torch.cuda.synchronize()
    res = time.time() - start
    return res


boolop_cuda = load(name="boolop_cuda", sources=["extensions/booleanOperations.cpp","extensions/booleanOperationsCuda.cu"])
boolutil_cuda = load(name="boolutil_cuda", sources=["extensions/booleanUtilities.cpp","extensions/booleanUtilitiesCuda.cu"])


def pack(input, dtype=torch.int32):
    bits = torch.iinfo(dtype).bits
    return boolutil_cuda.pack(input,  bits)


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

def sample(P, batch_dim):
    return boolutil_cuda.sample_bits(P, batch_dim, 64, 999)

def naive_sample(size):
    return torch.randint(2, size=size, device="cuda", dtype=torch.bool)

def naive_bmm(A,B):
    return torch.bmm(A, B)

def weighted_sum(packA, ws, inner_dim):
    return boolutil_cuda.binary_weighted_sum(packA, ws, inner_dim)

def conv(packed_input, packed_filter, padding=1, stride=1):
    packed_filter_reordered = packed_filter.permute([0, 2, 3, 4, 1]).view(packed_filter.size(0), -1, packed_filter.size(1)).contiguous()
    return boolop_cuda.binary_batch_conv2d(packed_input, packed_filter_reordered, packed_filter.size(2), packed_filter.size(3), padding, padding, stride, stride)

def reorder(packed_input, packed_filter, padding=1, stride=1):
    packed_filter_reordered = packed_filter.permute([0, 2, 3, 4, 1]).view(packed_filter.size(0), -1, packed_filter.size(1)).contiguous()

def conv_old(packed_input, packed_filter, padding=1, stride=1):
    return boolop_cuda.binary_batch_conv2d_old(packed_input, packed_filter, padding, padding, stride, stride)

def conv_old_shared(packed_input, packed_filter, padding=1, stride=1):
    return boolop_cuda.binary_batch_conv2d_old_shared(packed_input, packed_filter, padding, padding, stride, stride)

def im2col(packed_input, packed_filter, padding=1, stride=1):
    return boolop_cuda.batch_im2col(packed_input, packed_filter.size(2), packed_filter.size(3), padding, padding, stride, stride)

def im2colpycuda(packed_input, packed_filter, padding=1, stride=1):
    return booleanOperations.batch_im2col(packed_input, packed_filter.size(2), packed_filter.size(3), padding, padding, stride, stride)



def conv_naive(naive_input, naive_filter, batch_dim, padding=1, stride=1):
    return torch.conv2d(torch.nn.functional.pad(naive_input,(padding,) * 4, value=-1), naive_filter, stride=(stride, stride), groups=batch_dim)

def check_result():
    # batch_dim = 1024
    # out_dim = 65
    # in_dim = 63
    # input_size = 17
    batch_dim = 128
    out_dim = 65
    in_dim = 64
    input_size = 32
    dtype = torch.int64
    for filter_size in [1, 3, 7]:
        for padding in range(3):
            for stride in range(1,4):
                # padding = 1
                # filter_size = 3
                print(filter_size, padding, stride)

                input = torch.randint(2, size=(batch_dim * input_size ** 2, in_dim), device="cuda", dtype=torch.bool)
                filter = torch.randint(2, size=(batch_dim * out_dim * filter_size ** 2, in_dim), device="cuda", dtype=torch.bool)
                # input.fill_(1)
                # filter.fill_(0)



                packed_input = pack(input, dtype).view(batch_dim, input_size, input_size, -1)
                packed_filter = pack(filter, dtype).view(batch_dim, out_dim, filter_size, filter_size, -1)

                # packed_input = 1 + torch.arange(packed_input.nelement(), device="cuda", dtype=torch.int64).view_as(packed_input)
                # print(packed_input.squeeze())
                # print(boolop_cuda.batch_im2col(packed_input, filter_size, filter_size, padding, padding, stride, stride))
                # print(boolop_cuda.batch_im2col_old(packed_input, filter_size, filter_size, padding, padding, stride, stride))
                packed_filter = packed_filter.permute([0, 2, 3, 4, 1]).view(batch_dim, -1, out_dim).contiguous()
                cuda_res = boolop_cuda.binary_batch_conv2d(packed_input, packed_filter, filter_size, filter_size, padding, padding, stride, stride)
                naive_input = (2 * (input.type(torch.float16) - .5)).view(batch_dim, input_size, input_size, in_dim)\
                    .permute([0, 3, 1, 2]).contiguous().view(1, batch_dim * in_dim, input_size, input_size)
                naive_filter = (-2 * (filter.type(torch.float16) - .5)).view(batch_dim , out_dim, filter_size, filter_size, in_dim) \
                    .permute([0, 1, 4, 2, 3]).contiguous().view(batch_dim * out_dim, in_dim, filter_size, filter_size)
                naive_res = torch.conv2d(torch.nn.functional.pad(naive_input,(padding,) * 4, value=-1), naive_filter, stride=(stride, stride), groups=batch_dim)
                # print(naive_input, naive_filter)
                # print(naive_res.shape)
                naive_res = naive_res.view(batch_dim, out_dim, *naive_res.shape[-2:]).permute([0, 2, 3, 1])

                # print(cuda_res.shape, naive_res.shape)
                # print(cuda_res.sum(dim=[1,2,3]))
                print(torch.allclose((cuda_res * 2 - (filter_size ** 2 * in_dim)).type(torch.float16).view_as(naive_res), naive_res))
                # print((cuda_res * 2 - (filter_size ** 2 * in_dim)), naive_res)
                # exit()
# check_result()

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
    dtype = torch.int64
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
    repeat = 5
    batch_dim = 2 ** 5
    out_dim = 64
    in_dim = 4 * 24
    filter_size = 7
    input_size = 224
    padding = 3
    stride = 2
    # batch_dim = 2 ** 10
    # out_dim = 64
    # in_dim = 64
    # filter_size = 3
    # input_size = 56
    # padding = 1
    # stride = 1
    # batch_dim = 2 ** 10
    # out_dim = 128
    # in_dim = 128
    # filter_size = 3
    # input_size = 28
    # padding = 1
    # stride = 1
    # batch_dim = 2 ** 10
    # out_dim = 256
    # in_dim = 256
    # filter_size = 3
    # input_size = 14
    # padding = 1
    # stride = 1
    # repeat = 5
    # batch_dim = 1
    # out_dim = 64
    # in_dim = 256 * 512
    # filter_size = 3
    # input_size = 64
    # repeat = 5
    # batch_dim = 2 ** 8
    # out_dim = 1024 // 16
    # in_dim = 256
    # filter_size = 3
    # input_size = 16 * 8
    dtype = torch.int64
    dtype_naive = torch.float32
    input = torch.randint(2, size=(batch_dim * input_size ** 2, in_dim), device="cuda", dtype=torch.bool)
    filter = torch.randint(2, size=(batch_dim * out_dim * filter_size ** 2, in_dim), device="cuda", dtype=torch.bool)
    packed_input = pack(input, dtype).view(batch_dim, input_size, input_size, -1)
    packed_filter = pack(filter, dtype).view(batch_dim, out_dim, filter_size, filter_size, -1)
    naive_input = (2 * (input.type(dtype_naive) - .5)).view(batch_dim, input_size, input_size, in_dim) \
        .permute([0, 3, 1, 2]).contiguous().view(1, batch_dim * in_dim, input_size, input_size)
    naive_filter = (-2 * (filter.type(dtype_naive) - .5)).view(batch_dim , out_dim, filter_size, filter_size, in_dim) \
        .permute([0, 1, 4, 2, 3]).contiguous().view(batch_dim * out_dim, in_dim, filter_size, filter_size)
    small_input = naive_input[:,::32]
    small_filter = naive_filter[:,::32]
    dtype = torch.int64
    P = torch.rand(out_dim * filter_size ** 2,in_dim, device="cuda", dtype=torch.float16)
    ws = torch.rand(batch_dim, device="cuda", dtype=torch.float16)
    print(im2col(packed_input, packed_filter).shape)
    print(torch.allclose(im2col(packed_input, packed_filter),im2colpycuda(packed_input,packed_filter)))
    print("im2col")
    for i in range(repeat):
        print(time_function(im2col, packed_input, packed_filter, padding, stride))
    print("im2colpycuda")
    for i in range(repeat):
        print(time_function(im2colpycuda, packed_input, packed_filter, padding, stride))
    print("reorder")
    for i in range(repeat):
        print(time_function(reorder, packed_input, packed_filter, padding, stride))
    print("conv")
    for i in range(repeat):
        print(time_function(conv, packed_input, packed_filter, padding, stride))
    print("conv old")
    for i in range(repeat):
        print(time_function(conv_old, packed_input, packed_filter, padding, stride))
    print("conv old shared")
    for i in range(repeat):
        print(time_function(conv_old_shared, packed_input, packed_filter, padding, stride))
    print("conv naive")
    for i in range(repeat):
        print(time_function(conv_naive, naive_input, naive_filter, batch_dim, padding, stride))
    print("conv naive small")
    for i in range(repeat):
        print(time_function(conv_naive, small_input, small_filter, batch_dim, padding, stride))
    # print("pack")
    # for i in range(repeat):
    #     print(time_function(pack_test, filter, dtype))
    # print("pack32")
    # for i in range(repeat):
    #     print(time_function(pack32_test, filter))
    # print("pack8")
    # for i in range(repeat):
    #     print(time_function(pack8_test, filter))
    # print("pack (vector)")
    # for i in range(repeat):
    #     print(time_function(pack_test, input, dtype))
    # print("pack32 (vector)")
    # for i in range(repeat):
    #     print(time_function(pack32_test, input))
    # print("pack8 (vector)")
    # for i in range(repeat):
    #     print(time_function(pack8_test, input))
    # print("sample")
    # for i in range(repeat):
    #     print(time_function(sample, P, batch_dim))
    # print("weighted sum")
    # for i in range(repeat):
    #     print(time_function(weighted_sum, packed_filter.view(batch_dim, -1, packed_filter.size(-1)), ws, in_dim))


# speedtests()
speed_conv()