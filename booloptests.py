import torch
from torch.utils.cpp_extension import load
import time
def time_function(fun, *args):
    torch.cuda.synchronize()
    start = time.time()
    fun(*args)
    torch.cuda.synchronize()
    return time.time() - start


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

def conv(packed_input, packed_filter):
    padding = 2
    stride = 1
    return boolop_cuda.binary_batch_conv2d(packed_input, packed_filter, padding, padding, stride, stride)

def conv_old(packed_input, packed_filter):
    padding = 2
    stride = 1
    return boolop_cuda.binary_batch_conv2d_old(packed_input, packed_filter, padding, padding, stride, stride)

def conv_naive(naive_input, naive_filter, batch_dim):
    padding = 2
    stride = 1
    return torch.conv2d(torch.nn.functional.pad(naive_input,(padding,) * 4, value=-1), naive_filter, stride=(stride, stride), groups=batch_dim)

def check_result():
    batch_dim = 1024
    out_dim = 65
    in_dim = 63
    input_size = 17
    dtype = torch.int64
    for filter_size in [1,3,5,7]:
        for padding in range(3):
            for stride in range(1,5):
                input = torch.randint(2, size=(batch_dim * input_size ** 2, in_dim), device="cuda", dtype=torch.bool)
                filter = torch.randint(2, size=(batch_dim * out_dim * filter_size ** 2, in_dim), device="cuda", dtype=torch.bool)
                # input.fill_(1)
                # filter.fill_(0)
                packed_input = pack(input, dtype).view(batch_dim, input_size, input_size, -1)
                packed_filter = pack(filter, dtype).view(batch_dim, out_dim, filter_size, filter_size, -1)
                cuda_res = boolop_cuda.binary_batch_conv2d(packed_input, packed_filter, padding, padding, stride, stride)
                naive_input = (2 * (input.type(torch.float16) - .5)).view(batch_dim, input_size, input_size, in_dim)\
                    .permute([0, 3, 1, 2]).contiguous().view(1, batch_dim * in_dim, input_size, input_size)
                naive_filter = (-2 * (filter.type(torch.float16) - .5)).view(batch_dim , out_dim, filter_size, filter_size, in_dim) \
                    .permute([0, 1, 4, 2, 3]).contiguous().view(batch_dim * out_dim, in_dim, filter_size, filter_size)
                naive_res = torch.conv2d(torch.nn.functional.pad(naive_input,(padding,) * 4, value=-1), naive_filter, stride=(stride, stride), groups=batch_dim)
                # print(naive_input, naive_filter)
                # print(naive_res.shape)
                naive_res = naive_res.view(batch_dim, out_dim, *naive_res.shape[-2:]).permute([0, 2, 3, 1])
                print(filter_size, padding, stride)
                # print(cuda_res.shape, naive_res.shape)
                # print(cuda_res.sum(dim=[1,2,3]))
                print(torch.allclose((cuda_res * 2 - (filter_size ** 2 * in_dim)).type(torch.float16), naive_res))
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
    batch_dim = 2 ** 10
    out_dim = 128
    in_dim = 128
    filter_size = 3
    input_size = 64
    dtype = torch.int32
    input = torch.randint(2, size=(batch_dim * input_size ** 2, in_dim), device="cuda", dtype=torch.bool)
    filter = torch.randint(2, size=(batch_dim * out_dim * filter_size ** 2, in_dim), device="cuda", dtype=torch.bool)
    packed_input = pack(input, dtype).view(batch_dim, input_size, input_size, -1)
    packed_filter = pack(filter, dtype).view(batch_dim, out_dim, filter_size, filter_size, -1)
    naive_input = (2 * (input.type(torch.float16) - .5)).view(batch_dim, input_size, input_size, in_dim) \
        .permute([0, 3, 1, 2]).contiguous().view(1, batch_dim * in_dim, input_size, input_size)
    naive_filter = (-2 * (filter.type(torch.float16) - .5)).view(batch_dim , out_dim, filter_size, filter_size, in_dim) \
        .permute([0, 1, 4, 2, 3]).contiguous().view(batch_dim * out_dim, in_dim, filter_size, filter_size)
    dtype = torch.int64
    P = torch.rand(out_dim * filter_size ** 2,in_dim, device="cuda", dtype=torch.float16)
    ws = torch.rand(batch_dim, device="cuda", dtype=torch.float16)

    print("conv")
    for i in range(repeat):
        print(time_function(conv, packed_input, packed_filter))
    print("conv old")
    for i in range(repeat):
        print(time_function(conv_old, packed_input, packed_filter))
    print("conv naive")
    for i in range(repeat):
        print(time_function(conv_naive, naive_input, naive_filter, batch_dim))
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