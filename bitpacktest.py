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

boolop_cuda.unpack(torch.ones(5,8,device="cuda",dtype=torch.int32))


def pack(input, dtype=torch.int32):
    bits = torch.iinfo(dtype).bits
    return boolop_cuda.pack(input,  bits)
#
# print(boolop_cuda.unpack(pack(torch.ones(1, 65, device="cuda",dtype=torch.bool),torch.int64)))
# print(pack(torch.ones(1, 65, device="cuda", dtype=torch.bool),torch.int64))

# print(boolop_cuda.binary_bmm(3 * torch.ones(5,32,32, device="cuda", dtype=torch.int32), torch.ones(5,32,32, device="cuda", dtype=torch.int32)))

mult = 4
batch_dim = 2 ** 10
out_dim = 2 ** 7
inner_dim = 2 ** 13
repeat = 5
A = torch.randint(2, size=(batch_dim * out_dim, inner_dim), device="cuda", dtype=torch.bool)
A2 = torch.randint(2, size=(batch_dim * out_dim * 4, inner_dim // 4), device="cuda", dtype=torch.bool)
B = torch.randint(2, size=(batch_dim, inner_dim), device="cuda", dtype=torch.bool)
dtype = torch.int64
packA = pack(A, dtype)
packB = pack(B, dtype)
print(packA.shape)
print(packB.shape)
# P = torch.rand(out_dim,inner_dim, device="cuda", dtype=torch.float32)
P = torch.rand(out_dim,inner_dim, device="cuda", dtype=torch.float16)

def bmm_only(A, B):
    return boolop_cuda.binary_bmm(packA.view(batch_dim, out_dim, -1), packB.view(batch_dim, -1, 1))

def seeded_bmv(A,B):
    return boolop_cuda.binary_seeded_bmv(P,packB, 999)

def bmm(A, B):
    packA = pack(A).view(batch_dim, out_dim, -1)
    packB = pack(B).view(batch_dim, -1, 1)
    return boolop_cuda.binary_bmm(packA, packB)

def pack_only(A, B):
    packA = pack(A,dtype=dtype)
    # packA = pack(A,dtype=torch.int8)
    # packB = pack(B,dtype=torch.int8)

def pack32_only(A, B):
    packA = pack(A,dtype=torch.int32)
    packB = pack(B,dtype=torch.int32)

def pack8_only(A, B):
    packA = boolop_cuda.pack8(A)
    # packB = boolop_cuda.pack8(B)


def unpack_only(A, B):
    boolop_cuda.unpack(packA)
    boolop_cuda.unpack(packB)

def sample(A, B):
    
    return boolop_cuda.sample_bits(P, batch_dim, 8, 999)


def naive_bmm(A,B):
    return torch.bmm(A.view(batch_dim, out_dim, inner_dim).type(torch.float16),
                     B.view(batch_dim, inner_dim, 1).type(torch.float16))

ws = torch.rand(batch_dim, device="cuda", dtype=torch.float16)
# print(ws)
def weighted_sum(A,B):
    return boolop_cuda.binary_weighted_sum(packA.view(batch_dim, out_dim, -1),ws, inner_dim)

print(bmm(A,B).shape)
print(seeded_bmv(A,B).shape)
print(weighted_sum(A,B))
# print("bmm")
# for i in range(repeat):
#     print(time_function(bmm,A,B))
print("bmm_only")
for i in range(repeat):
    print(time_function(bmm_only,A,B))
print("seeded_bmv")
for i in range(repeat):
    print(time_function(seeded_bmv,A,B))
print("pack")
for i in range(repeat):
    print(time_function(pack_only,A,B))
# print("pack32")
# for i in range(repeat):
#     print(time_function(pack32_only,A,B))
print("pack8")
for i in range(repeat):
    print(time_function(pack8_only,A,B))


print("sample")
for i in range(repeat):
    print(time_function(sample,A,B))

print("weighted sum")
for i in range(repeat):
    print(time_function(weighted_sum,A,B))

# print(pack(A,dtype=torch.int8))
# print(boolop_cuda.pack8(A))
print("unpack")
for i in range(repeat):
    print(time_function(unpack_only,A,B))
print("naive")
for i in range(repeat):
    print(time_function(naive_bmm,A,B))
