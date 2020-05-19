import torch
from torch.utils.cpp_extension import load
import time
def time_function(fun, *args):
    torch.cuda.synchronize()
    start = time.time()
    fun(*args)
    torch.cuda.synchronize()
    return time.time() - start


boolop_cuda = load(name="boolop_cuda", sources=["booleanOperations.cpp","booleanOperationsCuda.cu"])

boolop_cuda.unpack(torch.ones(5,8,device="cuda",dtype=torch.int32))


def pack(input, dtype=torch.int32):
    bits = torch.iinfo(dtype).bits
    return boolop_cuda.pack(input, torch.zeros(input.size(0),(input.size(1) + bits - 1) // bits, device=input.device, dtype=dtype))

print(boolop_cuda.unpack(pack(torch.ones(1, 65, device="cuda",dtype=torch.bool),torch.int64)))
print(pack(torch.ones(1, 65, device="cuda", dtype=torch.bool),torch.int64))

print(boolop_cuda.binary_bmm(3 * torch.ones(5,32,32, device="cuda", dtype=torch.int32), torch.ones(5,32,32, device="cuda", dtype=torch.int32)))

batch_dim = 2048
out_dim = 256
inner_dim = 1024
repeat = 5
A = torch.randint(2, size=(batch_dim * out_dim, inner_dim), device="cuda", dtype=torch.bool)
B = torch.randint(2, size=(batch_dim, inner_dim), device="cuda", dtype=torch.bool)
packA = pack(A)
packB = pack(B)


def bmm_only(A, B):
    return boolop_cuda.binary_bmm(packA.view(batch_dim, out_dim, -1), packB.view(batch_dim, -1, 1))

def bmm(A, B):
    packA = pack(A).view(batch_dim, out_dim, -1)
    packB = pack(B).view(batch_dim, -1, 1)
    return boolop_cuda.binary_bmm(packA, packB)

def pack_only(A, B):
    packA = pack(A,dtype=torch.int8)
    packB = pack(B,dtype=torch.int8)

def pack8_only(A, B):
    packA = boolop_cuda.pack8(A)
    packB = boolop_cuda.pack8(B)

def unpack_only(A, B):
    boolop_cuda.unpack(packA)
    boolop_cuda.unpack(packB)

def naive_bmm(A,B):
    return torch.bmm(A.view(batch_dim, out_dim, inner_dim).type(torch.float16),
                     B.view(batch_dim, inner_dim, 1).type(torch.float16))

print(bmm(A,B))
for i in range(repeat):
    print(time_function(bmm,A,B))

for i in range(repeat):
    print(time_function(bmm_only,A,B))

for i in range(repeat):
    print(time_function(pack_only,A,B))

for i in range(repeat):
    print(time_function(pack8_only,A,B))

for i in range(repeat):
    print(time_function(unpack_only,A,B))

for i in range(repeat):
    print(time_function(naive_bmm,A,B))
