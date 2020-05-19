from modulesBinary import *
import time
import torch

def time_function(fun, *args):
    torch.cuda.synchronize()
    start = time.time()
    fun(*args)
    torch.cuda.synchronize()
    return time.time() - start


def binTest(device="cuda",times=10):
    def binHelp(a,b):
        return a ^ b
    l = 2 ** 30
    for dtype in [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64]:
        for _ in range(times):
            bit_size = 1 if dtype == torch.bool else torch.iinfo(dtype).bits
            a = torch.zeros(l // bit_size, device=device, dtype=dtype)
            b = torch.zeros(l // bit_size, device=device, dtype=dtype)
            print(dtype, time_function(binHelp,a,b))


def linearTest(device="cuda"):
    layer = BinarizedLinear(5, 10, 20).to(device)
    init_weights(layer)
    layer.set_noise(1)
    layer.perturbed_flag = True
    print(layer.forward(torch.zeros(20, 10, dtype=torch.bool, device=device)))


def extractBitTest(device="cuda"):
    layer = ExtractBits(torch.int8)
    print(layer.forward(torch.arange(10, dtype=torch.int8)))
binTest()
linearTest()
extractBitTest()