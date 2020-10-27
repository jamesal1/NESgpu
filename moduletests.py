from modules.base import *
from collections import defaultdict

def lin_vs_conv(device="cpu"):
    times = 5
    d = 2 ** 9
    p = 2
    x = 128
    x_sq = x ** 2
    out = 64
    batch_weight = torch.empty(d * x_sq * out, device=device)
    weight = torch.empty(x_sq * out, device=device)
    input = torch.empty(d * p * x_sq, device=device)
    batch_input_linear = input.view(d, x_sq, p)
    batch_input_conv_x = input.view(p, d, x, x)
    batch_input_conv_1 = input.view(p, d * x_sq, 1, 1)
    input_linear = input.view(x_sq, p * d)
    input_conv_x = input.view(p * d, 1, x, x)
    input_conv_1 = input.view(p * d, x_sq, 1, 1)
    batch_weight_linear = batch_weight.view(d, out, x_sq)
    batch_weight_conv_x = batch_weight.view(d * out, 1, x, x)
    batch_weight_conv_1 = batch_weight.view(d * out, x_sq, 1, 1)
    weight_linear = weight.view(out, x_sq)
    weight_conv_x = weight.view(out, 1, x, x)
    weight_conv_1 = weight.view(out, x_sq, 1, 1)
    for _ in range(2):
        for i in range(times):
            torch.cuda.synchronize()
            start = time.time()
            torch.bmm(batch_weight_linear, batch_input_linear)
            torch.cuda.synchronize()
            print("batch_linear", time.time() - start)
        for i in range(times):
            torch.cuda.synchronize()
            start = time.time()
            torch.mm(weight_linear, input_linear)
            torch.cuda.synchronize()
            print("linear", time.time() - start)
        for i in range(times):
            torch.cuda.synchronize()
            start = time.time()
            F.conv2d(batch_input_conv_x, weight=batch_weight_conv_x, groups=d)
            torch.cuda.synchronize()
            print("batch_conv_x", time.time() - start)
        for i in range(times):
            torch.cuda.synchronize()
            start = time.time()
            F.conv2d(input_conv_x, weight=weight_conv_x)
            torch.cuda.synchronize()
            print("conv_x", time.time() - start)
        for i in range(times):
            torch.cuda.synchronize()
            start = time.time()
            F.conv2d(batch_input_conv_1, weight=batch_weight_conv_1, groups=d)
            torch.cuda.synchronize()
            print("batch_conv_1", time.time() - start)
        for i in range(times):
            torch.cuda.synchronize()
            start = time.time()
            F.conv2d(input_conv_1, weight=weight_conv_1)
            torch.cuda.synchronize()
            print("conv_1", time.time() - start)


def randperm_speed(device="cuda"):
    start = time.time()
    m = 256
    n = 256


    for _ in range(20):

        x = torch.empty(m,n, dtype=torch.long)
        w = torch.zeros(n,n, dtype=torch.int32)
        f = torch.zeros(n,n)
        g = torch.zeros(n,n)
        y = torch.empty(m,n)
        a = torch.arange(n)
        b = torch.zeros(n, dtype=torch.long)
        y.random_()
        # f.random_()
        for i in range(m):
            torch.randperm(n, out=x[i])
        a = a.to(device)
        f = f.to(device)
        g = g.to(device)
        w = w.to(device)
        x = x.to(device)
        y = y.to(device)
        b = b.to(device)
        torch.cuda.synchronize()
        start = time.time()
        torch.argsort(x,dim=1)
        z = torch.gather(y, 1, x)
        # for i in range(m):
        #     g+=f[:,x[i]]

        for i in range(m):
            f[b,b] += 1
        print(f.sum())
        torch.mm(f,f)

        # print(w)
        torch.cuda.synchronize()
        # print(y,z)
        print(time.time() - start)


def test_perm(device="cuda"):
    d = 128
    m = 128
    n = 128
    l = PermutedLinear(n,m,d,permutation="out").to(device)
    # l = PermutedLinear(n,m,d).to(device)

    for _ in range(10):
        l.allocate_memory()
        l.set_noise_scale(1.)
        l.set_seed()
        l.set_noise()
        w = torch.rand(d,device=device)
        # w = torch.ones(d,device=device)
        torch.cuda.synchronize()
        start = time.time()
        l.set_grad(w)
        torch.cuda.synchronize()
        print(time.time() - start)

def test_perm_conv(device="cuda"):
    d = 4
    c_in = 5
    c_out = 5
    h = 32
    w = 32
    x = 3
    y = 3
    inp = torch.empty((d, c_in, h, w), device=device)
    inp.normal_()
    l = PermutedConv2d(c_in,c_out,(x,y),d,permutation="both").to(device)
    # l = PerturbedConv2d(c_in,c_out,(x,y),d).to(device)

    for _ in range(10):
        l.set_noise(1.)
        w = torch.rand(d,device=device)
        # w = torch.ones(d,device=device)
        torch.cuda.synchronize()
        start = time.time()
        l.perturbed_flag = True
        l.forward(inp)
        l.set_grad(w)
        torch.cuda.synchronize()
        print(time.time() - start)
    for _ in range(10):
        l.allocate_memory()
        l.set_noise_scale(1.)
        l.set_seed()
        l.set_noise()
        w = torch.rand(d,device=device)
        # w = torch.ones(d,device=device)
        torch.cuda.synchronize()
        start = time.time()
        l.perturbed_flag = False
        l.forward(inp)
        # l.set_grad(w)
        torch.cuda.synchronize()
        print("off",time.time() - start)


def time_forward_median(layer, input, times, repeat=1):
    results = []
    for _ in range(times):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeat):
            layer.forward(input)
        torch.cuda.synchronize()
        results += [time.time() - start]
    return 1000 * sorted(results)[times//2]

def test_linear(device="cuda", batch_size=1024, times=200):
    ret = defaultdict(lambda:dict())
    for test in [(256, 256), (512, 512), (1024, 1024)]:
        output_dim, input_dim = test
        l = PerturbedLinear(input_dim, output_dim, batch_size).to(device)
        s1 = SparsePerturbedLinear(input_dim, output_dim, batch_size, sparsity=1).to(device)
        s2 = SparsePerturbedLinear(input_dim, output_dim, batch_size, sparsity=2).to(device)
        plo = PermutedLinear(input_dim, output_dim, batch_size, permutation="out").to(device)
        pli = PermutedLinear(input_dim, output_dim, batch_size, permutation="in").to(device)
        plb = PermutedLinear(input_dim, output_dim, batch_size, permutation="both").to(device)
        nlo = SyntheticLinear(input_dim, output_dim, batch_size, flip="out").to(device)
        nli = SyntheticLinear(input_dim, output_dim, batch_size, flip="in").to(device)
        nlb = SyntheticLinear(input_dim, output_dim, batch_size, flip="both").to(device)
        splo = PermutedLinear(input_dim, output_dim, batch_size, permutation="out", out_sparsity=.99).to(device)
        spli = PermutedLinear(input_dim, output_dim, batch_size, permutation="in", in_sparsity=.5).to(device)
        spli2 = PermutedLinear(input_dim, output_dim, batch_size, permutation="in", in_sparsity=.1).to(device)
        splb = PermutedLinear(input_dim, output_dim, batch_size, permutation="both", in_sparsity=.1, out_sparsity=.1).to(device)
        if device == "cuda":
            for layer, name in [(l, "PerturbedLinear"),
                                (s1, "SparsePerturbedLinear (k=1)"), (s2, "SparsePerturbedLinear (k=2)"),
                                (pli, "PermutedLinear"),
                                (plo, "PermutedLinearOut"),
                                (nli, "SyntheticLinear"),
                                (nlo, "SyntheticLinearOut"),
                                (nlb, "SyntheticLinearBoth"),
                                (spli, "PermutedLinear(in_sparsity=.5)"),
                                (spli2, "PermutedLinear(in_sparsity=.1)")
                                ]:
                layer.allocate_memory()
                layer.set_noise_scale(1.)
                layer.set_seed()
                layer.set_noise()

                layer.perturbed_flag = True

                inp = torch.rand(batch_size, input_dim, device=device)
                ret[str(test)][name] = time_forward_median(layer, inp, times)
        l.perturbed_flag = False
        inp = torch.rand(batch_size, input_dim, device=device)
        ret[str(test)]["Base"] = time_forward_median(l, inp, times)
        ret[str(test)]["Naive"] = time_forward_median(l, inp[:1], times, repeat=batch_size)
    return ret

def test_conv(device="cuda", batch_size=1024, times=100):
    ret = defaultdict(lambda:dict())
    for test in \
            [(16, 16, (3, 3), (64, 64)),
             (32, 32, (3, 3), (64, 64)),
             (64, 64, (3, 3), (32, 32)),
             (32, 32, (1, 1), (32, 32)),
             (1024, 1024, (1, 1), (1, 1))]:
        output_dim, input_dim, filter_size, image_size = test
        l = PerturbedConv2d(input_dim, output_dim, filter_size, batch_size).to(device)
        plo = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, permutation="out").to(device)
        pli = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, permutation="in").to(device)
        plb = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, permutation="both").to(device)
        nli = SyntheticConv2d(input_dim, output_dim, filter_size, batch_size, flip="in").to(device)
        nlo = SyntheticConv2d(input_dim, output_dim, filter_size, batch_size, flip="out").to(device)
        nlb = SyntheticConv2d(input_dim, output_dim, filter_size, batch_size, flip="both").to(device)
        splo = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, permutation="out", out_sparsity=.5).to(device)
        splo2 = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, permutation="out", out_sparsity=.1).to(device)
        splb = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, permutation="both",out_sparsity=.9, in_sparsity=1.0).to(device)
        for layer, name in [(l, "PerturbedConv2d"), (plo, "PermutedConv2d"),
                            (nli, "SyntheticConv2dIn"),
                            (nlo, "SyntheticConv2dOut"),
                            (nlb, "SyntheticConv2dBoth"),
                             (splo, "PermutedConv2d(out_sparsity=.5)"),
                             (splo2, "PermutedConv2d(out_sparsity=.1)")
                            ]:
            layer.set_noise(1.)
            layer.perturbed_flag = True
            inp = torch.rand(batch_size, input_dim, *image_size, device=device)
            print(name)
            ret[str(test)][name] = time_forward_median(layer, inp, times)
            try:
                print(layer.in_sparsity)

            except:
                pass
        l.perturbed_flag = False
        ret[str(test)]["Base"] = time_forward_median(l, inp, times)
        ret[str(test)]["Naive"] = time_forward_median(l, inp[:1], times, repeat=batch_size)
    return ret


def to_markdown(result_dict):
    import pandas
    import json
    j = json.dumps(result_dict)
    df = pandas.read_json(json.dumps(result_dict))
    print("### Time (ms, lower is better)")
    print(df.round(3).to_markdown())
    print("### Speed multiplier vs. naive (higher is better)")
    naive_dict = {}
    for test, test_dict in result_dict.items():
        naive_time = test_dict["Naive"]
        naive_dict[test] = dict([(name, naive_time / time) for name, time in test_dict.items()])
    naive_df = pandas.read_json(json.dumps(naive_dict))
    print(naive_df.round(2).to_markdown())
    print("### Time multiplier vs. base (lower is better)")
    base_dict = {}
    for test, test_dict in result_dict.items():
        base_time = test_dict["Base"]
        base_dict[test] = dict([(name, time / base_time) for name, time in test_dict.items()])
    base_df = pandas.read_json(json.dumps(base_dict))
    print(base_df.round(2).to_markdown())
    print()



def index_select_vs_gather(device="cuda", size=(1024, 1024), times=100):
    permutations = torch.empty(size, dtype=torch.long, device=device)
    target = torch.arange(size[0] * size[1], device=device).view(*size)
    target_f32 = torch.arange(size[0] * size[1], device=device).view(*size).type(torch.float)
    target_f32b = torch.arange(size[0] * size[1], device=device).view(*size).type(torch.float)
    target_f16 = torch.arange(size[0] * size[1], device=device).view(*size).type(torch.float16)
    target_i8 = torch.arange(size[0] * size[1], device=device).view(*size).type(torch.int8)
    permutations_1d = (permutations + (torch.arange(size[0], device=device) * size[1]).unsqueeze(1)).flatten()
    for i in range(size[0]):
        torch.randperm(size[1], out=permutations[i])
    for _ in range(10):
        torch.cuda.synchronize()
        start =time.time()
        for _ in range(times):
            perms_1 = target.gather(1, permutations)
        torch.cuda.synchronize()
        print("gather", time.time() - start)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(times):
            perms_2 = torch.index_select(target.flatten(), 0, permutations_1d)
        torch.cuda.synchronize()
        print("select", time.time() - start)
        start = time.time()
        for _ in range(times):
            # perms_2 = torch.index_select(target.view(size[0],size[1]), 0, permutations[0])
            perms_2 = target.view(size[0],size[1])[permutations[0]]
        torch.cuda.synchronize()
        print("select_simple", time.time() - start)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(times):
            flip_1 = target_f32 * target_f32b
        torch.cuda.synchronize()
        print("32", time.time() - start)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(times):
            flip_1 = target_f32 * target_f32b
        torch.cuda.synchronize()
        print("32", time.time() - start)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(times):
            flip_1 = target_f32 * target_f16
        torch.cuda.synchronize()
        print("16", time.time() - start)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(times):
            flip_1 = target_f32 * target_i8
        torch.cuda.synchronize()
        print("8", time.time() - start)




if __name__ == "__main__":
    with torch.no_grad():
        pass
        # index_select_vs_gather(size=(1024, 256))
        # torch.set_num_threads(1)
        # print(test_linear("cpu", batch_size=10))
        # speed_test_conv("cuda")
        # lin_vs_conv("cpu")
        # randperm_speed("cuda")
        # sparse_test("cuda")
        # test_perm_conv("cuda")
        # to_markdown(test_linear("cuda"))
        # to_markdown(test_linear("cuda"))
        to_markdown(test_conv("cuda"))
        # test_linear("cuda")
        # test_add("cuda")