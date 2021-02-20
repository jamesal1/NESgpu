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
        l.update(w)
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
        l.update(w)
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
        # l.update(w)
        torch.cuda.synchronize()
        print("off",time.time() - start)


def time_func_median(func, input, times, repeat=1):
    results = []
    for _ in range(times):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeat):
            func(input)
        torch.cuda.synchronize()
        results += [time.time() - start]
    return 1000 * sorted(results)[times//2]

def get_linear_layers(input_dim, output_dim, batch_size, device="cuda", type=torch.float32):
    l = PerturbedLinear(input_dim, output_dim, batch_size).to(device).type(type)
    l2 = PerturbedLinear(input_dim, output_dim, batch_size // 2).to(device).type(type)
    plo = PermutedLinear(input_dim, output_dim, batch_size, permutation="out").to(device).type(type)
    plof = PermutedLinear(input_dim, output_dim, batch_size, permutation="out", options={"combined": True}).to(device).type(type)
    pli = PermutedLinear(input_dim, output_dim, batch_size, permutation="in").to(device).type(type)
    plif = PermutedLinear(input_dim, output_dim, batch_size, permutation="in", options={"combined": True, "allow_repeats":True}).to(device).type(type)
    plo2 = PermutedLinear(input_dim, output_dim, batch_size // 2, permutation="out").to(device).type(type)
    pli2 = PermutedLinear(input_dim, output_dim, batch_size // 2, permutation="in").to(device).type(type)
    plb = PermutedLinear(input_dim, output_dim, batch_size, permutation="both").to(device).type(type)
    nlo = SyntheticLinear(input_dim, output_dim, batch_size, flip="out").to(device).type(type)
    nli = SyntheticLinear(input_dim, output_dim, batch_size, flip="in").to(device).type(type)
    nli2 = SyntheticLinear(input_dim, output_dim, batch_size // 2, flip="in").to(device).type(type)
    nlb = SyntheticLinear(input_dim, output_dim, batch_size, flip="both").to(device).type(type)
    splo = PermutedLinear(input_dim, output_dim, batch_size, permutation="out", out_sparsity=.99).to(device).type(type)
    s2pli = PermutedLinear(input_dim, output_dim, batch_size, permutation="in", in_sparsity=.5).to(device).type(type)
    s10pli = PermutedLinear(input_dim, output_dim, batch_size, permutation="in", in_sparsity=.1).to(device).type(type)
    s10plif = PermutedLinear(input_dim, output_dim, batch_size, permutation="in", in_sparsity=.1, options={"combined": True, "allow_repeats":True}).to(device).type(type)
    s10plo = PermutedLinear(input_dim, output_dim, batch_size, permutation="out", out_sparsity=.1).to(device).type(type)
    s10plof = PermutedLinear(input_dim, output_dim, batch_size, bias=False, permutation="out", out_sparsity=.1, options={"combined": True}).to(device).type(type)
    s10plb = PermutedLinear(input_dim, output_dim, batch_size, permutation="both", in_sparsity=.1, out_sparsity=.1).to(device).type(type)
    return [
            # (l, "Batch Matrix Multiplication"),
            # (l2, "Antithetic Sampling"),
            (pli, "Permuted Sampling"),
            (plif, "Permuted Sampling Combined"),
            (pli2, "Antithetic Permuted Sampling"),
            (plo, "PermutedLinearOut"),
            (plof, "PermutedLinearOutCombined"),
            # (plb, "PermutedLinearBoth"),
            (nli, "Synthetic Sampling"),
            (nli2, "Antithetic Synthetic Sampling"),
            (nlo, "SyntheticLinearOut"),
            (nlb, "SyntheticLinearBoth"),
            (s2pli, "Permuted Sampling(in_sparsity=.5)"),
            (s10pli, "Permuted Sampling(in_sparsity=.1)"),
            (s10plif, "Permuted Sampling Combined(in_sparsity=.1)"),
            (s10plo, "Permuted Sampling(out_sparsity=.1)"),
            (s10plof, "Permuted Sampling Combined(out_sparsity=.1)"),
            # (s10plb, "PermutedLinear(in_sparsity=.1,out_sparsity=.1)")
            ]

def get_conv_layers(input_dim, output_dim, filter_size, batch_size, device="cuda"):
    padding = 1
    l = PerturbedConv2d(input_dim, output_dim, filter_size, batch_size, padding=padding, options={"direct":True}).to(device)
    l2 = PerturbedConv2d(input_dim, output_dim, filter_size, batch_size // 2, padding=padding).to(device)
    plo = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, padding=padding, permutation="out").to(device)
    pli = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, padding=padding, permutation="in").to(device)
    plif = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, padding=padding, permutation="in", options={"combined": True, "allow_repeats": True}).to(device)
    plb = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, padding=padding, permutation="both").to(device)
    nli = SyntheticConv2d(input_dim, output_dim, filter_size, batch_size, padding=padding, flip="in").to(device)
    nlo = SyntheticConv2d(input_dim, output_dim, filter_size, batch_size, padding=padding, flip="out").to(device)
    nlb = SyntheticConv2d(input_dim, output_dim, filter_size, batch_size, padding=padding, flip="both").to(device)
    s2pli = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, padding=padding, permutation="in", in_sparsity=.5).to(device)
    s10pli = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, padding=padding, permutation="in", in_sparsity=.1).to(device)
    s10plif = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, padding=padding, permutation="in", in_sparsity=.1, options={"combined": True, "allow_repeats": True}).to(device)
    s2plo = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, padding=padding, permutation="out", out_sparsity=.5).to(device)
    s10plo = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, padding=padding, permutation="out", out_sparsity=.1).to(device)
    splb = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, padding=padding, permutation="both",out_sparsity=.9, in_sparsity=1.0).to(device)
    return [
        (l, "Batch Convolution"),
        (l2, "Antithetic Sampling"),
        # (plo, "Permuted Sampling"),
        # (pli, "Permuted Sampling In"),
        (plif, "Permuted Sampling Combined"),
        (nli, "Synthetic Sampling"),
        (nlo, "SyntheticConv2dOut"),
        (nlb, "SyntheticConv2dBoth"),
        # (s2pli, "Permuted Sampling(in_sparsity=.5)"),
        # (s10pli, "Permuted Sampling(in_sparsity=.1)"),
        (s10plif, "Permuted Sampling Combined(in_sparsity=.1)"),
        # (s2plo, "Permuted Sampling(out_sparsity=.5)"),
        # (s10plo, "Permuted Sampling(out_sparsity=.1)")
    ]

def test_layers(device="cuda", batch_size=1024, times=100, func="forward", type=torch.float32, base="linear"):
    if device == "both":
        device = "cuda"
        add_cpu = True
    else:
        add_cpu = False
    ret = defaultdict(lambda: dict())
    if base == "linear":
        tests = [(256, 256), (512, 512), (1024, 1024)]
        tests = [(256, 256), (512, 512), (512, 10)]
    elif base == "conv":
        tests = [
                (64, 3, (3, 3), (32, 32)),
                (64, 64, (3, 3), (32, 32)),
                (128, 128, (3, 3), (16, 16)),
                (256, 256, (3, 3), (8, 8)),
                (512, 512, (3, 3), (4, 4)),
                # (32, 32, (1, 1), (32, 32)),
                # (1024, 1024, (1, 1), (1, 1))
        ]
    for test in tests:
        if base == "linear":
            output_dim, input_dim = test
            layers = get_linear_layers(input_dim, output_dim, batch_size, device, type)
        elif base == "conv":
            output_dim, input_dim, filter_size, image_size = test
            layers = get_conv_layers(input_dim, output_dim, filter_size, batch_size, device)

        for layer, name in layers:
            layer.allocate_memory()
            layer.set_noise_scale(1.)
            layer.set_seed()
            layer.set_noise()
            layer.perturbed_flag = True
            # print(name)
            if func == "forward":
                if base == "linear":
                    inp = torch.rand(batch_size, input_dim, device=device, dtype=type)
                elif base == "conv":
                    inp = torch.rand(batch_size, input_dim, *image_size, device=device, dtype=type)
            elif func =="update":
                inp = torch.rand(layer.directions, device=device, dtype=type)
            elif func == "set_noise":
                inp = 1
            # if device == "cpu":
            #     break
            ret[str(test)][name] = time_func_median(getattr(layer, func), inp, times)
            layer.free_memory()
        layer.perturbed_flag = False
        if func == "forward":
            ret[str(test)]["Base"] = time_func_median(getattr(layer, func), inp, times)
            repeat = min(1024, batch_size)
            ret[str(test)]["Naive"] = time_func_median(getattr(layer, func), inp[:1], times, repeat=repeat) \
                * (batch_size // repeat)
            if add_cpu:
                torch.set_num_threads(1)
                layer = layer.to("cpu")
                inp = inp.to("cpu")
                ret[str(test)]["Naive (CPU)"] = time_func_median(getattr(layer, func), inp[:1], times) * batch_size
    return ret

def to_markdown(result_dict):
    import pandas
    import json
    j = json.dumps(result_dict)
    df = pandas.read_json(json.dumps(result_dict))
    print("### Time (ms, lower is better)")
    print(df.round(3).to_markdown())

    naive_dict = {}
    for test, test_dict in result_dict.items():
        if "Naive" not in test_dict:
            break
        naive_time = test_dict["Naive"]
        naive_dict[test] = dict([(name, naive_time / time) for name, time in test_dict.items()])
    else:
        naive_df = pandas.read_json(json.dumps(naive_dict))
        print("### Speed multiplier vs. naive (higher is better)")
        print(naive_df.round(2).to_markdown())

    base_dict = {}
    for test, test_dict in result_dict.items():
        if "Naive" not in test_dict:
            break
        base_time = test_dict["Base"]
        base_dict[test] = dict([(name, time / base_time) for name, time in test_dict.items()])
    else:
        base_df = pandas.read_json(json.dumps(base_dict))
        print("### Time multiplier vs. base (lower is better)")
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


def test_combined(device="cuda", batch_size=1024):
    for test in \
            [(16, 16, (3, 3), (64, 64)),
             (32, 32, (3, 3), (64, 64)),
             (64, 64, (3, 3), (32, 32)),
             (128, 128, (3, 3), (16, 16)),
             (256, 256, (3, 3), (8, 8)),
             (32, 32, (1, 1), (32, 32)),
             (1024, 1024, (1, 1), (1, 1))]:
        output_dim, input_dim, filter_size, image_size = test
        layer = PermutedConv2d(input_dim, output_dim, filter_size, batch_size, permutation="in", in_sparsity=.1, options={"combined": True}).to(device)
        layer.set_noise(1e-5)
        layer.perturbed_flag = True
        inp = torch.rand(batch_size, input_dim, *image_size, device=device)
        a = layer.forward(inp)
        layer.options = {}
        b = layer.forward(inp)
        print(torch.allclose(a,b))
        print(torch.max(torch.abs(a-b)))


if __name__ == "__main__":
    with torch.no_grad():
        pass
        # index_select_vs_gather(size=(1024, 256))
        # torch.set_num_threads(1)
        # to_markdown(test_layers(batch_size=1024, func="set_noise", base="conv"))
        # to_markdown(test_layers(batch_size=1024, func="update", base="conv"))
        # to_markdown(test_layers(batch_size=512, base="conv"))
        batch_size = 2 ** 18
        to_markdown(test_layers(batch_size=batch_size, type=torch.float16))
        # to_markdown(test_layers(batch_size=1024, type=torch.float16, func="set_noise"))
        to_markdown(test_layers(batch_size=batch_size, type=torch.float16, func="update"))
        # to_markdown(test_layers(batch_size=batch_size))
        # to_markdown(test_layers(batch_size=1024, func="set_noise"))
        # to_markdown(test_layers(batch_size=batch_size, func="update"))
        # to_markdown(test_layers(base="conv", batch_size=1024, func="set_noise"))
        # to_markdown(test_layers(base="conv", batch_size=1024))
        # to_markdown(test_layers(base="conv", batch_size=1024 * 256, func="update"))
        # test_combined()
        # to_markdown(test_conv("cuda", batch_size=1024))
        # to_markdown(test_linear("cuda", batch_size=1024, type=torch.float16))
        # to_markdown(test_linear("both", batch_size=1024 * 4))
        # to_markdown(test_linear(batch_size=1024, func="update"))
        # speed_test_conv("cuda")
        # lin_vs_conv("cpu")
        # randperm_speed("cuda")
        # sparse_test("cuda")
        # test_perm_conv("cuda")
        # to_markdown(test_linear("cuda"))
        # to_markdown(test_conv("cuda"))
        # test_linear("cuda")
        # test_add("cuda")