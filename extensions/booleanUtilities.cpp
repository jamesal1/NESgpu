#include <torch/extension.h>


torch::Tensor cuda_pack8(torch::Tensor input);
torch::Tensor cuda_pack(torch::Tensor input, torch::Dtype dtype);

torch::Tensor cuda_unpack(torch::Tensor input);
torch::Tensor cuda_sample_bits(torch::Tensor p, int n, torch::Dtype dtype, unsigned long seed);
torch::Tensor cuda_binary_weighted_sum(torch::Tensor input, torch::Tensor weights, int z_bits);
#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor pack8(torch::Tensor input) {
    CHECK_INPUT(input);
    return cuda_pack8(input);
}


torch::Dtype bitsToIntType(int bits) {
    if (bits == 8) {
        return torch::kInt8;
    } else if (bits == 16) {
        return torch::kInt16;
    }  else if (bits == 32) {
        return torch::kInt32;
    }  else if (bits == 64) {
        return torch::kInt64;
    }  else {
        TORCH_CHECK(false, "invalid bit size");
    }
}

//takes in 2-d tensor, packs last dim
torch::Tensor pack(torch::Tensor input, int bits) {
    CHECK_INPUT(input);

    return cuda_pack(input, bitsToIntType(bits));
}


torch::Tensor unpack(torch::Tensor input) {
    CHECK_INPUT(input);
    return cuda_unpack(input);
}

//takes in 2-d tensor, packs last dim, returns 3-d tensor
torch::Tensor sample_bits(torch::Tensor p, int n, int bits, unsigned long seed) {
    CHECK_INPUT(p);
    return cuda_sample_bits(p, n, bitsToIntType(bits), seed);
}

torch::Tensor binary_weighted_sum(torch::Tensor input, torch::Tensor weights, int z_bits) {
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    TORCH_CHECK(input.device()==weights.device(), "Tensors are on different devices");
    return cuda_binary_weighted_sum(input, weights, z_bits) ;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack8", &pack8, "pack8 (CUDA)");
  m.def("pack", &pack, "pack (CUDA)");
  m.def("unpack", &unpack, "unpack (CUDA)");
  m.def("sample_bits", &sample_bits, "sample bits (CUDA)");
  m.def("binary_weighted_sum", &binary_weighted_sum, "binary weighted sum (CUDA)");
}