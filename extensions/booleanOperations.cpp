#include <torch/extension.h>


torch::Tensor cuda_pack8(torch::Tensor input);
torch::Tensor cuda_pack(torch::Tensor input, torch::Dtype dtype);

torch::Tensor cuda_unpack(torch::Tensor input);
torch::Tensor cuda_binary_bmm(torch::Tensor A, torch::Tensor B);
torch::Tensor cuda_binary_batch_conv2d(torch::Tensor input, torch::Tensor filter, int padx, int pady, int stridex, int stridey);
torch::Tensor cuda_binary_seeded_bmv(torch::Tensor A, torch::Tensor B, unsigned long seed);
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

torch::Tensor binary_bmm(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.size(0)==B.size(0), "Batch dim doesn't match");
    TORCH_CHECK(A.size(2)==B.size(1), "Mult dim doesn't match");
    TORCH_CHECK(A.device()==B.device(), "Tensors are on different devices");
    return cuda_binary_bmm(A,B);
}

torch::Tensor binary_batch_conv2d(torch::Tensor input, torch::Tensor filter, int padx, int pady, int stridex, int stridey) {
    CHECK_INPUT(input);
    CHECK_INPUT(filter);
    TORCH_CHECK(input.size(0)==filter.size(0), "Batch dim doesn't match");
    TORCH_CHECK(input.size(3)==filter.size(4), "number  of input channels don't match");
    TORCH_CHECK(input.device()==filter.device(), "Tensors are on different devices");
    TORCH_CHECK(input.scalar_type()==filter.scalar_type(), "Tensors have different types");
    return cuda_binary_batch_conv2d(input, filter, padx, pady, stridex, stridey);
}

torch::Tensor binary_seeded_bmv(torch::Tensor A, torch::Tensor B, unsigned long seed) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.device()==B.device(), "Tensors are on different devices");
    return cuda_binary_seeded_bmv(A,B, seed);
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
  m.def("binary_bmm", &binary_bmm, "binary bmm (CUDA)");
  m.def("binary_batch_conv2d", &binary_batch_conv2d, "binary batch conv2d (CUDA)");
  m.def("binary_seeded_bmv", &binary_seeded_bmv, "binary seeded bmv (CUDA)");
  m.def("sample_bits", &sample_bits, "sample bits (CUDA)");
  m.def("binary_weighted_sum", &binary_weighted_sum, "binary weighted sum (CUDA)");
}