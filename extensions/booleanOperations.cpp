#include <torch/extension.h>


torch::Tensor cuda_binary_bmm(torch::Tensor A, torch::Tensor B);
torch::Tensor cuda_binary_batch_conv2d(torch::Tensor input, torch::Tensor filter, int filterx, int filtery, int padx, int pady, int stridex, int stridey);
torch::Tensor cuda_binary_seeded_bmv(torch::Tensor A, torch::Tensor B, unsigned long seed);
torch::Tensor cuda_batch_im2col(torch::Tensor input, int filterx, int filtery, int padx, int pady, int stridex, int stridey);

torch::Tensor cuda_binary_batch_conv2d_old(torch::Tensor input, torch::Tensor filter, int padx, int pady, int stridex, int stridey);
torch::Tensor cuda_binary_batch_conv2d_old_shared(torch::Tensor input, torch::Tensor filter, int padx, int pady, int stridex, int stridey);


#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



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


torch::Tensor binary_bmm(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.size(0)==B.size(0), "Batch dim doesn't match");
    TORCH_CHECK(A.size(2)==B.size(1), "Mult dim doesn't match");
    TORCH_CHECK(A.device()==B.device(), "Tensors are on different devices");
    return cuda_binary_bmm(A,B);
}

torch::Tensor binary_batch_conv2d(torch::Tensor input, torch::Tensor filter, int filterx, int filtery, int padx, int pady, int stridex, int stridey) {
    CHECK_INPUT(input);
    CHECK_INPUT(filter);
    TORCH_CHECK(input.size(0)==filter.size(0), "Batch dim doesn't match");
    TORCH_CHECK(input.device()==filter.device(), "Tensors are on different devices");
    TORCH_CHECK(input.scalar_type()==filter.scalar_type(), "Tensors have different types");
    return cuda_binary_batch_conv2d(input, filter, filterx, filtery, padx, pady, stridex, stridey);
}

torch::Tensor batch_im2col(torch::Tensor input, int filterx, int filtery, int padx, int pady, int stridex, int stridey) {
    CHECK_INPUT(input);
    return cuda_batch_im2col(input, filterx, filtery , padx, pady, stridex, stridey);
}


torch::Tensor binary_seeded_bmv(torch::Tensor A, torch::Tensor B, unsigned long seed) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.device()==B.device(), "Tensors are on different devices");
    return cuda_binary_seeded_bmv(A,B, seed);
}

torch::Tensor binary_batch_conv2d_old(torch::Tensor input, torch::Tensor filter, int padx, int pady, int stridex, int stridey) {
    CHECK_INPUT(input);
    CHECK_INPUT(filter);
    TORCH_CHECK(input.size(0)==filter.size(0), "Batch dim doesn't match");
    TORCH_CHECK(input.size(3)==filter.size(4), "number  of input channels don't match");
    TORCH_CHECK(input.device()==filter.device(), "Tensors are on different devices");
    TORCH_CHECK(input.scalar_type()==filter.scalar_type(), "Tensors have different types");
    return cuda_binary_batch_conv2d_old(input, filter, padx, pady, stridex, stridey);
}

torch::Tensor binary_batch_conv2d_old_shared(torch::Tensor input, torch::Tensor filter, int padx, int pady, int stridex, int stridey) {
    CHECK_INPUT(input);
    CHECK_INPUT(filter);
    TORCH_CHECK(input.size(0)==filter.size(0), "Batch dim doesn't match");
    TORCH_CHECK(input.size(3)==filter.size(4), "number  of input channels don't match");
    TORCH_CHECK(input.device()==filter.device(), "Tensors are on different devices");
    TORCH_CHECK(input.scalar_type()==filter.scalar_type(), "Tensors have different types");
    return cuda_binary_batch_conv2d_old_shared(input, filter, padx, pady, stridex, stridey);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("binary_bmm", &binary_bmm, "binary bmm (CUDA)");
  m.def("batch_im2col", &batch_im2col, "batch im2col(CUDA)");
  m.def("binary_batch_conv2d", &binary_batch_conv2d, "binary batch conv2d (CUDA)");
  m.def("binary_seeded_bmv", &binary_seeded_bmv, "binary seeded bmv (CUDA)");

  m.def("binary_batch_conv2d_old", &binary_batch_conv2d_old, "binary batch conv2d (CUDA)");
  m.def("binary_batch_conv2d_old_shared", &binary_batch_conv2d_old_shared, "binary batch conv2d (CUDA)");
}