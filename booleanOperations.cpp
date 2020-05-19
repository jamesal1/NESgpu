#include <torch/extension.h>


torch::Tensor cuda_pack8(torch::Tensor input);
torch::Tensor cuda_pack(torch::Tensor input, torch::Tensor ret);

torch::Tensor cuda_unpack(torch::Tensor input);
torch::Tensor cuda_binary_bmm(torch::Tensor A, torch::Tensor B);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor pack8(torch::Tensor input) {
    CHECK_INPUT(input);
    return cuda_pack8(input);
}

torch::Tensor pack(torch::Tensor input, torch::Tensor ret) {
    CHECK_INPUT(input);
    CHECK_INPUT(ret);
    return cuda_pack(input,ret);
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack8", &pack8, "pack8 (CUDA)");
  m.def("pack", &pack, "pack (CUDA)");
  m.def("unpack", &unpack, "unpack (CUDA)");
  m.def("binary_bmm", &binary_bmm, "binary_bmm (CUDA)");
}