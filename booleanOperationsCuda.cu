#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


__global__ void cuda_pack8_kernel(torch::PackedTensorAccessor32<int8_t,2,torch::RestrictPtrTraits> ret,
                                 torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> input) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    const int y_input = y * 8;
    if (x < input.size(0)) {

        int end = input.size(1) - y_input;
        int8_t tmp = 0;
        if (end > 7) {
            tmp = 0x01 * input[x][y_input] |
             0x02 * input[x][y_input + 1] |
             0x04 * input[x][y_input + 2] |
             0x08 * input[x][y_input + 3] |
             0x10 * input[x][y_input + 4] |
             0x20 * input[x][y_input + 5] |
             0x40 * input[x][y_input + 6] |
             0x80 * input[x][y_input + 7];
        } else {
            /*
            int c = 1;
            for (int i = 0; i < end; i++) {
                tmp += c * input[x][y_input+i];
                c*=2;
            }*/
        }


        ret[x][y] = tmp;
    }
}

template <typename scalar_t>
__global__ void cuda_pack_kernel(torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> ret,
                                 torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> input, int elementSize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    const int y_input = y * elementSize;
    if (x < input.size(0)) {
        unsigned long long int c = 1;
        int end = input.size(1) - y_input;
        if (end > elementSize) {
            end = elementSize;
        }
        for (int i = 0; i < end; i++) {
            ret[x][y] |= c * input[x][y_input+i];
            c *= 2;
        }
    }
}



template <typename scalar_t>
__global__ void cuda_unpack_kernel(torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> ret,
                               torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input, int elementSize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    const int y_output = y * elementSize;
    if (x < input.size(0)) {
            unsigned long long int c = 1;
            for (int i = 0; i < elementSize; i++) {
                ret[x][y_output+i] = (c & input[x][y]) > 0;
                c *= 2;
            }
        }
}


template <typename scalar_t>
__global__ void cuda_binary_bmm_kernel(torch::PackedTensorAccessor32<int32_t,3,torch::RestrictPtrTraits> C,
                               torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> A,
                               torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> B) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < C.size(0) && y < C.size(1) && z < C.size(2)) {
        int tmp = 0;
        for (int i = 0; i < A.size(2); i++) {
            tmp += __popc(A[x][y][i] ^ B[x][i][z]);
        }
        C[x][y][z] = tmp;
    }
}



torch::Tensor cuda_pack8(torch::Tensor input) {
    const int threads = 1024;
    const dim3 blocks((input.size(0) + threads - 1)/threads, (input.size(1) + 7) / 8);
    auto ret = torch::zeros({input.size(0), (input.size(1)+7)/8}, torch::TensorOptions().dtype(torch::kInt8).device(input.device()));
    cuda_pack8_kernel<<<blocks,threads>>>(
        ret.packed_accessor32<int8_t,2,torch::RestrictPtrTraits>(),
        input.packed_accessor32<bool,2,torch::RestrictPtrTraits>());
    return ret;
}

torch::Tensor cuda_pack(torch::Tensor input, torch::Tensor ret) {
    const int threads = 1024;
    const dim3 blocks((ret.size(0) + threads -1)/threads,ret.size(1));
    AT_DISPATCH_INTEGRAL_TYPES(ret.scalar_type(), "pack_cuda", ([&] {
        cuda_pack_kernel<<<blocks,threads>>>(
            ret.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            input.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
            8 * elementSize(ret.scalar_type()));
    }));
    return ret;
}




torch::Tensor cuda_unpack(torch::Tensor input) {
    int bitsize = 8 * elementSize(input.scalar_type());
    const int threads = 1024;
    const dim3 blocks((input.size(0) + threads - 1) / threads, input.size(1));
    auto ret = torch::zeros({input.size(0), input.size(1) * bitsize}, torch::TensorOptions().dtype(torch::kBool).device(input.device()));
    AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "unpack_cuda", ([&] {
        cuda_unpack_kernel<<<blocks,threads>>>(
                    ret.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
                    input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    bitsize);

    }));
    return ret;
}

torch::Tensor cuda_binary_bmm(torch::Tensor A, torch::Tensor B) {
    auto C = torch::zeros({A.size(0), A.size(1), B.size(2)}, torch::TensorOptions().dtype(torch::kInt32).device(A.device()));
     dim3 threads;
    if (C.size(1) == 1) { //improve later
        threads = dim3(1, 1, 1024);
    } else if (C.size(2) == 1) {
        threads = dim3(1, 1024, 1);
    } else {
        threads = dim3(1, 32, 32);
    }

    const dim3 blocks((C.size(0) + threads.x - 1) / threads.x, (C.size(1) + threads.y - 1) / threads.y, (C.size(2) + threads.z - 1) / threads.z);
    AT_DISPATCH_INTEGRAL_TYPES(A.scalar_type(), "binary_bmm_cuda", ([&] {
            cuda_binary_bmm_kernel<<<blocks,threads>>>(
                        C.packed_accessor32<int32_t,3,torch::RestrictPtrTraits>(),
                        A.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                        B.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>()
                        );

        }));
    return C;
}