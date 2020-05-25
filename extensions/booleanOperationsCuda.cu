#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>


//can't use __clz
int next_pow2(int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

int next_pow2to1024(int v) {
    if (v>512)
        return 1024;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

int ceil_div (int a, int b) {
    return (a + b - 1) / b;

}


__global__ void cuda_pack8_kernel(torch::PackedTensorAccessor32<int8_t,2,torch::RestrictPtrTraits> ret,
                                 torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> input) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int y_input = y * 8;
    if (x < input.size(0)) {

        int end = input.size(1) - y_input;
        int8_t tmp = 0;
        if (end>7) {
            end = 8;
        }
        int c = 1;
        for (int i = 0; i < end; i++) {
            tmp += c * input[x][y_input+i];
            c*=2;
        }
        ret[x][y] = tmp;
    }
}


//slower than non-templated version for some reason
template <typename scalar_t>
__global__ void cuda_pack_kernel(torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> ret,
                                 torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> input,
                                 int elementSize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int y_input = y * elementSize;
    if (x < ret.size(0) && y < ret.size(1)) {
        scalar_t c = 1;
        int end = input.size(1) - y_input;
        if (end > elementSize) {
            end = elementSize;
        }
        scalar_t tmp = 0;
        for (int i = 0; i < end; i++) {
            tmp |= c * input[x][y_input+i];
            c *= 2;
        }
        ret[x][y] = tmp;
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
            tmp += __popcll(A[x][y][i] ^ B[x][i][z]);
        }
        C[x][y][z] = tmp;
    }
}

template <typename scalar_t>
__global__ void cuda_binary_seeded_bmv_kernel(torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> C,
                               torch::PackedTensorAccessor32<torch::Half,2,torch::RestrictPtrTraits> A,
                               torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> B,
                               int elementSize,
                               unsigned long seed) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int ylen = blockDim.y * gridDim.y;
    const int seq = x * ylen + y;
    curandState state;
    curand_init(seed + seq, 0, 0, &state);
    if (x < C.size(0) && y < C.size(1)) {
        int tmp = 0;
        for (int i = 0; i < B.size(1); i++) {
            scalar_t c = 1;
            int i_bits = i * elementSize;
            int end = A.size(1) - i_bits;
            if (end > elementSize) {
                end = elementSize;
            }
            scalar_t Axyi = 0;
            for (int j = 0; j < end; j++) {
                Axyi |= c * ( __half2float(A[y][i_bits + j]) > curand_uniform(&state));
                c *= 2;
            }
            tmp += __popcll(Axyi ^ B[x][i]);
        }
        C[x][y] = tmp;
    }
}


template <typename scalar_t>
__global__ void cuda_sample_bits_kernel(torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> ret,
                                        const torch::PackedTensorAccessor32<torch::Half,2,torch::RestrictPtrTraits> input,
                                        const int elementSize,
                                        const unsigned long seed) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y; //thank whoever made 64 max threads for z
    const int y = blockIdx.z * blockDim.z + threadIdx.z;
    const int zlen = blockDim.y * gridDim.y;
    const int ylen = blockDim.z * gridDim.z;
    const int seq = x * ylen * zlen + y * zlen  + z;
    curandState state;
    curand_init(seed + seq, 0, 0, &state);
    const int z_input = z * elementSize;
    if (x < ret.size(0) && y < ret.size(1) && z < ret.size(2)) {
        scalar_t c = 1;
        int end = input.size(1) - z_input;
        if (end > elementSize) {
            end = elementSize;
        }
        scalar_t tmp = 0;
        for (int i = 0; i < end; i++) {
            tmp |= c * ( __half2float(input[y][z_input+i]) > curand_uniform(&state));
            //tmp |= c * ( (input[y][z_input+i]) > curand_normal(&state));
            //tmp |= c * ( (input[y][z_input+i]) > .5);
            c *= 2;
        }
        ret[x][y][z] = tmp;
    }
}

template <typename scalar_t>
__global__ void cuda_binary_weighted_sum_kernel(torch::PackedTensorAccessor32<torch::Half,2,torch::RestrictPtrTraits> ret,
                                    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
                                    torch::PackedTensorAccessor32<torch::Half,1,torch::RestrictPtrTraits> weights,
                                    int elementSize) {
    //const int y = blockIdx.y * blockDim.y + threadIdx.y;
    //const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < ret.size(0) && z < ret.size(1)) {
        const int z_input = z / elementSize;
        const int z_bit = z - z_input * elementSize;
        scalar_t c = 1;
        c <<= z_bit;
        torch::Half tmp(0);
        for (int i = 0; i < input.size(0); i++) {
            tmp += weights[i] * ((input[i][y][z_input] & c) >> z_bit);
        }
        ret[y][z] = tmp;
    }
}


torch::Tensor cuda_pack8(torch::Tensor input) {
    const int ret_size1 = (input.size(1) + 7) / 8;
    const int threadsy = ret_size1 > 512 ? 1024 : next_pow2(ret_size1);
    const dim3 threads(1024 / threadsy, threadsy);
    const dim3 blocks((input.size(0) + threads.x - 1)/threads.x, ret_size1 / threadsy);
    auto ret = torch::zeros({input.size(0), ret_size1}, torch::TensorOptions().dtype(torch::kInt8).device(input.device()));
    cuda_pack8_kernel<<<blocks,threads>>>(
        ret.packed_accessor32<int8_t,2,torch::RestrictPtrTraits>(),
        input.packed_accessor32<bool,2,torch::RestrictPtrTraits>());
    return ret;
}




torch::Tensor cuda_pack(torch::Tensor input, torch::Dtype dtype) {
    const int bitsize = 8 * elementSize(dtype);
    const int ret_size1 = (input.size(1) + bitsize - 1) / bitsize;
    const int threadsy = ret_size1 > 512 ? 1024 : next_pow2(ret_size1);
    const dim3 threads(1024 / threadsy, threadsy);
    const dim3 blocks((input.size(0) + threads.x - 1)/threads.x, ret_size1 / threadsy);
    auto ret = torch::zeros({input.size(0), ret_size1}, torch::TensorOptions().dtype(dtype).device(input.device()));
    AT_DISPATCH_INTEGRAL_TYPES(ret.scalar_type(), "pack_cuda", ([&] {
        cuda_pack_kernel<<<blocks,threads>>>(
            ret.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            input.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
            bitsize);
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
    //threads=dim3(1024);
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

torch::Tensor cuda_binary_seeded_bmv(torch::Tensor A, torch::Tensor B, unsigned long seed) {
    const int bitsize = 8 * elementSize(B.scalar_type());
    auto C = torch::zeros({B.size(0), A.size(0)}, torch::TensorOptions().dtype(torch::kInt32).device(A.device()));
    //dim3 threads(1,1024);
    dim3 threads(1024);
    const dim3 blocks((C.size(0) + threads.x - 1) / threads.x, (C.size(1) + threads.y - 1) / threads.y);
    AT_DISPATCH_INTEGRAL_TYPES(B.scalar_type(), "binary_seeded_mm_cuda", ([&] {
            cuda_binary_seeded_bmv_kernel<<<blocks,threads>>>(
                    C.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
                    A.packed_accessor32<torch::Half,2,torch::RestrictPtrTraits>(),
                    B.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    bitsize,
                    seed
                    );

        }));
    return C;
}


torch::Tensor cuda_sample_bits(torch::Tensor p, int n, torch::Dtype dtype, unsigned long seed) {
    const int bitsize = 8 * elementSize(dtype);
    const int ret_size2 = ceil_div(p.size(1), bitsize);
    auto ret = torch::zeros({n, p.size(0), ret_size2}, torch::TensorOptions().dtype(dtype).device(p.device()));
    const int threads2 = next_pow2to1024(ret_size2);
    int threads0 = next_pow2(n);
        if (1024 / threads2 < threads0) {
            threads0 = 1024 / threads2;
        }
    int threads1 = 1024 / threads0 / threads2;
    if (threads1 > 64) {
        threads1 = 64;
    }
    const dim3 threads(threads0, threads2, threads1);
    const dim3 blocks(ceil_div(n, threads.x), ceil_div(ret_size2, threads.y), ceil_div(ret.size(1), threads.z));
    printf("%d %d %d\n",threads.x, threads.y, threads.z);
    printf("%d %d %d\n",blocks.x, blocks.y, blocks.z);

    AT_DISPATCH_INTEGRAL_TYPES(dtype, "sample_bits_cuda", ([&] {
                cuda_sample_bits_kernel<<<blocks,threads>>>(
                            ret.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                            p.packed_accessor32<torch::Half,2,torch::RestrictPtrTraits>(),
                            bitsize,
                            seed);

            }));
    return ret;
}

torch::Tensor cuda_binary_weighted_sum(torch::Tensor input, torch::Tensor weights, int z_bits) {
    const int bitsize = 8 * elementSize(input.scalar_type());
    const int threadsz = next_pow2to1024(z_bits);
    int threadsy = next_pow2(input.size(1));
    if (1024 / threadsz < threadsy) {
        threadsy = 1024 / threadsz;
    }
    const dim3 threads(threadsy, threadsz);
    const dim3 blocks(ceil_div(input.size(1), threadsy), ceil_div(z_bits, threadsz));
    auto ret = torch::zeros({input.size(1), z_bits}, torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));
    AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "binary_weighted_sum_cuda", ([&] {
        cuda_binary_weighted_sum_kernel<<<blocks,threads>>>(
            ret.packed_accessor32<torch::Half,2,torch::RestrictPtrTraits>(),
            input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            weights.packed_accessor32<torch::Half,1,torch::RestrictPtrTraits>(),
            bitsize);
                           }));
    return ret;
}