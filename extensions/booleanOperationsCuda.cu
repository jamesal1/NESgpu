#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 16

//can't use __clz
int next_pow2_clip(int v, int cap) {
    if (v > cap / 2)
        return cap;
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




__global__ void cuda_long_bmm_kernel(torch::PackedTensorAccessor32<int32_t,3,torch::RestrictPtrTraits> C,
                               const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> A,
                               const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> B) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int x_sub = threadIdx.y;
    const int y_sub = threadIdx.x;
    __shared__ long Asub[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ long Bsub[BLOCK_SIZE][BLOCK_SIZE + 1];
    int tmp = 0;

    for (int inner_block = 0; inner_block < (A.size(2) + BLOCK_SIZE - 1) / BLOCK_SIZE; inner_block++) {
        const int inner = inner_block * BLOCK_SIZE;
//        Asub[y_sub][z_sub] = z_sub + inner < A.size(2) ? A[x][y][z_sub + inner] : 0;
//        Bsub[y_sub][z_sub] = y_sub + inner < A.size(2) ? B[x][y_sub + inner][z] : 0;
        Asub[x_sub][y_sub] = A[z][x][y_sub + inner];
        Bsub[x_sub][y_sub] = B[z][x_sub + inner][y];


        __syncthreads();
        for (int j = 0; j < BLOCK_SIZE; j++) {
            tmp += __popcll(Asub[j][x_sub] ^ Bsub[j][y_sub]);
        }
        __syncthreads();
    }


    if (x < C.size(1) && y < C.size(2)) {
            C[z][x][y] = tmp;
    }
}

__global__ void cuda_int_bmm_kernel(torch::PackedTensorAccessor32<int32_t,3,torch::RestrictPtrTraits> C,
                               const torch::PackedTensorAccessor32<int32_t,3,torch::RestrictPtrTraits> A,
                               const torch::PackedTensorAccessor32<int32_t,3,torch::RestrictPtrTraits> B) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int x_sub = threadIdx.y;
    const int y_sub = threadIdx.x;
    __shared__ int Asub[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ int Bsub[BLOCK_SIZE][BLOCK_SIZE + 1];
    int tmp = 0;

    for (int inner_block = 0; inner_block < (A.size(2) + BLOCK_SIZE - 1) / BLOCK_SIZE; inner_block++) {
        const int inner = inner_block * BLOCK_SIZE;
//        Asub[y_sub][z_sub] = z_sub + inner < A.size(2) ? A[x][y][z_sub + inner] : 0;
//        Bsub[y_sub][z_sub] = y_sub + inner < A.size(2) ? B[x][y_sub + inner][z] : 0;
        Asub[x_sub][y_sub] = A[z][x][y_sub + inner];
        Bsub[x_sub][y_sub] = B[z][x_sub + inner][y];
//
//
//        __syncthreads();
//        for (int j = 0; j < BLOCK_SIZE; j++) {
//            tmp += __popc(Asub[y_sub][j] ^ Bsub[j][z_sub]);
//        }
        for (int j = 0; j < BLOCK_SIZE; j++) {
            tmp += __popc(Asub[j][x_sub] ^ Bsub[j][y_sub]);
        }
        __syncthreads();
    }


    if (x < C.size(1) && y < C.size(2)) {
        C[z][x][y] = tmp;
    }
}

template <typename scalar_t>
__global__ void cuda_binary_bmm_kernel(torch::PackedTensorAccessor32<int32_t,3,torch::RestrictPtrTraits> C,
                               const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> A,
                               const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> B) {
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




__global__ void cuda_batch_im2col_kernel(torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> output,
                                                   const torch::PackedTensorAccessor32<long,4,torch::RestrictPtrTraits> input,
                                                   const int filterx, const int filtery,
                                                   const int padx, const int pady,
                                                   const int stridex, const int stridey) {

        const int batch = blockIdx.z * blockDim.z + threadIdx.z;
        const int loc = blockIdx.y * blockDim.y + threadIdx.y;
        const int element = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch < output.size(0) && loc < output.size(1) && element < output.size(2)) {
            const int w = (input.size(2) - filtery + 2 * pady) / stridey + 1;
            const int x_out = loc / w;
            const int y_out = loc - x_out * w;
            const int element_idx = element / input.size(3);
            const int channel = element - element_idx * input.size(3);
            const int i = element_idx / filtery;
            const int j = element_idx - i * filtery;
            const int x_in = i + x_out * stridex - padx;
            const int y_in = j + y_out * stridey - pady;
            output[batch][loc][element] = (x_in >= 0 && x_in < input.size(1) && y_in >= 0 && y_in < input.size(2)) ?
                input[batch][x_in][y_in][channel] : 0;
        }
}

__global__ void cuda_batch_im2colint_kernel(torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> output,
                                                   const torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> input,
                                                   const int filterx, const int filtery,
                                                   const int padx, const int pady,
                                                   const int stridex, const int stridey) {

        const int batch = blockIdx.z * blockDim.z + threadIdx.z;
        const int loc = blockIdx.y * blockDim.y + threadIdx.y;
        const int element = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch < output.size(0) && loc < output.size(1) && element < output.size(2)) {
            const int w = (input.size(2) - filtery + 2 * pady) / stridey + 1;
            const int x_out = loc / w;
            const int y_out = loc - x_out * w;
            const int element_idx = element / input.size(3);
            const int channel = element - element_idx * input.size(3);
            const int i = element_idx / filtery;
            const int j = element_idx - i * filtery;
            const int x_in = i + x_out * stridex - padx;
            const int y_in = j + y_out * stridey - pady;
            output[batch][loc][element] = (x_in >= 0 && x_in < input.size(1) && y_in >= 0 && y_in < input.size(2)) ?
                input[batch][x_in][y_in][channel] : 0;
        }
}



template <typename scalar_t>
__global__ void cuda_binary_seeded_bmv_kernel(torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> C,
                               const torch::PackedTensorAccessor32<torch::Half,2,torch::RestrictPtrTraits> A,
                               const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> B,
                               const int elementSize,
                               const unsigned long seed) {
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



torch::Tensor cuda_binary_bmm(torch::Tensor A, torch::Tensor B) {
    auto C = torch::zeros({A.size(0), A.size(1), B.size(2)}, torch::TensorOptions().dtype(torch::kInt32).device(A.device()));
    const int threadsx = BLOCK_SIZE;
    const int threadsy = BLOCK_SIZE;
    const int threadsz = 1;
    const dim3 threads(threadsx, threadsy, threadsz);
//    const dim3 blocks(ceil_div(C.size(0), threads.x) , ceil_div(C.size(1), threads.y), ceil_div(C.size(2), threads.z));
    const dim3 blocks(ceil_div(C.size(2), threads.x) , ceil_div(C.size(1), threads.y), ceil_div(C.size(0), threads.z));
    if (A.scalar_type() == torch::kInt64) {
    cuda_long_bmm_kernel<<<blocks,threads>>>(
        C.packed_accessor32<int32_t,3,torch::RestrictPtrTraits>(),
        A.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
        B.packed_accessor32<long,3,torch::RestrictPtrTraits>()
        );
    } else {
    cuda_int_bmm_kernel<<<blocks,threads>>>(
        C.packed_accessor32<int32_t,3,torch::RestrictPtrTraits>(),
        A.packed_accessor32<int32_t,3,torch::RestrictPtrTraits>(),
        B.packed_accessor32<int32_t,3,torch::RestrictPtrTraits>()
        );
    }

//    AT_DISPATCH_INTEGRAL_TYPES(A.scalar_type(), "binary_bmm_cuda", ([&] {
//            cuda_binary_bmm_kernel<<<blocks,threads>>>(
//                        C.packed_accessor32<int32_t,3,torch::RestrictPtrTraits>(),
//                        A.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//                        B.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>()
//                        );
//
//        }));

    return C;
}




torch::Tensor cuda_batch_im2col(torch::Tensor input, int filterx, int filtery, int padx, int pady, int stridex, int stridey) {
    const int h = (input.size(1) - filterx + 2 * padx) / stridex + 1;
    const int w = (input.size(2) - filtery + 2 * pady) / stridey + 1;
    auto output = torch::zeros({input.size(0), h * w, filterx * filtery * input.size(3)}, torch::TensorOptions().dtype(input.scalar_type()).device(input.device()));
    const int threadsx = next_pow2_clip(output.size(2), 256);
    const int threadsy = next_pow2_clip(output.size(1), 256 / threadsx);
    const int threadsz = 1;
    const dim3 threads(threadsx, threadsy, threadsz);
    const dim3 blocks(ceil_div(output.size(2) ,threads.x) , ceil_div(output.size(1), threads.y), ceil_div(output.size(0), threads.z));
    if (input.scalar_type() == torch::kInt64) {
    cuda_batch_im2col_kernel<<<blocks,threads>>>(output.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
                                   input.packed_accessor32<long,4,torch::RestrictPtrTraits>(),
                                   filterx, filtery,
                                   padx, pady, stridex, stridey);
    } else {
    cuda_batch_im2colint_kernel<<<blocks,threads>>>(output.packed_accessor32<int32_t,3,torch::RestrictPtrTraits>(),
                                   input.packed_accessor32<int32_t,4,torch::RestrictPtrTraits>(),
                                   filterx, filtery,
                                   padx, pady, stridex, stridey);
    }
        return output;
}


//thread config assumes square input.
torch::Tensor cuda_binary_batch_conv2d(torch::Tensor input, torch::Tensor filter, int filterx, int filtery, int padx, int pady, int stridex, int stridey) {
    torch::Tensor cols = cuda_batch_im2col(input, filterx, filtery, padx, pady, stridex, stridey);
    return cuda_binary_bmm(cols, filter);
}


torch::Tensor cuda_binary_seeded_bmv(torch::Tensor A, torch::Tensor B, unsigned long seed) {
    const int bitsize = 8 * elementSize(B.scalar_type());
    auto C = torch::zeros({B.size(0), A.size(0)}, torch::TensorOptions().dtype(torch::kInt32).device(A.device()));
    //const int threadsy = next_pow2_clip(A.size(1), 1024);
    //const int threadsx = next_pow2_clip(A.size(0), 1024 / threadsy);
    const int threadsx = next_pow2_clip(A.size(0), 1024);
    const int threadsy = next_pow2_clip(A.size(1), 1024 / threadsx);
    const dim3 threads(threadsx, threadsy);
    const dim3 blocks(ceil_div(C.size(0), threads.x) , ceil_div(C.size(1), threads.y));

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


__global__ void cuda_binary_batch_conv2d_kernel_old(torch::PackedTensorAccessor32<int32_t,4,torch::RestrictPtrTraits> output,
                               torch::PackedTensorAccessor32<long,4,torch::RestrictPtrTraits> input,
                               torch::PackedTensorAccessor32<long,5,torch::RestrictPtrTraits> filter,
                               int padx, int pady,
                               int stridex, int stridey) {
        const int idx_bc = blockIdx.y * blockDim.y + threadIdx.y;
        const int idx_xy = blockIdx.x * blockDim.x + threadIdx.x;
        const int x_out = idx_xy / output.size(2);
        const int y_out = idx_xy - x_out * output.size(2);
        const int batch = idx_bc / output.size(3);
        const int ch_out = idx_bc - batch * output.size(3);
        if (batch < output.size(0) && x_out < output.size(1) && y_out < output.size(2) && ch_out < output.size(3)) {
            int tmp = 0;
            for (int ch_in = 0; ch_in < input.size(3); ch_in++) {
                for (int i = 0; i < filter.size(2); i++) {
                    for (int j = 0; j < filter.size(3); j++) {
                        int x_in = i + x_out * stridex - padx;
                        int y_in = j + y_out * stridey - pady;
                        long inp = (x_in >= 0 && x_in < input.size(1) && y_in >= 0 && y_in < input.size(2)) ?
                            input[batch][x_in][y_in][ch_in] : 0;
                        tmp += __popcll(filter[batch][ch_out][i][j][ch_in] ^ inp);
                    }
                }
            }
            output[batch][x_out][y_out][ch_out] = tmp;
        }

}

torch::Tensor cuda_binary_batch_conv2d_old(torch::Tensor input, torch::Tensor filter, int padx, int pady, int stridex, int stridey) {
    const int h = (input.size(1) - filter.size(2) + 2 * padx) / stridex + 1;
    const int w = (input.size(2) - filter.size(3) + 2 * pady) / stridey + 1;
    auto output = torch::zeros({filter.size(0), h, w, filter.size(1)}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
    const int hw = h * w;
    const int bo = filter.size(0) * filter.size(1);
    const int threadsx = next_pow2_clip(hw, 1024);
    const int threadsy = next_pow2_clip(bo, 1024 / threadsx);
    const dim3 threads(threadsx, threadsy);
    const dim3 blocks(ceil_div(hw, threads.x),ceil_div(bo, threads.y));
//    const int hwo = h * w * filter.size(1);
//    const int threadsy = next_pow2_clip(hwo, 1024);
//    const int threadsx = next_pow2_clip(filter.size(0), 1024 / threadsy);
//    const dim3 threads(threadsx, threadsy);
//    const dim3 blocks(ceil_div(filter.size(0), threads.x) , ceil_div(hwo, threads.y));
    cuda_binary_batch_conv2d_kernel_old<<<blocks,threads>>>(output.packed_accessor32<int32_t,4,torch::RestrictPtrTraits>(),
                                   input.packed_accessor32<long,4,torch::RestrictPtrTraits>(),
                                   filter.packed_accessor32<long,5,torch::RestrictPtrTraits>(),
                                   padx, pady, stridex, stridey);



        return output;
}

__global__ void cuda_binary_batch_conv2d_kernel_old_shared(torch::PackedTensorAccessor32<int32_t,4,torch::RestrictPtrTraits> output,
                               const torch::PackedTensorAccessor32<long,4,torch::RestrictPtrTraits> input,
                               const torch::PackedTensorAccessor32<long,5,torch::RestrictPtrTraits> filter,
                               const int output_block_length,
                               const int padx, const int pady,
                               const int stridex, const int stridey) {
        const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
        const int y_out = blockIdx.y * blockDim.y + threadIdx.y;
        const int x_block = blockIdx.x * blockDim.x;
        const int y_block = blockIdx.y * blockDim.y;
        const int x_thread = threadIdx.x;
        const int y_thread = threadIdx.y;
        const int idx_bc = blockIdx.z * blockDim.z + threadIdx.z;
        const int batch = idx_bc / (output_block_length * blockDim.z);
        const int ch_out = idx_bc - batch * (output_block_length * blockDim.z);
        const int h_cache = (blockDim.x - 1) * stridex + filter.size(2);
        const int w_cache = (blockDim.y - 1) * stridey + filter.size(3);
        const int hw_cache = h_cache * w_cache;
        extern __shared__ int shared_mem[];
        long *shared_input = (long*)&shared_mem[0];

        if (batch < output.size(0)) {
            if (threadIdx.z == 0) { // try using threads to handle different ch_in

                const int blocksize = blockDim.x * blockDim.y;
                for (int xy_thread = x_thread * blockDim.y + y_thread; xy_thread < hw_cache; xy_thread += blocksize) {
                    int x_cache = xy_thread / w_cache;
                    int y_cache = xy_thread - x_cache * w_cache;
                    int x_in = x_cache + x_block - padx;
                    int y_in = y_cache + y_block - pady;
                    bool in_bound = (x_in >= 0 && x_in < input.size(1) && y_in >= 0 && y_in < input.size(2));
                    for (int ch_in = 0; ch_in < input.size(3); ch_in++) {
                        shared_input[(x_cache * w_cache + y_cache) * input.size(3) + ch_in] = in_bound ? input[batch][x_in][y_in][ch_in] : 0;
                    }
                }
            }
            __syncthreads();
            if (x_out < output.size(1) && y_out < output.size(2) && ch_out < output.size(3)) {
                int tmp = 0;

                for (int i = 0; i < filter.size(2); i++) {
                    for (int j = 0; j < filter.size(3); j++) {
                        int x_in = i + x_thread * stridex;
                        int y_in = j + y_thread * stridey;
                        for (int ch_in = 0; ch_in < input.size(3); ch_in++) {
//                            tmp += __popcll(filter[batch][ch_out][i][j][ch_in] ^ shared_input[(x_in * w_cache + y_in) * input.size(3) + ch_in]);
                            tmp += __popcll(1L ^ shared_input[(x_in * w_cache + y_in) * input.size(3) + ch_in]);
//                            tmp += __popcll(filter[batch][ch_out][i][j][ch_in] ^ 1L);
                        }
                    }
                }
                output[batch][x_out][y_out][ch_out] = tmp;
            }
        }
}

torch::Tensor cuda_binary_batch_conv2d_old_shared(torch::Tensor input, torch::Tensor filter, int padx, int pady, int stridex, int stridey) {
    const int h = (input.size(1) - filter.size(2) + 2 * padx) / stridex + 1;
    const int w = (input.size(2) - filter.size(3) + 2 * pady) / stridey + 1;
    auto output = torch::zeros({filter.size(0), h, w, filter.size(1)}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
    const int threadsx = next_pow2_clip(h, 16);
    const int threadsy = next_pow2_clip(w, 16);
    const int threadsz = next_pow2_clip(filter.size(1), 1024 / threadsy / threadsx);
//    const int threadsx = 1;
    const dim3 threads(threadsx, threadsy, threadsz);
    const int output_block_length = ceil_div(filter.size(1), threads.z);

    const dim3 blocks(ceil_div(h, threads.x), ceil_div(w, threads.y), filter.size(0) * output_block_length);
    const int h_cache = (threads.x - 1) * stridex + filter.size(2);
    const int w_cache = (threads.y - 1) * stridey + filter.size(3);
    const int sharedMemory = h_cache * w_cache * filter.size(4) * elementSize(input.scalar_type());
//    printf("%d \n", sharedMemory);
    cuda_binary_batch_conv2d_kernel_old_shared<<<blocks,threads, sharedMemory>>>(output.packed_accessor32<int32_t,4,torch::RestrictPtrTraits>(),
                                   input.packed_accessor32<long,4,torch::RestrictPtrTraits>(),
                                   filter.packed_accessor32<long,5,torch::RestrictPtrTraits>(),
                                   output_block_length,
                                   padx, pady, stridex, stridey);



        return output;
}
