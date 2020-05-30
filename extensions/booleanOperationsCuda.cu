#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>



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

//order of channels for input/output is N H W C and order of filter is N O H W C
__global__ void cuda_binary_batch_conv2d_kernel(torch::PackedTensorAccessor32<int32_t,4,torch::RestrictPtrTraits> output,
                               const torch::PackedTensorAccessor32<long,4,torch::RestrictPtrTraits> input,
                               const torch::PackedTensorAccessor32<long,5,torch::RestrictPtrTraits> filter,
                               const int output_block_length,
                               const int padx, const int pady,
                               const int stridex, const int stridey) {
        const int x_out = blockIdx.y * blockDim.y + threadIdx.y;
        const int y_out = blockIdx.z * blockDim.z + threadIdx.z;
        const int x_block = blockIdx.y * blockDim.y;
        const int y_block = blockIdx.z * blockDim.z;
        const int x_thread = threadIdx.y;
        const int y_thread = threadIdx.z;
        const int idx_bc = blockIdx.x * blockDim.x + threadIdx.x;
        const int batch = idx_bc / (output_block_length * blockDim.x);
        const int ch_out = idx_bc - batch * (output_block_length * blockDim.x);
        const int h_cache = (blockDim.y - 1) * stridex + filter.size(2);
        const int w_cache = (blockDim.z - 1) * stridey + filter.size(3);
        const int hw_cache = h_cache * w_cache;
        extern __shared__ int shared_mem[];
        long *shared_input = (long*)&shared_mem[0];

        if (batch < output.size(0)) {
//            if (threadIdx.x == 0) { // try using threads to handle different ch_in
//
//                const int blocksize = blockDim.y * blockDim.z;
//                for (int xy_thread = x_thread * blockDim.z + y_thread; xy_thread < hw_cache; xy_thread += blocksize) {
//                    int x_cache = xy_thread / w_cache;
//                    int y_cache = xy_thread - x_cache * w_cache;
//                    int x_in = x_cache + x_block - padx;
//                    int y_in = y_cache + y_block - pady;
//                    bool in_bound = (x_in >= 0 && x_in < input.size(1) && y_in >= 0 && y_in < input.size(2));
//                    for (int ch_in = 0; ch_in < input.size(3); ch_in++) {
//                        shared_input[(x_cache * w_cache + y_cache) * input.size(3) + ch_in] = in_bound ? input[batch][x_in][y_in][ch_in] : 0;
//                    }
//                }
//            }
//            __syncthreads();
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

__global__ void cuda_batch_im2col_kernel(torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> output,
                               const torch::PackedTensorAccessor32<long,4,torch::RestrictPtrTraits> input,
                               const int output_block_length,
                               const int padx, const int pady,
                               const int stridex, const int stridey) {
        const int batch = blockDim.x;
        const int x_out = blockIdx.y * blockDim.y + threadIdx.y;
        const int y_out = blockIdx.z * blockDim.z + threadIdx.z;
        const int x_block = blockIdx.y * blockDim.y;
        const int y_block = blockIdx.z * blockDim.z;
        const int x_thread = threadIdx.y;
        const int y_thread = threadIdx.z;
        const int h_cache = (blockDim.y - 1) * stridex + filter.size(2);
        const int w_cache = (blockDim.z - 1) * stridey + filter.size(3);
        const int hw_cache = h_cache * w_cache;
        const int loc = x_out *  + y_out;
        extern __shared__ int shared_mem[];
        long *shared_input = (long*)&shared_mem[0];

        if (threadIdx.x == 0) { // try using threads to handle different ch_in

            const int blocksize = blockDim.y * blockDim.z;
            for (int xy_thread = x_thread * blockDim.z + y_thread; xy_thread < hw_cache; xy_thread += blocksize) {
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
            int e = 0;
            for (int i = 0; i < filter.size(2); i++) {
                for (int j = 0; j < filter.size(3); j++) {
                    int x_in = i + x_thread * stridex;
                    int y_in = j + y_thread * stridey;
                    for (int ch_in = 0; ch_in < input.size(3); ch_in++) {
                        output[batch][loc][e++] = shared_input[(x_in * w_cache + y_in) * input.size(3) + ch_in];
                    }
                }
            }
        }
}

//order of channels for input/output is N H W C and order of filter is N O H W C
//stride 1
__global__ void cuda_binary_batch_3x3_kernel(torch::PackedTensorAccessor32<int32_t,4,torch::RestrictPtrTraits> output,
                               const torch::PackedTensorAccessor32<long,4,torch::RestrictPtrTraits> input,
                               const torch::PackedTensorAccessor32<long,5,torch::RestrictPtrTraits> filter,
                               const int output_block_length,
                               const int padx, const int pady) {
    const int x_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int y_out = blockIdx.z * blockDim.z + threadIdx.z;
    const int x_block = blockIdx.y * blockDim.y;
    const int y_block = blockIdx.z * blockDim.z;
    const int x_thread = threadIdx.y;
    const int y_thread = threadIdx.z;
    const int idx_bc = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = idx_bc / (output_block_length * blockDim.x);
    const int ch_out = idx_bc - batch * (output_block_length * blockDim.x);
    const int h_cache = blockDim.y + 2;
    const int w_cache = blockDim.z + 2;
    const int hw_cache = h_cache * w_cache;
    const int in_channels = input.size(3);
    extern __shared__ int shared_mem[];
    long *shared_input = (long*)&shared_mem[0];
    if (batch < output.size(0)) {
        if (threadIdx.x == 0) { // try using threads to handle different ch_in
            const int blocksize = blockDim.y * blockDim.z;
            for (int xy_thread = x_thread * blockDim.z + y_thread; xy_thread < hw_cache; xy_thread += blocksize) {
                int x_cache = xy_thread / w_cache;
                int y_cache = xy_thread - x_cache * w_cache;
                int x_in = x_cache + x_block - padx;
                int y_in = y_cache + y_block - pady;
                bool in_bound = (x_in >= 0 && x_in < input.size(1) && y_in >= 0 && y_in < input.size(2));
                for (int ch_in = 0; ch_in < in_channels; ch_in++) {
                    shared_input[(x_cache * w_cache + y_cache) * in_channels + ch_in] = in_bound ? input[batch][x_in][y_in][ch_in] : 0;
                }
            }
        }
        __syncthreads();
        if (x_out < output.size(1) && y_out < output.size(2) && ch_out < output.size(3)) {

            int tmp = 0;

            for (int ch_in = 0; ch_in < in_channels; ch_in++) {
                int start = ((x_thread) * w_cache + y_thread) * in_channels + ch_in;
                tmp += __popcll(filter[batch][ch_out][0][0][ch_in] ^ shared_input[start])
                + __popcll(filter[batch][ch_out][0][1][ch_in] ^ shared_input[start + in_channels])
                + __popcll(filter[batch][ch_out][0][2][ch_in] ^ shared_input[start + 2 * in_channels])
                + __popcll(filter[batch][ch_out][1][0][ch_in] ^ shared_input[start + w_cache * in_channels])
                + __popcll(filter[batch][ch_out][1][1][ch_in] ^ shared_input[start + (w_cache + 1) * in_channels])
                + __popcll(filter[batch][ch_out][1][2][ch_in] ^ shared_input[start + (w_cache + 2) * in_channels])
                + __popcll(filter[batch][ch_out][2][0][ch_in] ^ shared_input[start + 2 * w_cache * in_channels])
                + __popcll(filter[batch][ch_out][2][1][ch_in] ^ shared_input[start + (2 * w_cache + 1) * in_channels])
                + __popcll(filter[batch][ch_out][2][2][ch_in] ^ shared_input[start + (2 * w_cache + 2) * in_channels]);
            }
            output[batch][x_out][y_out][ch_out] = tmp;

        }
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



// thread config assumes B.size(2) = 1
torch::Tensor cuda_binary_bmm(torch::Tensor A, torch::Tensor B) {
    auto C = torch::zeros({A.size(0), A.size(1), B.size(2)}, torch::TensorOptions().dtype(torch::kInt32).device(A.device()));
    const int threadsy = next_pow2_clip(A.size(1), 1024);
    const int threadsx = next_pow2_clip(A.size(0), 1024 / threadsy);
    const dim3 threads(threadsx, threadsy);
    const dim3 blocks(ceil_div(C.size(0), threads.x) , ceil_div(C.size(1), threads.y), ceil_div(C.size(2), threads.z));
    AT_DISPATCH_INTEGRAL_TYPES(A.scalar_type(), "binary_bmm_cuda", ([&] {
            cuda_binary_bmm_kernel<<<blocks,threads>>>(
                        C.packed_accessor32<int32_t,3,torch::RestrictPtrTraits>(),
                        A.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                        B.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>()
                        );

        }));
    return C;
}

//thread config assumes square input.
torch::Tensor cuda_binary_batch_conv2d(torch::Tensor input, torch::Tensor filter, int padx, int pady, int stridex, int stridey) {
    const int h = (input.size(1) - filter.size(2) + 2 * padx) / stridex + 1;
    const int w = (input.size(2) - filter.size(3) + 2 * pady) / stridey + 1;
    auto output = torch::zeros({filter.size(0), h, w, filter.size(1)}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
    const int threadsy = next_pow2_clip(h, 16);
    const int threadsz = next_pow2_clip(w, 16);
    const int threadsx = next_pow2_clip(filter.size(1), 1024 / threadsy / threadsz);
//    const int threadsx = 1;
    const dim3 threads(threadsx, threadsy, threadsz);
    const int output_block_length = ceil_div(filter.size(1), threads.x);

    const dim3 blocks(filter.size(0) * output_block_length , ceil_div(h, threads.y), ceil_div(w, threads.z));
    const int h_cache = (threads.y - 1) * stridex + filter.size(2);
    const int w_cache = (threads.z - 1) * stridey + filter.size(3);
    const int sharedMemory = h_cache * w_cache * filter.size(4) * elementSize(input.scalar_type());
//    printf("%d \n", sharedMemory);
    if (filter.size(2) == 3 && filter.size(3 == 3) && stridex == 1 && stridey == 1) {
        printf("using 3x3\n");

        cuda_binary_batch_3x3_kernel<<<blocks,threads, sharedMemory>>>(output.packed_accessor32<int32_t,4,torch::RestrictPtrTraits>(),
                                       input.packed_accessor32<long,4,torch::RestrictPtrTraits>(),
                                       filter.packed_accessor32<long,5,torch::RestrictPtrTraits>(),
                                       output_block_length,
                                       padx, pady);

    } else {
                cuda_binary_batch_conv2d_kernel<<<blocks,threads, sharedMemory>>>(output.packed_accessor32<int32_t,4,torch::RestrictPtrTraits>(),
                                               input.packed_accessor32<long,4,torch::RestrictPtrTraits>(),
                                               filter.packed_accessor32<long,5,torch::RestrictPtrTraits>(),
                                               output_block_length,
                                               padx, pady, stridex, stridey);



    }
        return output;
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



//order of channels for input/output is N H W C and order of filter is N O H W C
__global__ void cuda_binary_batch_conv2d_kernel_old(torch::PackedTensorAccessor32<int32_t,4,torch::RestrictPtrTraits> output,
                               torch::PackedTensorAccessor32<long,4,torch::RestrictPtrTraits> input,
                               torch::PackedTensorAccessor32<long,5,torch::RestrictPtrTraits> filter,
                               int padx, int pady,
                               int stridex, int stridey) {
        const int idx_xy = blockIdx.y * blockDim.y + threadIdx.y;
        const int idx_bc = blockIdx.x * blockDim.x + threadIdx.x;
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
    const int threadsy = next_pow2_clip(hw, 1024);
    const int threadsx = next_pow2_clip(bo, 1024 / threadsy);
    const dim3 threads(threadsx, threadsy);
    const dim3 blocks(ceil_div(bo, threads.x) , ceil_div(hw, threads.y));
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