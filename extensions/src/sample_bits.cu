#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#define BLOCK_SIZE X
#define ELEMENT_SIZE 64
extern "C"


__global__ void sample_bits_kernel(long *ret,
                                        const __half *input,
                                        const int ret0, const int ret1, const int ret2,
                                        const int input1,
                                        const unsigned long seed) {
    const int z = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    const int z_offset = blockIdx.x * blockDim.x * ELEMENT_SIZE;
    const int end = input1 - z_offset;
    const int ratio = BLOCK_SIZE / ELEMENT_SIZE;
    const int tid = threadIdx.x;
    const int z_cache = tid / ELEMENT_SIZE;
    const int e_cache = tid - z_cache * ELEMENT_SIZE;

    const int seq = y * ret2  + z;
    curandState state;
    curand_init(seed + seq, 0, 0, &state);
    __half input_cache[ELEMENT_SIZE];
    //    float input_cache[ELEMENT_SIZE];
    __shared__ __half input_shared[BLOCK_SIZE][ELEMENT_SIZE + 1];
//    __shared__ float input_shared[BLOCK_SIZE][ELEMENT_SIZE + 1];

    if (y < ret1) {
        #pragma unroll
        for (int i=0; i < ELEMENT_SIZE; i++) {
            int idx = i * BLOCK_SIZE + tid;
                input_shared[i * ratio + z_cache][e_cache] = idx < end ? input[y * input1 + z_offset + idx] : __half(0);
//                input_shared[i * ratio + z_cache][e_cache] = idx < end ? __half2float(input[y * input1 + z_offset + idx]) : 0.0;
            }
    }
    __syncthreads();
    if (y < ret1 && z < ret2) {

        #pragma unroll
        for (int i=0; i < ELEMENT_SIZE; i++) {
//            input_cache[i] = __half2float(input_shared[tid][i]);
            input_cache[i] = input_shared[tid][i];
        }
        for (int x = 0; x < ret0; x++) {
//            const int seq = x * ret1 * ret2 + y * ret2  + z;
//            curandState state;
//            curand_init(seed + seq, 0, 0, &state);
                long tmp = 0;
                #pragma unroll
                for (int i = 0; i < ELEMENT_SIZE; i++) {
                    tmp |=  ((long) ( __half2float(input_cache[i]) > curand_uniform(&state))) << i;
//                    tmp |=  ((long) ( input_cache[i] > curand_uniform(&state))) << i;
                }
                ret[(x * ret1 + y) * ret2 + z] = tmp;
        }
    }
}