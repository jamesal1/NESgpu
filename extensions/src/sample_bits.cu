#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#define BLOCK_SIZE 128
#define ELEMENT_SIZE 64
extern "C"

__global__ void sample_bits_kernel(long *ret,
                                        const __half *input,
                                        const int ret0, const int ret1, const int ret2,
                                        const int input1,
                                        const unsigned long seed) {
    const int tid = threadIdx.x;
    const int y = blockIdx.y;
    const int z = blockIdx.z;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z_input = z * ELEMENT_SIZE;
    const int end = input1 - z_input;
//    const int seq = y * ret2  + z;
//    curandState state;
//    curand_init(seed + seq, 0, 0, &state);
//     __shared__ __half input_cache[ELEMENT_SIZE];
    __shared__ float input_cache[ELEMENT_SIZE];

    if (y < ret1 && z < ret2) {
        if (tid < ELEMENT_SIZE) {
//            input_cache[tid] = i < end ? input[y * input1 + z_input + i] : __half(0);
            input_cache[tid] = tid < end ? __half2float(input[y * input1 + z_input + tid]) : 0.0;
        }
        if (x < ret0) {
            const int seq = x * ret1 * ret2 + y * ret2 + z;
            curandState state;
            curand_init(seed + seq, 0, 0, &state);
            long tmp = 0;
            #pragma unroll
            for (int i = 0; i < ELEMENT_SIZE; i++) {
                int ii = (i + tid) & (ELEMENT_SIZE - 1);
//                    tmp |=  ((long) ( __half2float(input_cache[i]) > curand_uniform(&state))) << i;
                tmp |=  ((long) ( input_cache[ii] > curand_uniform(&state))) << ii;
            }
            ret[(x * ret1 + y) * ret2 + z] = tmp;
        }
    }
}

//__global__ void sample_bits_kernel(long *ret,
//                                        const __half *input,
//                                        const int ret0, const int ret1, const int ret2,
//                                        const int input1,
//                                        const unsigned long seed) {
//    const int z = blockIdx.x * blockDim.x + threadIdx.x;
//    const int y = blockIdx.y * blockDim.y + threadIdx.y;
//    const int z_input = z * ELEMENT_SIZE;
//    const int end = input1 - z_input;
////    const int seq = y * zlen  + z;
////    curandState state;
////    curand_init(seed + seq, 0, 0, &state);
////    __half input_cache[ELEMENT_SIZE];
//    float input_cache[ELEMENT_SIZE];
//    if (y < ret1 && z < ret2) {
//        for (int i=0; i < ELEMENT_SIZE; i++) {
////            input_cache[i] = i < end ? input[y * input1 + z_input + i] : __half(0);
//            input_cache[i] = i < end ? __half2float(input[y * input1 + z_input + i]) : 0.0;
//        }
//        for (int x = 0; x < ret0; x++) {
//            const int seq = x * ret1 * ret2 + y * ret2  + z;
//            curandState state;
//            curand_init(seed + seq, 0, 0, &state);
//                long tmp = 0;
//                #pragma unroll
//                for (int i = 0; i < ELEMENT_SIZE; i++) {
////                    tmp |=  ((long) ( __half2float(input_cache[i]) > curand_uniform(&state))) << i;
//                    tmp |=  ((long) ( input_cache[i] > curand_uniform(&state))) << i;
//                }
//                ret[(x * ret1 + y) * ret2 + z] = tmp;
//        }
//    }
//}

