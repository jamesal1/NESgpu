#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#define ELEMENT_SIZE 64
extern "C"

__global__ void cuda_sample_bits_kernel(long *ret,
                                        const __half *input,
                                        const int ret0, const int ret1, const int ret2,
                                        const int input1,
                                        const unsigned long seed) {
    const int z = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.z * blockDim.z + threadIdx.z;
    const int zlen = blockDim.x * gridDim.x;
    const int ylen = blockDim.y * gridDim.y;
    const int seq = x * ylen * zlen + y * zlen  + z;
    curandState state;
    curand_init(seed + seq, 0, 0, &state);
    const int z_input = z * ELEMENT_SIZE;
    if (x < ret0 && y < ret1 && z < ret2) {
        int end = input1 - z_input;
        long tmp = 0;
        for (int i = 0; i < ELEMENT_SIZE; i++) {
            tmp |= i < end ? ( __half2float(input[y * input1 + z_input + i]) > curand_uniform(&state)) << i : 0;
        }
        ret[(x * ret1 + y) * ret2 + z] = tmp;
    }
}