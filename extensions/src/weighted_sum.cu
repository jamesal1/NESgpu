#include <cuda_fp16.h>
#define ELEMENT_SIZE 64
#define BLOCK_SIZE 64
#define WEIGHT_MAX_LENGTH 2048
extern "C"
//use constant memory for weights if needs to be faster

__global__ void weighted_sum_kernel(__half *ret,
                                    const long *input,
                                    const __half *weights,
                                    const int ret0, const int ret1,
                                    const int input0, const int input1, const int input2
                                    ) {
    __shared__ __half weight_cache[BLOCK_SIZE];
    __shared__ long cache[BLOCK_SIZE][BLOCK_SIZE / ELEMENT_SIZE];
    const int z = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    const int tid = threadIdx.x;
    const int ratio = BLOCK_SIZE / ELEMENT_SIZE;
    const int ratio_sq = BLOCK_SIZE / ratio;
    const int x_cache = tid / ratio;
    const int z_cache = tid - x_cache * ratio;
    const int z_offset = blockIdx.x * ratio;
    const int z_element = tid / ELEMENT_SIZE;
    const int z_bit = tid - z_element * ELEMENT_SIZE;
    float tmp = 0; // half precision performance crippled on Pascal?
//    __half tmp = 0;
    for (int x_offset = 0; x_offset < input0; x_offset += BLOCK_SIZE) {
        for (int x = 0; x < ratio; x++) {
            const int x_block = x * ratio_sq + x_cache;
            cache[x_block][z_cache] = x_offset + x_block < input0 && z_cache + z_offset < input2 ?
                input[((x_offset + x_block) * input1 + y) * input2 + z_cache + z_offset] : 0;
        }
        weight_cache[tid] = weights[tid + x_offset];
        __syncthreads();
        #pragma unroll
        for (int x = 0; x < BLOCK_SIZE; x++) {
            if ((cache[x][z_element] >> z_bit) & 1) {
                tmp += (float) weight_cache[x];
//                tmp += weight_cache[x];
            }
        }
        __syncthreads();
    }
    if (z<ret1) {
        ret[y * ret1 + z] = (__half) tmp;
    }
}