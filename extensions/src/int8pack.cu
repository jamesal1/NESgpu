#define ELEMENT_SIZE 64
#define BLOCK_SIZE 16
extern "C"
__global__ void int8pack_kernel(long *ret,  const unsigned char *input, const int ret0, const int ret1, const int input1) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int cache1 = ELEMENT_SIZE / 8 * blockDim.x;
    const int square_size = BLOCK_SIZE * BLOCK_SIZE;
    const int offset_y = blockIdx.x * blockDim.x * ELEMENT_SIZE / 8;
    const int offset = blockIdx.y * blockDim.y * input1 + offset_y;
    __shared__ unsigned char cache[BLOCK_SIZE * BLOCK_SIZE * ELEMENT_SIZE / 8];
    for (int i = 0; i < ELEMENT_SIZE / 8; i++){
        const int idx = i * square_size + tid;
        const int x_sub = idx / cache1;
        const int y_sub = idx - x_sub * cache1;
        cache[x_sub * cache1 + y_sub] = y_sub + offset_y < input1 ? input[x_sub * input1 + y_sub + offset] : 0;

    }
    __syncthreads();
    const int y_input = threadIdx.x * ELEMENT_SIZE / 8;
    if (x < ret0 && y < ret1) {
        long tmp = 0;
        #pragma unroll
        for (int i = 0; i < ELEMENT_SIZE / 8; i++) {
            tmp |= ((long) cache[threadIdx.y * cache1 + y_input + i] ) << (8 * i);
        }
        ret[x * ret1 + y] = tmp;
    }
}