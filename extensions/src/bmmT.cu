#define BLOCK_SIZE 16
#define MULT_A 4
#define MULT_B 4
extern "C"
__global__ void bmmT_kernel(int *C,
                           long *A,
                           long *B,
                           const int C1, const int C2,
                           const int A1, const int A2,
                           const int B1, const int B2) {
    const int y = blockIdx.x * MULT_B * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * MULT_A * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int x_sub = threadIdx.y;
    const int y_sub = threadIdx.x;
    __shared__ long Asub[MULT_A][BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ long Bsub[MULT_B][BLOCK_SIZE][BLOCK_SIZE];
    int tmp[MULT_A * MULT_B] = {0};
    for (int inner_block = 0; inner_block < (A2 + BLOCK_SIZE - 1) / BLOCK_SIZE; inner_block++) {
        const int inner = inner_block * BLOCK_SIZE;
        #pragma unroll
        for (int i = 0; i < MULT_A; i++) {
//            Asub[i][y_sub][x_sub] = y_sub + inner < A2 ? A[(z * A1 + x + i * BLOCK_SIZE) * A2 + y_sub + inner] : 0;
            Asub[i][x_sub][y_sub] = y_sub + inner < A2 ? A[(z * A1 + x + i * BLOCK_SIZE) * A2 + y_sub + inner] : 0;
        }
        #pragma unroll
        for (int i = 0; i < MULT_B; i++) {
            Bsub[i][x_sub][y_sub] = x_sub + inner < B2 ? B[(z * B1 + y + i * BLOCK_SIZE) * B2 + x_sub + inner] : 0;
        }
        __syncthreads();


        #pragma unroll
        for (int a = 0; a < MULT_A; a++) {
        #pragma unroll
        for (int b = 0; b < MULT_B; b++) {
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; j++) {
//            const int j1 = (x_sub + y_sub + j) & 15;
//                    tmp[a * MULT_B + b] += __popcll(Asub[a][x_sub][j1] ^ Bsub[b][j1][y_sub]);
                    tmp[a * MULT_B + b] += __popcll(Asub[a][x_sub][j] ^ Bsub[b][j][y_sub]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int a = 0; a < MULT_A; a++) {
        if (x + a * BLOCK_SIZE < C1) {
            #pragma unroll
            for (int b = 0; b < MULT_B; b++) {
                  if (y + b * BLOCK_SIZE< C2) {
                    C[(z * C1 + x + a * BLOCK_SIZE) * C2 + y + b * BLOCK_SIZE] = tmp[a * MULT_B + b];
                 }

            }
        }
    }
}
//__global__ void bmm_kernel(int *C,
//                           long *A,
//                           long *B,
//                           int C1, int C2,
//                           int A1, int A2,
//                           int B1, int B2) {
//    const int y = blockIdx.x * blockDim.x + threadIdx.x;
//    const int x = blockIdx.y * blockDim.y + threadIdx.y;
//    const int z = blockIdx.z * blockDim.z + threadIdx.z;
//    const int x_sub = threadIdx.y;
//    const int y_sub = threadIdx.x;
//    __shared__ long Asub[BLOCK_SIZE][BLOCK_SIZE + 1];
//    __shared__ long Bsub[BLOCK_SIZE][BLOCK_SIZE + 1];
//    int tmp = 0;
//    for (int inner_block = 0; inner_block < (A2 + BLOCK_SIZE - 1) / BLOCK_SIZE; inner_block++) {
//        const int inner = inner_block * BLOCK_SIZE;
////        Asub[y_sub][z_sub] = z_sub + inner < A.size(2) ? A[x][y][z_sub + inner] : 0;
////        Bsub[y_sub][z_sub] = y_sub + inner < A.size(2) ? B[x][y_sub + inner][z] : 0;
//        Asub[x_sub][y_sub] = y_sub + inner < A2 ? A[(z * A1 + x) * A2 + y_sub + inner] : 0;
//        Bsub[x_sub][y_sub] = x_sub + inner < A2 ? B[(z * B1 + x_sub + inner) * B2 + y] : 0;
//        __syncthreads();
//        for (int j = 0; j < BLOCK_SIZE; j++) {
//            tmp += __popcll(Asub[x_sub][j] ^ Bsub[j][y_sub]);
//        }
//        __syncthreads();
//    }
//
//
//    if (x < C1 && y < C2) {
//            C[(z * C1 + x) * C2 + y] = tmp;
//    }
//}