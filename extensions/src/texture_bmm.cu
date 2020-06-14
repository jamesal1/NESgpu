#define BLOCK_SIZE 8
extern "C"
__global__ void texture_bmm_kernel(int *C,
                           cudaTextureObject_t A,
                           cudaTextureObject_t B,
                           int C1, int C2,
                           int A1, int A2,
                           int B1, int B2) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int x_sub = threadIdx.y;
    const int y_sub = threadIdx.x;
    __shared__ long Asub[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ long Bsub[BLOCK_SIZE][BLOCK_SIZE + 1];
    int tmp = 0;
    for (int inner_block = 0; inner_block < (A2 + BLOCK_SIZE - 1) / BLOCK_SIZE; inner_block++) {
        const int inner = inner_block * BLOCK_SIZE;
//        Asub[y_sub][z_sub] = z_sub + inner < A.size(2) ? A[x][y][z_sub + inner] : 0;
//        Bsub[y_sub][z_sub] = y_sub + inner < A.size(2) ? B[x][y_sub + inner][z] : 0;
        Asub[x_sub][y_sub] = tex1Dfetch<int>(A,(z * A1 + x) * A2 + y_sub + inner);
        Bsub[x_sub][y_sub] = tex1Dfetch<int>(B,(z * B1 + x_sub + inner) * B2 + y);
//        Asub[x_sub][y_sub] = tex3D<int>(A, z, x, y_sub + inner);
//        Bsub[x_sub][y_sub] = tex3D<int>(B, z, x_sub + inner, y);
        __syncthreads();
        for (int j = 0; j < BLOCK_SIZE; j++) {
            tmp += __popcll(Asub[j][x_sub] ^ Bsub[j][y_sub]);
        }
        __syncthreads();
    }


    if (x < C1 && y < C2) {
            C[(z * C1 + x) * C2 + y] = tmp;
    }
}