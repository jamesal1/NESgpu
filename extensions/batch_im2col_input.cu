#define REPEAT 1
extern "C"
__global__ void batch_im2col_input_kernel(long *output,
                                    const long *input,
                                    const int output0, const int output1, const int output2,
                                    const int input1, const int input2, const int input3,
                                    const int filterx, const int filtery,
                                    const int padx, const int pady,
                                    const int stridex, const int stridey) {

        const int idx_xy = blockIdx.y * blockDim.y  + threadIdx.y;
        const int channel = blockIdx.x * blockDim.x + threadIdx.x;
//        for (int r=0;r<REPEAT; r++) {
//            const int batch = (blockIdx.z * REPEAT + r) * blockDim.z + threadIdx.z;
            const int batch = blockIdx.z * blockDim.z + threadIdx.z;
            const int x_in = idx_xy / input2;
            const int y_in = idx_xy - x_in * input2;
            if (batch < output0 && x_in < input1 && y_in < input2 && channel < input3) {
                const int w = (input2 - filtery + 2 * pady) / stridey + 1;
                const long tmp = input[((batch * input1 + x_in) * input2 + y_in) * input3 + channel];

                int x_out = (x_in + padx) / stridex;
                const int y_init = (y_in + pady) / stridey;
                const int j_init = (y_in + pady) - y_init * stridey;
                for (int i=(x_in + padx) % stridex; i < filterx && x_out >= 0; i+=stridex) {
                    int y_out = y_init;
                    for (int j=j_init; j < filtery && y_out >= 0; j+=stridey) {

                        const int loc = x_out * w + y_out;
                        if (loc < output1 && y_out < w) {
                        const int element = (i * filtery + j) * input3 + channel;
                        output[(batch * output1 + loc) * output2 + element] = tmp;
                        }

                        y_out--;
                    }
                    x_out--;
                }

            }
//        }
}

