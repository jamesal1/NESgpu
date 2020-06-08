
extern "C"
__global__ void batch_conv2d_kernel(int *output,
//                               long *input,
                               cudaTextureObject_t input,
//                               long *filter,
                               int *filter,
                               int output0, int output1, int output2, int output3,
                               int input1, int input2, int input3,
                               int filter1, int filter2, int filter3, int filter4,
                               int padx, int pady,
                               int stridex, int stridey) {

        const int idx_bc = blockIdx.y * blockDim.y + threadIdx.y;
        const int idx_xy = blockIdx.x * blockDim.x + threadIdx.x;
        const int x_out = idx_xy / output2;
        const int y_out = idx_xy - x_out * output2;
        const int batch = idx_bc / output3;
        const int ch_out = idx_bc - batch * output3;
        if (batch < output0 && x_out < output1 && y_out < output2 && ch_out < output3) {
            int tmp = 0;
            for (int ch_in = 0; ch_in < input3; ch_in++) {
                for (int i = 0; i < filter2; i++) {
                    for (int j = 0; j < filter3; j++) {
                        int x_in = i + x_out * stridex - padx;
                        int y_in = j + y_out * stridey - pady;
                        long inp = (x_in >= 0 && x_in < input1 && y_in >= 0 && y_in < input2) ?
//                            input[((batch * input1 + x_in) * input2 + y_in) * input3 + ch_in] : 0;
                               tex3D<int>(input, ch_in, y_in, batch * input1 + x_in) : 0;
//                        long inp = 0;
                        tmp += __popcll(filter[((((batch * filter1 + ch_out) * filter2 + i) * filter3 + j) * filter4 + ch_in)] ^ inp);
                    }
                }
            }
            output[((batch * output1 + x_out) * output2 + y_out) * output3 + ch_out] = tmp;
        }

}