#define REPEAT X
extern "C"
__global__ void max_pool2d_kernel(long *output,
                                    const long *input,
                                    const int output0, const int output1, const int output2,
                                    const int input1, const int input2, const int input3,
                                                   const int filterx, const int filtery,
                                                   const int padx, const int pady,
                                                   const int stridex, const int stridey) {

        const int loc = blockIdx.y * blockDim.y + threadIdx.y;
        const int element = blockIdx.x * blockDim.x + threadIdx.x;
        if (loc < output1 & element < output2) {
            const int w = (input2 - filtery + 2 * pady) / stridey + 1;
            const int x_out = loc / w;
            const int y_out = loc - x_out * w;
            int batch = blockIdx.z * REPEAT * blockDim.z + threadIdx.z;
            for (int r=0;r<REPEAT & batch < output0; r++) {
                long tmp = 0;
                #pragma unroll
                for (int i=0; i < 2; i++) {
                #pragma unroll
                    for (int j=0; j < 2; j++) {
                        const int x_in = i + x_out * stridex - padx;
                        const int y_in = j + y_out * stridey - pady;
                        const bool in_bound = (x_in >= 0 & x_in < input1 & y_in >= 0 & y_in < input2);
                        tmp |= in_bound ? input[((batch * input1 + x_in) * input2 + y_in) * input3 + element] : 0;
                    }
                }


                output[(batch * output1 + loc) * output2 + element] = tmp;
                batch += blockDim.z;
            }
        }
}

//__global__ void batch_im2col_kernel(long *output,
//                                    const long *input,
//                                    const int output0, const int output1, const int output2,
//                                    const int input1, const int input2, const int input3,
//                                                   const int filterx, const int filtery,
//                                                   const int padx, const int pady,
//                                                   const int stridex, const int stridey) {
//        const int batch = blockIdx.z * blockDim.z + threadIdx.z;
////        const int loc = blockIdx.y * blockDim.y + threadIdx.y;
//        const int element = blockIdx.x * blockDim.x + threadIdx.x;
//        const int w = (input2 - filtery + 2 * pady) / stridey + 1;
////        const int x_out = loc / w;
////        const int y_out = loc - x_out * w;
//        const int element_idx = element / input3;
//        const int channel = element - element_idx * input3;
//        const int i = element_idx / filtery;
//        const int j = element_idx - i * filtery;
//        if (batch < output0) {
//            for (int r=0;r<REPEAT; r++) {
//    //            const int batch = (blockIdx.z * REPEAT + r) * blockDim.z + threadIdx.z;
//                const int loc = (blockIdx.y * REPEAT + r) * blockDim.y + threadIdx.y;
//                const int x_out = loc / w;
//                const int y_out = loc - x_out * w;
//                const int x_in = i + x_out * stridex - padx;
//                const int y_in = j + y_out * stridey - pady;
//                if (loc < output1 && element < output2) {
//
//                    output[(batch * output1 + loc) * output2 + element] = (x_in >= 0 && x_in < input1 && y_in >= 0 && y_in < input2) ?
//                        input[((batch * input1 + x_in) * input2 + y_in) * input3 + channel] : 0;
//                }
//            }
//        }
//}


