extern "C"
__global__ void texture_batch_im2col_kernel(int *output,
                                    cudaTextureObject_t input,
                                    const int output0, const int output1, const int output2,
                                    const int input1, const int input2, const int input3,
                                                   const int filterx, const int filtery,
                                                   const int padx, const int pady,
                                                   const int stridex, const int stridey) {
        const int batch = blockIdx.z * blockDim.z + threadIdx.z;
        const int loc = blockIdx.y * blockDim.y + threadIdx.y;
        const int element = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch < output0 && loc < output1 && element < output2) {
            const int w = (input2 - filtery + 2 * pady) / stridey + 1;
            const int x_out = loc / w;
            const int y_out = loc - x_out * w;
            const int element_idx = element / input3;
            const int channel = element - element_idx * input3;
            const int i = element_idx / filtery;
            const int j = element_idx - i * filtery;
            const int x_in = i + x_out * stridex - padx;
            const int y_in = j + y_out * stridey - pady;
            output[(batch * output1 + loc) * output2 + element] = (x_in >= 0 && x_in < input1 && y_in >= 0 && y_in < input2) ?
                tex1Dfetch<int>(input,((batch * input1 + x_in) * input2 + y_in) * input3 + channel) : 0;
//                tex3D<int>(input,batch * input1 + x_in, y_in, channel) : 0;
//                tex3D<int>(input, channel, y_in, batch * input1 + x_in) : 0;
        }
}