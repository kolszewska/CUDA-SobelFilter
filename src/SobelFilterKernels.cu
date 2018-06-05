#include <cuda_runtime.h>

__global__ void cudaComputeXGradient(int* x_gradient, unsigned char* channel, int image_width, int image_height) {
    int x_kernel[3][3] = { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } }; 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0) {
	    return;
    }
    x_gradient[index] =
        x_kernel[0][0] * channel[index - 1] +
        x_kernel[1][0] * channel[index] +
        x_kernel[2][0] * channel[index + 1] +
        x_kernel[0][1] * channel[index + image_width - 1] +
        x_kernel[1][1] * channel[index + image_width] +
        x_kernel[2][1] * channel[index + image_width + 1] +
        x_kernel[0][2] * channel[index + 2 * image_width - 1] +
        x_kernel[1][2] * channel[index + 2 * image_width] +
        x_kernel[2][2] * channel[index + 2 * image_width + 1];
    return;
}

__global__ void cudaComputeYGradient(int* y_gradient, unsigned char* channel, int image_width, int image_height) {
    int y_kernel[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } }; 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0) {
	    return;
    }
    y_gradient[index] =
        y_kernel[0][0] * channel[index - 1] +
        y_kernel[1][0] * channel[index] +
        y_kernel[2][0] * channel[index + 1] +
        y_kernelb[0][1] * channel[index + image_width - 1] +
        y_kernel[1][1] * channel[index + image_width] +
        y_kernel[2][1] * channel[index + image_width + 1] +
        y_kernel[0][2] * channel[index + 2 * image_width - 1] +
        y_kernel[1][2] * channel[index + 2 * image_width] +
        y_kernel[2][2] * channel[index + 2 * image_width + 1];
    return;
}

__global__ void cudaComputeAndNormalizeGradientLength(unsigned char *channel_values, int* x_gradient, int* y_gradient) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int gradient_length = int(sqrt(float(x_gradient[index] * x_gradient[index] + y_gradient[index] * y_gradient[index])));
    if (gradient_length > 255) {
        gradient_length = 255;
    }
	channel_values[index] = gradient_length;
    return;
}

extern "C" unsigned char* cudaGetNewChannelValues(unsigned char* channel, int image_width, int image_height) {
    unsigned char* device_new_channel_values;
    unsigned char* device_old_channel_values;
    int* x_gradient;
    int* y_gradient;

    cudaMalloc(&device_old_channel_values, image_width * image_height * sizeof(unsigned char));
    cudaMemcpy(device_old_channel_values, channel, sizeof(unsigned char) * image_width * image_height, cudaMemcpyHostToDevice);
	
    cudaMalloc(&device_new_channel_values, image_width * image_height * sizeof(unsigned char));
    cudaMalloc(&x_gradient, (image_width * image_height) * sizeof(int));
    cudaMalloc(&y_gradient, (image_width * image_height) * sizeof(int));

    int threads_per_block = 16;
    int blocks_per_grid = (image_width * image_height)/threads_per_block;	
	
    cudaComputeXGradient<<<blocks_per_grid, threads_per_block>>>(x_gradient, device_old_channel_values,
            image_width, image_height);

    cudaComputeYGradient<<<blocks_per_grid, threads_per_block>>>(y_gradient, device_old_channel_values,
            image_width, image_height);

    cudaComputeAndNormalizeGradientLength<<<blocks_per_grid, threads_per_block>>>(device_new_channel_values, x_gradient, y_gradient);

    unsigned char* host_new_channel_values = new unsigned char[image_width * image_height * sizeof(unsigned char)];
	
    cudaMemcpy(host_new_channel_values, device_new_channel_values, sizeof(unsigned char) * image_width * image_height,
        cudaMemcpyDeviceToHost);
        
    return host_new_channel_values;
}

