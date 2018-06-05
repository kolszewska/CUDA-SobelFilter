#include <cuda_runtime.h>
#include <stdio.h>

__global__ void cudaComputeXGradient(int* x_gradient, unsigned char* channel, int image_width, int image_height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
 if(index ==0) {
	return;
}
int a[3][3] = { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } }; 
x_gradient[index] =
        a[0][0] * channel[index - 1] +
        a[1][0] * channel[index] +
        a[2][0] * channel[index + 1] +
        a[0][1] * channel[index + image_width - 1] +
        a[1][1] * channel[index + image_width] +
        a[2][1] * channel[index + image_width + 1] +
        a[0][2] * channel[index + 2 * image_width - 1] +
        a[1][2] * channel[index + 2 * image_width] +
        a[2][2] * channel[index + 2 * image_width + 1];
    return;
}

__global__ void cudaComputeYGradient(int* y_gradient, unsigned char* channel, int image_width, int image_height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index ==0) {
	return;
}
int b[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } }; 
y_gradient[index] =
        b[0][0] * channel[index - 1] +
        b[1][0] * channel[index] +
        b[2][0] * channel[index + 1] +
        b[0][1] * channel[index + image_width - 1] +
        b[1][1] * channel[index + image_width] +
        b[2][1] * channel[index + image_width + 1] +
        b[0][2] * channel[index + 2 * image_width - 1] +
        b[1][2] * channel[index + 2 * image_width] +
        b[2][2] * channel[index + 2 * image_width + 1];
    return;
}

__global__ void cudaComputeGradientLength(unsigned char *channel_values, int* x_gradient, int* y_gradient) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int dupa = int(sqrt(float(x_gradient[index] * x_gradient[index] + y_gradient[index] * y_gradient[index])));
    if (dupa > 255) {
	dupa = 255;
    }
	channel_values[index] = dupa;
    return;
}

__global__ void cudaNormalizeGradient(unsigned char* channel_values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel_values[index] > 255) {
        channel_values[index] = int(255);
    } 
    channel_values[index] = int(channel_values[index]);
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

    cudaComputeGradientLength<<<blocks_per_grid, threads_per_block>>>(device_new_channel_values, x_gradient, y_gradient);

//    cudaNormalizeGradient<<<blocks_peir_grid, threads_per_block>>>(device_new_channel_values);

    unsigned char* host_new_channel_values = new unsigned char[image_width * image_height * sizeof(unsigned char)];
	
    cudaMemcpy(host_new_channel_values, device_new_channel_values, sizeof(unsigned char) * image_width * image_height,
        cudaMemcpyDeviceToHost);
        
    return host_new_channel_values;
}

