#include <cuda_runtime.h>

__global__ void cudaComputeXGradient(int* x_gradient, unsigned char* channel, int image_width, int image_height, int chunk_size_per_thread) {
    int x_kernel[3][3] = { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } }; 
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = index * chunk_size_per_thread; i < (index + 1) * chunk_size_per_thread; i++) {
        if (i == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
            continue;
        } else {
            x_gradient[i] =
                x_kernel[0][0] * channel[i - 1] +
                x_kernel[1][0] * channel[i] +
                x_kernel[2][0] * channel[i + 1] +
                x_kernel[0][1] * channel[i + image_width - 1] +
                x_kernel[1][1] * channel[i + image_width] +
                x_kernel[2][1] * channel[i + image_width + 1] +
                x_kernel[0][2] * channel[i + 2 * image_width - 1] +
                x_kernel[1][2] * channel[i + 2 * image_width] +
                x_kernel[2][2] * channel[i + 2 * image_width + 1];
        }
    }
    return;
}

__global__ void cudaComputeYGradient(int* y_gradient, unsigned char* channel, int image_width, int image_height, int chunk_size_per_thread) {
    int y_kernel[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } }; 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
   
    for (int i = index * chunk_size_per_thread; i < (index + 1) * chunk_size_per_thread; i++) {
        if (i == 0 && blockIdx.x == 0 && blockIdx.x == 0) {
            continue;
        } else {
            y_gradient[i] =
                y_kernel[0][0] * channel[i - 1] +
                y_kernel[1][0] * channel[i] +
                y_kernel[2][0] * channel[i + 1] +
                y_kernel[0][1] * channel[i + image_width - 1] +
                y_kernel[1][1] * channel[i + image_width] +
                y_kernel[2][1] * channel[i + image_width + 1] +
                y_kernel[0][2] * channel[i + 2 * image_width - 1] +
                y_kernel[1][2] * channel[i + 2 * image_width] +
                y_kernel[2][2] * channel[i + 2 * image_width + 1];
        }
    }
    return;
}

__global__ void cudaComputeAndNormalizeGradientLength(unsigned char *channel_values, int* x_gradient, int* y_gradient, int chunk_size_per_thread) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index * chunk_size_per_thread; i < (index + 1) * chunk_size_per_thread; i++) {
        int gradient_length = int(sqrt(float(x_gradient[i] * x_gradient[i] + y_gradient[i] * y_gradient[i])));
        if (gradient_length > 255) {
            gradient_length = 255;
        }
        channel_values[i] = gradient_length;
    }
    return;
}

extern "C" unsigned char* cudaGetNewChannelValuesChunked(unsigned char* channel, int image_width, int image_height, int threads_per_block, int blocks_per_grid) {
    unsigned char* device_new_channel_values;
    unsigned char* device_old_channel_values;
    int* x_gradient;
    int* y_gradient;

    cudaMalloc(&device_old_channel_values, image_width * image_height * sizeof(unsigned char));
    cudaMemcpy(device_old_channel_values, channel, sizeof(unsigned char) * image_width * image_height, cudaMemcpyHostToDevice);
	
    cudaMalloc(&device_new_channel_values, image_width * image_height * sizeof(unsigned char));
    cudaMalloc(&x_gradient, (image_width * image_height) * sizeof(int));
    cudaMalloc(&y_gradient, (image_width * image_height) * sizeof(int));

    int chunk_size_per_block = (image_width * image_height) / blocks_per_grid;
    int chunk_size_per_thread = chunk_size_per_block / threads_per_block;

    cudaComputeXGradient<<<blocks_per_grid, threads_per_block>>>(x_gradient, device_old_channel_values, image_width, image_height, chunk_size_per_thread);

    cudaComputeYGradient<<<blocks_per_grid, threads_per_block>>>(y_gradient, device_old_channel_values, image_width, image_height, chunk_size_per_thread);

    cudaComputeAndNormalizeGradientLength<<<blocks_per_grid, threads_per_block>>>(device_new_channel_values, x_gradient, y_gradient, chunk_size_per_thread);

    unsigned char* host_new_channel_values = new unsigned char[image_width * image_height * sizeof(unsigned char)];
	
    cudaMemcpy(host_new_channel_values, device_new_channel_values, sizeof(unsigned char) * image_width * image_height,
        cudaMemcpyDeviceToHost);

    cudaFree(&device_new_channel_values);
    cudaFree(&device_old_channel_values);
    cudaFree(&x_gradient);
    cudaFree(&y_gradient);
    delete[] host_new_channel_values;

    return host_new_channel_values;
}