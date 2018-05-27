#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void cudaComputeXGradient(int* x_gradient, unsigned char* channel, int image_width, int image_height) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index == 0) { return; }			
		x_gradient[index] =
			1 * channel[index - 1] +
			2 * channel[index] +
			1 * channel[index + 1] +
			0 * channel[index + image_width - 1] +
			0 * channel[index + image_width] +
			0 * channel[index + image_width + 1] +
			(-1) * channel[index + 2 * image_width - 1] +
			(-2) * channel[index + 2 * image_width] +
			(-1) * channel[index + 2 * image_width + 1];
		
		return;
}

__global__ void cudaComputeYGradient(int* y_gradient, unsigned char* channel, int image_width, int image_height) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index == 0) { return; }			
		y_gradient[index] = 
			1 * channel[index - 1] +
			0 * channel[index] +
			(-1) * channel[index + 1] +
			2 * channel[index + image_width - 1] +
			0 * channel[index + image_width] +
			(-2) * channel[index + image_width + 1] +
			1 * channel[index + 2 * image_width - 1] +
		        0 * channel[index + 2 * image_width] +
			(-1) * channel[index + 2 * image_width + 1];
		
		return;
}

__global__ void cudaComputeGradientLength(unsigned char *channel_values, int* x_gradient, int* y_gradient) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	channel_values[index] = sqrt(float(x_gradient[index] * x_gradient[index] + y_gradient[index] * y_gradient[index]));
	return;
}

__global__ void cudaNormalizeGradient(unsigned char* channel_values) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (channel_values[index] > 255) {
		channel_values[index] = 255;
	}
	else if (channel_values[index] < 0) {
		channel_values[index] = 0;
	}
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
	
        cudaComputeXGradient<<<blocks_per_grid, threads_per_block>>>(x_gradient, device_old_channel_values, image_width, image_height);

        cudaComputeYGradient<<<blocks_per_grid, threads_per_block>>>(y_gradient, device_old_channel_values, image_width, image_height);

	cudaComputeGradientLength<<<blocks_per_grid, threads_per_block>>>(device_new_channel_values, x_gradient, y_gradient);

        cudaNormalizeGradient<<<blocks_per_grid, threads_per_block>>>(device_new_channel_values);

	unsigned char* host_new_channel_values = new unsigned char[image_width * image_height * sizeof(unsigned char)];
	cudaMemcpy(host_new_channel_values, device_new_channel_values, sizeof(unsigned char) * image_width * image_height, cudaMemcpyDeviceToHost);
        
	return host_new_channel_values;
}


