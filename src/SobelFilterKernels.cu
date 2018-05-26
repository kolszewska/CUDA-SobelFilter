#include <cuda_runtime.h>

__global__ 
extern "C" unsigned char* cudaGetNewChannelValues(unsigned char* channel, int image_width, int image_height) {
	unsigned char* new_channel_values;
	cudaMallocManaged(&new_channel_values, image_width * image_height);
	
	for (int x = 1; x < image_width * image_height - 2; x++) {
		int x_gradient = cudaComputeXGradient<<1,1>>(channel, x, image_width, image_height);
	}
	return new_channel_values;
}

__global__
extern "C" int cudaComputeXGradient(unsigned char* channel, int index, int image_width, int image_height) {
		if (index + 2 * image_width + 1 < image_height * image_width)) {
		int grad_x =
			1 * channel[index - 1] +
			2 * channel[index] +
			1 * channel[index + 1] +
			0 * channel[index + image_width - 1] +
			0 * channel[index + image_width] +
			0 * channel[index + image_width + 1] +
			(-1) * channel[index + 2 * image_width - 1] +
			(-2) * channel[index + 2 * image_width] +
			(-1) * channel[index + 2 * image_width + 1];
		return grad_x;
	}
	else {
		return 0;
	}
}
