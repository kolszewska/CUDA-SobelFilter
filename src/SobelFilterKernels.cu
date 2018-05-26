#include <stdio.h>

#include <cuda_runtime.h>

__global__ void cudaComputeXGradient(int* x_gradient, unsigned char* channel, int index, int image_width, int image_height) {
		int grad_x;
		if (index + 2 * image_width + 1 < image_height * image_width) {
		grad_x =
			1 * channel[index - 1] +
			2 * channel[index] +
			1 * channel[index + 1] +
			0 * channel[index + image_width - 1] +
			0 * channel[index + image_width] +
			0 * channel[index + image_width + 1] +
			(-1) * channel[index + 2 * image_width - 1] +
			(-2) * channel[index + 2 * image_width] +
			(-1) * channel[index + 2 * image_width + 1];
	}
	else {
		grad_x = 0;
	}
	x_gradient = &grad_x;
}

extern "C" unsigned char* cudaGetNewChannelValues(unsigned char* channel, int image_width, int image_height) {
	unsigned char* new_channel_values;
	cudaMallocManaged(&new_channel_values, image_width * image_height);
	
        int *x_gradient;
        cudaMallocManaged(&x_gradient, sizeof(int));

	for (int x = 1; x < 5; x++) {
		cudaComputeXGradient<<<1,1>>>(x_gradient, channel, x, image_width, image_height);
		printf("%d/n",&x_gradient);
	}
	return new_channel_values;
}

