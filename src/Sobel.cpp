#include "Sobel.h"

extern "C" unsigned char* cudaGetNewChannelValues(unsigned char* channel, int image_width, int image_height);

int Sobel::g_x_kernel[3][3] = { { 1, 0, -1 },{ 2, 0, -2 },{ 1, 0, -1 } };
int Sobel::g_y_kernel[3][3] = { { 1, 2, 1 },{ 0, 0, 0 },{ -1, -2, -1 } };

void Sobel::applySobelFilterOnCpu(Image* image) {
	clock_t begin = clock();
	image->out_r_channel = getNewChannelValues(image->r_channel, image->width, image->height);
	image->out_g_channel = getNewChannelValues(image->g_channel, image->width, image->height);
	image->out_b_channel = getNewChannelValues(image->b_channel, image->width, image->height);
	clock_t end = clock();
	double execution_time = double(end - begin) / CLOCKS_PER_SEC;
	printf("\n[CPU] Elapsed time: %f", execution_time);
}

std::vector<unsigned char> Sobel::getNewChannelValues(std::vector<unsigned char> channel, int image_width, int image_height) {
	std::vector<unsigned char> new_channel_values;
	for (int x = 1; x < image_width * image_height - 2; x++) {
		int x_gradient = computeXGradient(channel, x, image_width);
		int y_gradient = computeYGradient(channel, x, image_width);
		new_channel_values.push_back(normalizeGradient(computeGradientLength(x_gradient, y_gradient)));
	}
	return new_channel_values;
}

int Sobel::computeXGradient(std::vector<unsigned char> channel, int index, int image_width) {
	if (index + 2 * image_width + 1 < channel.size()) {
		int grad_x =
			g_x_kernel[0][0] * channel.at(index - 1) +
			g_x_kernel[1][0] * channel.at(index) +
			g_x_kernel[2][0] * channel.at(index + 1) +
			g_x_kernel[0][1] * channel.at(index + image_width - 1) +
			g_x_kernel[1][1] * channel.at(index + image_width) +
			g_x_kernel[2][1] * channel.at(index + image_width + 1) +
			g_x_kernel[0][2] * channel.at(index + 2 * image_width - 1) +
			g_x_kernel[1][2] * channel.at(index + 2 * image_width) +
			g_x_kernel[2][2] * channel.at(index + 2 * image_width + 1);
		return grad_x;
	}
	else {
		return 0;
	}
}

int Sobel::computeYGradient(std::vector<unsigned char> channel, int index, int image_width) {
	if (index + 2 * image_width + 1 < channel.size()) {
		int grad_y =
			g_y_kernel[0][0] * channel.at(index - 1) +
			g_y_kernel[1][0] * channel.at(index) +
			g_y_kernel[2][0] * channel.at(index + 1) +
			g_y_kernel[0][1] * channel.at(index + image_width - 1) +
			g_y_kernel[1][1] * channel.at(index + image_width) +
			g_y_kernel[2][1] * channel.at(index + image_width + 1) +
			g_y_kernel[0][2] * channel.at(index + 2 * image_width - 1) +
			g_y_kernel[1][2] * channel.at(index + 2 * image_width) +
			g_y_kernel[2][2] * channel.at(index + 2 * image_width + 1);
		return grad_y;
	}
	else {
		return 0;
	}
}

int Sobel::computeGradientLength(int gradient_x, int gradient_y) {
	return sqrt(gradient_x * gradient_x + gradient_y * gradient_y);
}

int Sobel::normalizeGradient(int gradient_value) {
	if (gradient_value > 255) {
		return 255;
	}
	else if (gradient_value < 0) {
		return 0;
	}
	else {
		return gradient_value;
	}
}

void Sobel::applySobelFilterOnGpu(Image* image) {
	 clock_t begin = clock();

	 unsigned char* out_r_channel = new unsigned char[image->r_channel.size()];
	 unsigned char* out_g_channel = new unsigned char[image->g_channel.size()];
	 unsigned char* out_b_channel = new unsigned char[image->b_channel.size()];

	 out_r_channel = cudaGetNewChannelValues(reinterpret_cast<unsigned char*> (image->r_channel.data()), image->width, image->height);

	//image->out_r_channel = getNewChannelValues(image->r_channel, image->width, image->height);
	//image->out_g_channel = getNewChannelValues(image->g_channel, image->width, image->height);
	//image->out_b_channel = getNewChannelValues(image->b_channel, image->width, image->height);
	clock_t end = clock();
	double execution_time = double(end - begin) / CLOCKS_PER_SEC;
	printf("\n[GPU] Elapsed time: %f", execution_time);
}
