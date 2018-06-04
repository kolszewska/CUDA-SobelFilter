#include "Image.h"

Image::Image() {
}

Image::~Image() {
}

void Image::loadImage(const char* name) {
	unsigned decoding_error = lodepng::decode(pixels, width, height, name);
	if (decoding_error) {
		std::cout << "Decoder error " << decoding_error << ": " << lodepng_error_text(decoding_error) << std::endl;
	}
    extractChannels();
}

void Image::saveImage(const char* name) {
	combineChannels();
	unsigned encoding_error = lodepng::encode(name, out_pixels, width, height);
	if (encoding_error) {
		std::cout << "Encoder error " << encoding_error << ": " << lodepng_error_text(encoding_error) << std::endl;
	}
}

void Image::extractChannels() {
	for (int i = 0; i < pixels.size(); i++) {
		if (i % 4 == 0) {
			r_channel.push_back(pixels.at(i));
		}
		else if (i % 4 == 1) {
			g_channel.push_back(pixels.at(i));
		}
		else if (i % 4 == 2) {
			b_channel.push_back(pixels.at(i));
		}
		else {
			a_channel.push_back(pixels.at(i));
		}
	}
}

void Image::combineChannels() {
	for (int i = 0; i < width * height * 4 - 12; i++) {
		if (i % 4 == 0) {
			out_pixels.push_back(out_r_channel.at(i / 4));
		}
		else if (i % 4 == 1) {
			out_pixels.push_back(out_g_channel.at(i / 4));
		}
		else if (i % 4 == 2) {
			out_pixels.push_back(out_b_channel.at(i / 4));
		}
		else {
			out_pixels.push_back(a_channel.at(i / 4));
		}
	}
	for (int j = width * height * 4 - 12; j < width * height * 4; j++) {
		out_pixels.push_back(0);
	}
}
