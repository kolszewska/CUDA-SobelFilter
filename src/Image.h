#include <iostream>
#include <fstream>
#include <vector>

#include "lodepng.h"

#pragma once

class Image {
public:
	Image();
	~Image();

	std::vector<unsigned char> pixels;
	std::vector<unsigned char> out_pixels;

	std::vector<unsigned char> r_channel;
	std::vector<unsigned char> g_channel;
	std::vector<unsigned char> b_channel;
	std::vector<unsigned char> a_channel;

	std::vector<unsigned char> out_r_channel;
	std::vector<unsigned char> out_g_channel;
	std::vector<unsigned char> out_b_channel;

	unsigned width, height;

	void loadImage();
	void saveImage();
	void extractChannels();
	void combineChannels();
private:
	const char* inputFilename = "in.png";
	const char* outputFilename = "out.png";
};