#include <vector>
#include <ctime>
#include <math.h>

#include "Image.h"

#pragma once

class Sobel {
 public:
    static void applySobelFilterOnCpu(Image* image);
    static void applySobelFilterOnGpu(Image* image);
    static void applySobelFilterOnGpuChunked(Image* image, int threads_per_block, int blocks_per_grid);

    static std::vector<unsigned char> getNewChannelValues(std::vector<unsigned char> channel,
            int image_width, int image_height);
    static int computeXGradient(std::vector<unsigned char> channel, int index, int image_width);
    static int computeYGradient(std::vector<unsigned char> channel, int index, int image_width);
    static int computeGradientLength(int gradient_x, int gradient_y);
    static int normalizeGradient(int gradient_value);

 private:
    static int g_x_kernel[3][3];
    static int g_y_kernel[3][3];
};

