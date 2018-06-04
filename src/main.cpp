#include "Image.h"
#include "Sobel.h"

int main(int argc, char *argv[]) {
    Image *image = new Image;
    image->loadImage("in.png");
    Sobel::applySobelFilterOnCpu(image);
    image->saveImage("cpu.png");
    Sobel::applySobelFilterOnGpu(image);
    image->saveImage("gpu.png");
    return 0;
}

