#include "Image.h"
#include "Sobel.h"

int main(int argc, char *argv[])
{
    Image *image = new Image;
    Sobel::applySobelFilterOnCpu(image);
    image->saveImage("cpu.png");
    Sobel::applySobelFilterOnGpu(image);
    image->saveImage("gpu.png");
    return 0;
}
