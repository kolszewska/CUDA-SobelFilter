#include "Image.h"
#include "Sobel.h"

int main(int argc, char *argv[]) {
    Image *image_to_be_computed_on_cpu = new Image;
    Image *image_to_be_computed_on_gpu = new Image;

    image_to_be_computed_on_cpu->loadImage("lenna.png");
    image_to_be_computed_on_gpu->loadImage("dog.png");

    Sobel::applySobelFilterOnCpu(image_to_be_computed_on_cpu);
    image_to_be_computed_on_cpu->saveImage("cpu.png");

    Sobel::applySobelFilterOnGpu(image_to_be_computed_on_gpu);
    image_to_be_computed_on_gpu->saveImage("gpu.png");

    return 0;
}

