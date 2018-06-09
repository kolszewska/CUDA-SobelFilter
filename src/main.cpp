#include "Image.h"
#include "Sobel.h"

int main(int argc, char *argv[]) {

    const char* input_file_path;
    const char* output_file_path;
    int threads_per_block;
    int blocks_per_grid;

    if (argc == 5) {
        input_file_path = argv[1];
        output_file_path = argv[2];
        threads_per_block = atoi(argv[3]);
        blocks_per_grid = atoi(argv[4]);
    } else {
        printf("\nRunning Sobel filter with default configuration.\n");
        input_file_path = "../assets/dog.png";
        output_file_path = "out.png";
        threads_per_block = 16;
        blocks_per_grid = 256;
    }

    printf("==================================\n");
    printf("    Sobel Filter configuration    \n");
    printf("==================================\n");
    
    printf("Path to input file in PNG format: %s\n", input_file_path);
    printf("Path to the output file: %s\n", output_file_path);
    printf("Number of threads per block: %d\n", threads_per_block);
    printf("Number of blocks per grid: %d\n", blocks_per_grid);

    printf("==================================\n");
    printf("             Results:             \n");
    printf("==================================\n");

    Image *image = new Image;

    image->loadImage(input_file_path);

    Sobel::applySobelFilterOnGpu(image, threads_per_block, blocks_per_grid);
    image->saveImage(output_file_path);

    return 0;
}

