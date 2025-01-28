#include <stdio.h>
#include <stdlib.h>
#include "../include/load_weights.h"


/* Function to load model weights from a .bin file
   The file should contain the weights in binary format
   The weights array should be pre-allocated with the correct size
   The number of weights should be equal to the size of the weights array
*/
void load_model(const char *filename, float *weights, size_t num_weights) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file for reading");
        exit(EXIT_FAILURE);
    }

    // Read weights from the file
    size_t read = fread(weights, sizeof(float), num_weights, file);
    if (read != num_weights) {
        perror("Failed to read all weights from file");
        exit(EXIT_FAILURE);
    }

    fclose(file);
    printf("Model loaded successfully from %s\n", filename);
}