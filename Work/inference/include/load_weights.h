#ifndef LOAD_MODEL_H
#define LOAD_MODEL_H

#include <stddef.h> // For size_t
/**
 * @brief Loads model weights from a binary file.
 *
 * This function reads an array of weights from the specified binary file
 * and stores them in the provided buffer.
 *
 * @param filename Path to the binary file containing the weights.
 * @param weights Pointer to the buffer where the weights will be stored.
 * @param num_weights The number of weights to read from the file.
 *
 * @note Ensure the `weights` buffer is pre-allocated and large enough to
 *       hold `num_weights` elements.
 */
void load_model(const char *filename, float *weights, size_t num_weights);

typedef struct {
    int  layer_index;
    char class_name[50];

    // Poids
    char weight_file[200];
    int  weight_shape[4];      // ex: [500, 784], ou [10, 500], etc.
    int  weight_shape_len;     // taille réelle du tableau weight_shape

    // Biais
    char bias_file[200];
    int  bias_shape[4];
    int  bias_shape_len;

    // Activation
    char activation[20];

} Layer;


Layer* reconstruct_architecture(const char *json_file, int *num_layers);

float* run_inference(Layer *model_layers, int num_layers, float *input, size_t input_size);

#endif // LOAD_MODEL_H