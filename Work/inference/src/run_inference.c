#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/load_weights.h"

float* run_inference(Layer *model_layers, int num_layers, float *input, size_t input_size) {
    float *current_input = input;

    for (int i = 0; i < num_layers; i++) {
        printf("Processing layer %d: %s\n", i + 1, model_layers[i].name);

        // Charger les poids
        size_t num_weights = model_layers[i].shape[0] * model_layers[i].shape[1];
        float *weights = (float *)malloc(num_weights * sizeof(float));
        load_model(model_layers[i].filename, weights, num_weights);

        // Calculer la sortie de la couche
        float *output = (float *)malloc(model_layers[i].shape[0] * sizeof(float));
        for (int j = 0; j < model_layers[i].shape[0]; j++) {
            output[j] = 0;
            for (int k = 0; k < model_layers[i].shape[1]; k++) {
                output[j] += current_input[k] * weights[j * model_layers[i].shape[1] + k];
                printf("poids quelconque %f\n", weights[j * model_layers[i].shape[1] + k]);
            }

            // Appliquer la fonction d'activation
            if (strcmp(model_layers[i].activation, "ReLU") == 0) {
                output[j] = output[j] > 0 ? output[j] : 0;
            } //else if (strcmp(model_layers[i].activation, "Sigmoid") == 0) {
                //output[j] = 1 / (1 + exp(-output[j]));
            //} 
        }

        free(weights);
        current_input = output;  // Mettre à jour l'entrée pour la couche suivante
    }

    return current_input;  // Retourne la dernière sortie
}