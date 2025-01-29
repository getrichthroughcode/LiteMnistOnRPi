#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/load_weights.h"


/**
 * @brief Effectue l'inférence en parcourant les couches décrites par model_layers.
 * 
 * @param model_layers  Tableau de couches.
 * @param num_layers    Nombre de couches dans le modèle.
 * @param input         Tableau d'entrées (features).
 * @param input_size    Taille de l'entrée (ex: 784 pour un MLP).
 * 
 * @return Pointeur vers le tableau alloué contenant la dernière sortie.
 *         (À toi de free() après utilisation.)
 */
float* run_inference(Layer *model_layers, int num_layers,
                     float *input, size_t input_size) 
{
    // current_input pointer = on copie (ou on pointe) l'entrée initiale
    // Attention: si on veut éviter d’écraser l’entrée, on peut faire une copie
    float *current_input = (float*)malloc(input_size * sizeof(float));
    memcpy(current_input, input, input_size * sizeof(float));
    size_t current_input_size = input_size;  // dimension = nb de neurones en entrée

    for (int i = 0; i < num_layers; i++) 
    {
        printf("=== Layer %d (index=%d, class=%s) ===\n",
               i, model_layers[i].layer_index, model_layers[i].class_name);

        // -----------------------------------------
        //  Si c’est une couche Linear :
        // -----------------------------------------
        if (strcmp(model_layers[i].class_name, "Linear") == 0) 
        {
            // 1) Lire weight_shape (ex: [out_features, in_features])
            int out_features = 0;
            int in_features = 0;
            
            if (model_layers[i].weight_shape_len >= 2) {
                out_features = model_layers[i].weight_shape[0];
                in_features  = model_layers[i].weight_shape[1];
            } else {
                fprintf(stderr, "Error: weight_shape has less than 2 dims\n");
                exit(EXIT_FAILURE);
            }

            // Vérifier la compatibilité in_features vs current_input_size
            if (in_features != current_input_size) {
                fprintf(stderr, "Dimension mismatch: layer expects %d, but got %zu\n",
                        in_features, current_input_size);
                exit(EXIT_FAILURE);
            }

            // 2) Charger les poids
            size_t num_weights = (size_t)out_features * in_features;
            float *weights = (float *)malloc(num_weights * sizeof(float));
            load_model(model_layers[i].weight_file, weights, num_weights);

            // 3) Charger le biais (optionnel)
            float *bias = NULL;
            int bias_len = 0;
            if (strlen(model_layers[i].bias_file) > 0 && model_layers[i].bias_shape_len > 0) {
                bias_len = model_layers[i].bias_shape[0];  // ex: [500]
                if (bias_len != out_features) {
                    fprintf(stderr, "Bias dimension mismatch: got %d vs out_features=%d\n",
                            bias_len, out_features);
                    exit(EXIT_FAILURE);
                }
                bias = (float*)malloc(bias_len * sizeof(float));
                load_model(model_layers[i].bias_file, bias, bias_len);
            }

            // 4) Effectuer le matmul : output = W*x + b
            float *output = (float*)malloc(out_features * sizeof(float));
            for (int out_i = 0; out_i < out_features; out_i++) 
            {
                float sum = 0.0f;
                for (int in_i = 0; in_i < in_features; in_i++) {
                    sum += current_input[in_i] * weights[out_i * in_features + in_i];
                }
                // ajout du biais
                if (bias) {
                    sum += bias[out_i];
                }
                output[out_i] = sum;
            }

            // 5) Appliquer l’activation si nécessaire
            if (strcmp(model_layers[i].activation, "ReLU") == 0) {
                for (int n = 0; n < out_features; n++) {
                    if (output[n] < 0.0f)
                        output[n] = 0.0f;
                }
            } else if (strcmp(model_layers[i].activation, "Sigmoid") == 0) {
                for (int n = 0; n < out_features; n++) {
                    output[n] = 1.0f / (1.0f + expf(-output[n]));
                }
            } else if (strcmp(model_layers[i].activation, "Tanh") == 0) {
                for (int n = 0; n < out_features; n++) {
                    // Tanh
                    output[n] = tanhf(output[n]);
                }
            }
            // etc. pour d’autres activations

            // 6) Libérer la mémoire des poids
            free(weights);
            if (bias) free(bias);

            // 7) Mettre à jour current_input
            free(current_input); // on libère l'entrée précédente
            current_input = output;
            current_input_size = out_features;
        }
        // -----------------------------------------
        //  Si c’est juste une couche ReLU "pure" (sans poids)
        //  (cas où tu as un layer séparé pour l’activation)
        // -----------------------------------------
        else if (strcmp(model_layers[i].class_name, "ReLU") == 0) 
        {
            // On applique ReLU directement sur current_input
            for (int n = 0; n < (int)current_input_size; n++) {
                if (current_input[n] < 0.0f) 
                    current_input[n] = 0.0f;
            }
            // Pas de modification de la taille
        }
        // -----------------------------------------
        //  Autres types de couches (Conv2d, etc.)
        // (à compléter si jamais on a le temps de les implémenter)
        // -----------------------------------------
        else {
            fprintf(stderr, "Unsupported layer class_name: %s\n", model_layers[i].class_name);
            // Soit tu fais un exit, soit tu gères
            exit(EXIT_FAILURE);
        }
    }

    // A la fin, current_input pointe vers la dernière sortie
    // (allouée dynamiquement). L’appelant devra free() après usage.
    return current_input;
}