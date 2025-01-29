#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/load_weights.h"        
#include "../include/cJSON.h"


/**
 * main.c
 *
 * Exécute l'inférence sur un jeu de poids décrit par un fichier JSON.
 * Paramètre d'entrée : (optionnel) le chemin vers le JSON, sinon chemin par défaut.
 */
int main(int argc, char *argv[])
{
    // 1) Déterminer le chemin vers model_info.json
    const char *json_file = "../models/SimpleNN/model_info.json";
    if (argc > 1) {
        json_file = argv[1];  // si l'utilisateur fournit son propre chemin
    }

    // 2) Reconstruire l’architecture depuis le JSON
    int num_layers = 0;
    Layer *model_layers = reconstruct_architecture(json_file, &num_layers);

    if (num_layers <= 0) {
        fprintf(stderr, "No layers found in JSON. Exiting.\n");
        return 1;
    }

    printf("Nombre de couches (num_layers) : %d\n", num_layers);

    // 3) Préparer une entrée factice (28x28 = 784) : 
    //    Ici, on la remplit de 0... sauf quelques valeurs à 255 pour simuler un pixel "allumé".
    float input[784] = {0};
    // Juste un exemple: on pourrait imaginer que le coin en haut à gauche est allumé
    // ou que c'est un chiffre manuscrit normalisé.
    for (int i = 400; i < 410; i++) {
        input[i] = 255.0f;
    }
    size_t input_size = 784; // la taille de l'entrée
    
    // 4) Exécuter l’inférence
    float *output = run_inference(model_layers, num_layers, input, input_size);
    // run_inference renvoie un nouveau buffer alloué pour la sortie finale.
    // Cf. implémentation que tu as adaptée.

    // 5) Déterminer la taille de la sortie finale
    //    - Si la dernière couche est "Linear", alors la sortie a `out_features = weight_shape[0]`
    //    - Si la dernière couche est un "ReLU" pur (sans poids), alors la taille
    //      est la même que l'entrée de cette couche, etc.
    //
    //    Pour simplifier, on suppose ici que la dernière couche est un "Linear".
    //    Sinon, il faudrait une logique plus générale (ou stocker la dimension
    //    de sortie calculée dans run_inference).

    Layer last_layer = model_layers[num_layers - 1];
    int final_output_size = 0;

    if (strcmp(last_layer.class_name, "Linear") == 0) {
        // Par convention, out_features = weight_shape[0]
        final_output_size = last_layer.weight_shape[0];
    }
    else if (strcmp(last_layer.class_name, "ReLU") == 0) {
        // Admettons que la couche juste avant est un Linear => la sortie c’est la même
        // dimension que la sortie de la couche précédente. 
        // Donc on peut regarder la couche précédente :
        if (num_layers >= 2) {
            Layer prev_layer = model_layers[num_layers - 2];
            if (strcmp(prev_layer.class_name, "Linear") == 0) {
                final_output_size = prev_layer.weight_shape[0];
            }
            else {
                // logiquement, si c'est ReLU->ReLU c'est la même dimension qu'avant, 
                // et on continue à remonter tant qu'on est sur des activations pures
                // ... (à toi de gérer ce cas).
                fprintf(stderr, "Cannot deduce final output size automatically.\n");
                final_output_size = 1;
            }
        }
        else {
            // Il n'y a qu'une couche, qui est ReLU ?
            fprintf(stderr, "Cannot deduce final output size automatically.\n");
            final_output_size = 1;
        }
    }
    else {
        // si c'est un type non géré, par ex "Conv2d", tu devras avoir un code adapté
        fprintf(stderr, "WARNING: last layer is not recognized for shape. Defaulting to 1.\n");
        final_output_size = 1;
    }

    // 6) Afficher la sortie finale
    printf("Final output (size=%d):\n", final_output_size);
    for (int i = 0; i < final_output_size; i++) {
        printf("  %f\n", output[i]);
    }

    // 7) Libérer la mémoire
    free(output);
    free(model_layers);

    return 0;
}