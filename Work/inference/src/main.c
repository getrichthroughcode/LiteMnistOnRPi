#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>              // pour expf
#include "../include/load_weights.h"        
#include "../include/cJSON.h"
#include "../include/Bmp2Matrix.h"

/**
 * \brief Programme principal d'inférence : 
 *        usage : ./test [json_file] [bmp_file]
 *
 * \param argv[1] Chemin vers le model_info.json (défaut : "../models/SimpleNNbmp/model_info.json")
 * \param argv[2] Chemin vers le BMP à prédire (défaut : "../data/1_8.bmp")
 */
int main(int argc, char *argv[])
{
    // -----------------------------------------------------------------
    // 1) Chemins par défaut (JSON + BMP). 
    //    On peut les surcharger en arguments de ligne de commande.
    // -----------------------------------------------------------------
    const char *json_file = "../models/SimpleNNbmp/model_info.json";
    const char *bmp_file  = "../data/1_8.bmp";

    if (argc > 1) {
        json_file = argv[1];
    }
    if (argc > 2) {
        bmp_file = argv[2];
    }

    printf("JSON file : %s\n", json_file);
    printf("BMP file  : %s\n", bmp_file);

    // -----------------------------------------------------------------
    // 2) Lecture / reconstruction du modèle depuis le JSON
    // -----------------------------------------------------------------
    int num_layers = 0;
    Layer *model_layers = reconstruct_architecture(json_file, &num_layers);
    if (num_layers <= 0) {
        fprintf(stderr, "No layers found in JSON. Exiting.\n");
        return 1;
    }
    printf("Nombre de couches (num_layers) : %d\n", num_layers);

    // -----------------------------------------------------------------
    // 3) Lecture du BMP (28×28) et conversion en niveaux de gris
    // -----------------------------------------------------------------
    FILE *f_bmp = fopen(bmp_file, "rb");  // Ouvrir le BMP en binaire
    if (!f_bmp) {
        perror("Erreur ouverture BMP file");
        free(model_layers);
        return 1;
    }

    BMP monImage;
    LireBitmap(f_bmp, &monImage);  // Remplit la structure (monImage)
    fclose(f_bmp);

    // Convertir en niveaux de gris
    ConvertRGB2Gray(&monImage);

    // Vérifier la taille
    if (monImage.infoHeader.largeur != 28 || monImage.infoHeader.hauteur != 28) {
        fprintf(stderr, "Erreur: l'image BMP n'est pas 28x28 ! (trouvé %dx%d)\n",
                monImage.infoHeader.largeur, monImage.infoHeader.hauteur);
        DesallouerBMP(&monImage);
        free(model_layers);
        return 1;
    }

    // -----------------------------------------------------------------
    // 4) Construire un tableau de 784 floats à partir du BMP
    //    (Ici, pas de normalisation, ou alors /255.0f si besoin.)
    // -----------------------------------------------------------------
    float input[784];
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            // monImage.mPixelsGray[i][j] est un unsigned char (0..255)
            // Convertir en float
            // Optionnel : input[i*28 + j] = monImage.mPixelsGray[i][j] / 255.0f;
            input[i*28 + j] = (float) monImage.mPixelsGray[i][j];
        }
    }
    DesallouerBMP(&monImage); // Libérer la structure BMP

    size_t input_size = 784;

    // -----------------------------------------------------------------
    // 5) Exécuter l’inférence (logits)
    // -----------------------------------------------------------------
    float *output = run_inference(model_layers, num_layers, input, input_size);
    // run_inference() alloue dynamiquement un tableau pour la sortie finale.

    // -----------------------------------------------------------------
    // 6) Déterminer la taille de la sortie finale
    //    (Linear => weight_shape[0], etc.)
    // -----------------------------------------------------------------
    Layer last_layer = model_layers[num_layers - 1];
    int final_output_size = 0;

    if (strcmp(last_layer.class_name, "Linear") == 0) {
        final_output_size = last_layer.weight_shape[0];
    }
    else if (strcmp(last_layer.class_name, "ReLU") == 0) {
        // si la dernière couche est ReLU, on remonte à la précédente
        if (num_layers >= 2) {
            Layer prev_layer = model_layers[num_layers - 2];
            if (strcmp(prev_layer.class_name, "Linear") == 0) {
                final_output_size = prev_layer.weight_shape[0];
            } else {
                final_output_size = 1;
            }
        } else {
            final_output_size = 1;
        }
    }
    else {
        fprintf(stderr, "WARNING: last layer not recognized for shape. Default=1\n");
        final_output_size = 1;
    }

    // -----------------------------------------------------------------
    // 7) Calcul du softmax stable
    // -----------------------------------------------------------------
    //  => output[i] contiennent des logits. 
    //     on va exponentier (x_i - max_x) pour éviter l'inf ou NaN.

    // Trouver max_logit
    float max_logit = output[0];
    for (int i = 1; i < final_output_size; i++) {
        if (output[i] > max_logit) {
            max_logit = output[i];
        }
    }

    // Somme des exponentielles décalées
    float sum_exp = 0.0f;
    for (int i = 0; i < final_output_size; i++) {
        sum_exp += expf(output[i] - max_logit);
    }

    // Allocation du tableau softmax
    float *softmax_vals = (float*)malloc(sizeof(float)*final_output_size);
    if (!softmax_vals) {
        fprintf(stderr, "Allocation error for softmax.\n");
        free(output);
        free(model_layers);
        return 1;
    }

    // Calcul du softmax
    for (int i = 0; i < final_output_size; i++) {
        softmax_vals[i] = expf(output[i] - max_logit) / sum_exp;
    }

    // Trouver l'index avec la plus grande probabilité
    int predicted_label = 0;
    float max_prob = softmax_vals[0];
    for (int i = 1; i < final_output_size; i++) {
        if (softmax_vals[i] > max_prob) {
            max_prob = softmax_vals[i];
            predicted_label = i;
        }
    }

    // -----------------------------------------------------------------
    // 8) Affichage de la sortie
    // -----------------------------------------------------------------
    printf("\nLogits (size=%d):\n", final_output_size);
    for (int i = 0; i < final_output_size; i++) {
        printf("  output[%d] = %f\n", i, output[i]);
    }

    printf("\nSoftmax probabilities:\n");
    for (int i = 0; i < final_output_size; i++) {
        printf("  Class %d: %.4f\n", i, softmax_vals[i]);
    }

    printf("\nPredicted label: %d (prob=%.4f)\n", predicted_label, max_prob);

    // -----------------------------------------------------------------
    // 9) Nettoyage
    // -----------------------------------------------------------------
    free(softmax_vals);
    free(output);
    free(model_layers);

    return 0;
}