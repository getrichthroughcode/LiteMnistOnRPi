#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/load_weights.h"        
#include "../include/cJSON.h"
#include "../include/Bmp2Matrix.h"

int main(int argc, char *argv[])
{
    // -----------------------------------------------------------------
    // 1) Chemins par défaut (JSON + BMP). L’utilisateur peut les modifier via argv.
    // -----------------------------------------------------------------
    const char *json_file = "../models/SimpleNN/model_info.json";
    const char *bmp_file  = "../data/1_8.bmp";

    // Si l'utilisateur fournit des arguments :
    //   argv[1] = chemin vers model_info.json
    //   argv[2] = chemin vers le .bmp
    if (argc > 1) {
        json_file = argv[1];
    }
    if (argc > 2) {
        bmp_file = argv[2];
    }

    printf("JSON file : %s\n", json_file);
    printf("BMP file  : %s\n", bmp_file);

    // -----------------------------------------------------------------
    // 2) Lecture / reconstruction du modèle
    // -----------------------------------------------------------------
    int num_layers = 0;
    Layer *model_layers = reconstruct_architecture(json_file, &num_layers);
    if (num_layers <= 0) {
        fprintf(stderr, "No layers found in JSON. Exiting.\n");
        return 1;
    }
    printf("Nombre de couches (num_layers) : %d\n", num_layers);

    // -----------------------------------------------------------------
    // 3) Lecture du BMP 28×28 et conversion en niveaux de gris
    // -----------------------------------------------------------------
    FILE *f_bmp = fopen(bmp_file, "rb");   // Ouvre le BMP en binaire
    if (!f_bmp) {
        perror("Erreur ouverture BMP file");
        return 1;
    }

    BMP monImage;
    LireBitmap(f_bmp, &monImage);          // Remplit la structure (monImage)
    fclose(f_bmp);

    // Convertir en niveaux de gris (remplit monImage.mPixelsGray)
    ConvertRGB2Gray(&monImage);

    // Vérifier la taille. 
    // On suppose ici que c'est 28×28, 24 bits, etc.
    if (monImage.infoHeader.largeur != 28 || monImage.infoHeader.hauteur != 28) {
        fprintf(stderr, "Erreur: l'image BMP n'est pas 28x28 ! (trouvé %dx%d)\n",
                monImage.infoHeader.largeur, monImage.infoHeader.hauteur);
        DesallouerBMP(&monImage);
        return 1;
    }

    // -----------------------------------------------------------------
    // 4) Construire un tableau de 784 floats à partir du BMP
    // -----------------------------------------------------------------
    float input[784];
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            // monImage.mPixelsGray[i][j] est un unsigned char (0..255)
            // On le convertit en float
            // Optionnel : normaliser entre 0 et 1 => input[i*28+j] = monImage.mPixelsGray[i][j] / 255.0f
            input[i*28 + j] = (float) monImage.mPixelsGray[i][j];
        }
    }

    // Libérer la mémoire BMP (on a copié les pixels dans 'input')
    DesallouerBMP(&monImage);

    size_t input_size = 784;

    // -----------------------------------------------------------------
    // 5) Exécuter l’inférence
    // -----------------------------------------------------------------
    float *output = run_inference(model_layers, num_layers, input, input_size);

    // -----------------------------------------------------------------
    // 6) Déterminer la taille de la sortie finale 
    //    (cf. la logique existante : Linear => weight_shape[0], etc.)
    // -----------------------------------------------------------------
    Layer last_layer = model_layers[num_layers - 1];
    int final_output_size = 0;

    if (strcmp(last_layer.class_name, "Linear") == 0) {
        final_output_size = last_layer.weight_shape[0];
    }
    else if (strcmp(last_layer.class_name, "ReLU") == 0) {
        if (num_layers >= 2) {
            Layer prev_layer = model_layers[num_layers - 2];
            if (strcmp(prev_layer.class_name, "Linear") == 0) {
                final_output_size = prev_layer.weight_shape[0];
            } else {
                // etc., logique d'auto-déduction
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
    // 7) Afficher la sortie finale
    // -----------------------------------------------------------------
    printf("Final output (size=%d):\n", final_output_size);
    for (int i = 0; i < final_output_size; i++) {
        printf("  %f\n", output[i]);
    }

    // -----------------------------------------------------------------
    // 8) Libérer la mémoire
    // -----------------------------------------------------------------
    free(output);
    free(model_layers);

    return 0;
}