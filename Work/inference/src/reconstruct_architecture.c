#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/load_weights.h"
#include "../include/cJSON.h"

Layer* reconstruct_architecture(const char *json_file, int *num_layers) {
    // Charger le fichier JSON
    FILE *file = fopen(json_file, "r");
    printf("fichier bien ouvert\n");
    if (!file) {
        perror("Failed to open JSON file, wrong path?");
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    printf("longueur du fichier : %ld\n", length);
    fseek(file, 0, SEEK_SET);

    char *data = (char *)malloc(length + 1);
    fread(data, 1, length, file);
    fclose(file);
    data[length] = '\0';

    // Parser le fichier JSON
    printf("début du parsing\n");
    cJSON *json = cJSON_Parse(data);
    if (!json) {
        printf("Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        free(data);
        exit(EXIT_FAILURE);
    }

    printf("Stockage des couches\n");

    cJSON *layers = cJSON_GetObjectItem(json, "layers");
    if (!layers) {
        printf("JSON does not contain 'layers' key\n");
        free(data);
        cJSON_Delete(json);
        exit(EXIT_FAILURE);
    }
    
    *num_layers = cJSON_GetArraySize(layers);
    printf("nombre de couches : %d\n", *num_layers);
    Layer *model_layers = (Layer *)malloc((*num_layers) * sizeof(Layer));
    printf("Récupératon des informations des couches\n");
    for (int i = 0; i < *num_layers; i++) {
        printf("couche %d\n", i);
        cJSON *layer = cJSON_GetArrayItem(layers, i);
        printf("Information globale récupérée\n");

        // Extraire les informations de la couche
        printf("Extraction du nom et du type de la couche\n");
        strcpy(model_layers[i].name, cJSON_GetObjectItem(layer, "name")->valuestring);
        strcpy(model_layers[i].type, cJSON_GetObjectItem(layer, "type")->valuestring);

        printf("extraction de la forme de la couche\n");
        cJSON *shape = cJSON_GetObjectItem(layer, "shape");
        model_layers[i].shape[0] = cJSON_GetArrayItem(shape, 0)->valueint;
        model_layers[i].shape[1] = cJSON_GetArrayItem(shape, 1)->valueint;
        printf("extraction du nom du fichier\n");
        strcpy(model_layers[i].filename, cJSON_GetObjectItem(layer, "filename")->valuestring);
        printf("Check fonction d'activation ou pas\n");
        cJSON *activation = cJSON_GetObjectItem(layer, "activation");
        if (activation) {
            printf("extraction de la fonction d'activation\n");
            strcpy(model_layers[i].activation, activation->valuestring);
        } else {
            printf("pas de fonction d'activation\n");
            strcpy(model_layers[i].activation, "None");
        }
        printf("fin de la couche\n");
    }

    // Nettoyage
    free(data);
    cJSON_Delete(json);

    return model_layers;
}