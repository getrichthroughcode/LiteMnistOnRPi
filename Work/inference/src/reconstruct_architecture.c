#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/load_weights.h"
#include "../include/cJSON.h"

Layer* reconstruct_architecture(const char *json_file, int *num_layers) {
    // Ouvrir le fichier JSON
    FILE *file = fopen(json_file, "r");
    if (!file) {
        perror("Failed to open JSON file");
        exit(EXIT_FAILURE);
    }

    // Lire tout le contenu du fichier en mémoire
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *data = (char *)malloc(length + 1);
    if (!data) {
        fprintf(stderr, "Memory allocation error\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fread(data, 1, length, file);
    fclose(file);
    data[length] = '\0';

    // Parser le JSON
    cJSON *json = cJSON_Parse(data);
    if (!json) {
        printf("Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        free(data);
        exit(EXIT_FAILURE);
    }

    // Récupérer le tableau "layers"
    cJSON *layers = cJSON_GetObjectItem(json, "layers");
    if (!layers) {
        printf("JSON does not contain 'layers' key\n");
        free(data);
        cJSON_Delete(json);
        exit(EXIT_FAILURE);
    }

    // Combien de couches ?
    *num_layers = cJSON_GetArraySize(layers);

    // Allouer le tableau de Layer
    Layer *model_layers = (Layer *)malloc((*num_layers) * sizeof(Layer));
    if (!model_layers) {
        fprintf(stderr, "Memory allocation error for model_layers\n");
        free(data);
        cJSON_Delete(json);
        exit(EXIT_FAILURE);
    }

    // Parcourir chaque objet-couche
    for (int i = 0; i < *num_layers; i++) {

        cJSON *layer_obj = cJSON_GetArrayItem(layers, i);
        if (!layer_obj) {
            fprintf(stderr, "Invalid layer index %d in JSON\n", i);
            free(data);
            cJSON_Delete(json);
            free(model_layers);
            exit(EXIT_FAILURE);
        }

        // 1) layer_index
        cJSON *item = cJSON_GetObjectItem(layer_obj, "layer_index");
        if (cJSON_IsNumber(item)) {
            model_layers[i].layer_index = item->valueint;
        } else {
            // S'il n'y est pas, on met -1 ou un autre défaut
            model_layers[i].layer_index = -1;
        }

        // 2) class_name
        item = cJSON_GetObjectItem(layer_obj, "class_name");
        if (cJSON_IsString(item)) {
            strcpy(model_layers[i].class_name, item->valuestring);
        } else {
            strcpy(model_layers[i].class_name, "Unknown");
        }

        // 3) weight_file
        item = cJSON_GetObjectItem(layer_obj, "weight_file");
        if (cJSON_IsString(item)) {
            strcpy(model_layers[i].weight_file, item->valuestring);
        } else {
            // Vide si absent
            model_layers[i].weight_file[0] = '\0';
        }

        // 4) weight_shape
        model_layers[i].weight_shape_len = 0;
        cJSON *weight_shape_arr = cJSON_GetObjectItem(layer_obj, "weight_shape");
        if (cJSON_IsArray(weight_shape_arr)) {
            int arr_size = cJSON_GetArraySize(weight_shape_arr);
            model_layers[i].weight_shape_len = arr_size > 4 ? 4 : arr_size; // Pour éviter de déborder
            for (int k = 0; k < model_layers[i].weight_shape_len; k++) {
                cJSON *val = cJSON_GetArrayItem(weight_shape_arr, k);
                model_layers[i].weight_shape[k] = val->valueint;
            }
        }

        // 5) bias_file
        item = cJSON_GetObjectItem(layer_obj, "bias_file");
        if (cJSON_IsString(item)) {
            strcpy(model_layers[i].bias_file, item->valuestring);
        } else {
            model_layers[i].bias_file[0] = '\0';
        }

        // 6) bias_shape
        model_layers[i].bias_shape_len = 0;
        cJSON *bias_shape_arr = cJSON_GetObjectItem(layer_obj, "bias_shape");
        if (cJSON_IsArray(bias_shape_arr)) {
            int arr_size = cJSON_GetArraySize(bias_shape_arr);
            model_layers[i].bias_shape_len = arr_size > 4 ? 4 : arr_size;
            for (int k = 0; k < model_layers[i].bias_shape_len; k++) {
                cJSON *val = cJSON_GetArrayItem(bias_shape_arr, k);
                model_layers[i].bias_shape[k] = val->valueint;
            }
        }

        // 7) activation
        item = cJSON_GetObjectItem(layer_obj, "activation");
        if (cJSON_IsString(item)) {
            strcpy(model_layers[i].activation, item->valuestring);
        } else {
            strcpy(model_layers[i].activation, "None");
        }
    }

    // Libérer data, et supprimer l'objet JSON
    free(data);
    cJSON_Delete(json);

    return model_layers;
}