import os
import json
import torch
import torch.nn as nn

def export_model_info(model, folder_path):
    """
    Exporte la structure et les paramètres du `model` dans un dossier `folder_path`.
    
    Retourne un dictionnaire `model_info` qui contient les métadonnées (fichier JSON).
    Les poids/biais sont exportés dans des fichiers binaires.
    """
    
    os.makedirs(folder_path, exist_ok=True)
    model_info = {
        "layers": []
    }
    
    # On va numéroter les couches pour l'export
    layer_idx = 0
    
    # model.modules() parcourt récursivement toutes les sous-couches,
    # y compris le module racine. On veut souvent ignorer le tout premier (qui est le modèle global).
    for layer in model.modules():
        if layer == model:
            continue  # ignore le module racine lui-même

        # On prépare un dictionnaire pour décrire la couche courante
        layer_dict = {
            "layer_index": layer_idx,
            "class_name": layer.__class__.__name__
        }
        
        #
        # 1) Exporter poids/biais s'il y en a
        #
        #   Dans PyTorch, un module qui a des paramètres possède souvent .weight et .bias
        #   (ex: Linear, Conv2d, etc.). Mais une activation comme nn.ReLU n’a pas de poids.
        #
        #   On peut tester: hasattr(layer, 'weight') and layer.weight is not None
        #
        
        # S'il y a un attribut `weight` :
        if hasattr(layer, 'weight') and layer.weight is not None:
            weight_file = os.path.join(folder_path, f"layer_{layer_idx}_weight.bin")
            # On sauve en binaire
            layer.weight.data.cpu().numpy().tofile(weight_file)
            layer_dict["weight_file"] = weight_file
            layer_dict["weight_shape"] = list(layer.weight.shape)
        
        # Idem pour le biais
        if hasattr(layer, 'bias') and layer.bias is not None:
            bias_file = os.path.join(folder_path, f"layer_{layer_idx}_bias.bin")
            layer.bias.data.cpu().numpy().tofile(bias_file)
            layer_dict["bias_file"] = bias_file
            layer_dict["bias_shape"] = list(layer.bias.shape)+[1]  # On ajoute une dimension pour broadcast
        
        #
        # 2) Exporter les *hyperparamètres spécifiques* à certains types de couches
        #
        #   Exemples :
        #   - nn.Conv2d a "kernel_size", "stride", "padding", etc.
        #   - nn.Linear a "in_features", "out_features"
        #
        #   On peut faire un if / elif pour détecter certains types
        #   ou plus simplement mettre "dir(layer)" et piocher dedans.
        #
        
        # Pour un Linear :
        if isinstance(layer, nn.Linear):
            layer_dict["in_features"] = layer.in_features
            layer_dict["out_features"] = layer.out_features

        # Pour un Conv2d (exemple)
        elif isinstance(layer, nn.Conv2d):
            layer_dict["in_channels"] = layer.in_channels
            layer_dict["out_channels"] = layer.out_channels
            layer_dict["kernel_size"] = layer.kernel_size
            layer_dict["stride"] = layer.stride
            layer_dict["padding"] = layer.padding
            # etc...

        # Pour un ReLU ou autre activation :
        elif isinstance(layer, nn.ReLU):
            layer_dict["activation"] = "ReLU"

        elif isinstance(layer, nn.Sigmoid):
            layer_dict["activation"] = "Sigmoid"

        elif isinstance(layer, nn.Tanh):
            layer_dict["activation"] = "Tanh"

        # etc.

        model_info["layers"].append(layer_dict)
        layer_idx += 1

    # Sauvegarde du JSON (métadonnées)
    json_path = os.path.join(folder_path, "model_info.json")
    with open(json_path, "w") as f:
        json.dump(model_info, f, indent=4)

    return model_info