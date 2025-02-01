# **LiteMnistOnRpi**

Ce dépôt propose un **pipeline complet** pour entraîner un modèle (de type MLP) en **Python** sur un PC ou autre machine, puis **compiler et exécuter l’inférence** en **C** (par exemple sur une Raspberry Pi). L’idée est d’avoir un code d’inférence très léger et portable, idéal pour un usage embarqué.  
Ce travail a été effectué dans un cadre académique, concernant le cours d'IA embarquée dispensé par **Mr Chatrie** à l'ENSEIRB-MATMECA.

---

## **1. Modules extérieurs utilisés**
Au-delà des librairies « classiques » (PyTorch, GCC, etc.) mentionnées plus loin, ce projet repose sur :

- **Dockerfile** (fourni par **Mr Chatrie**) : pour disposer d’un environnement de travail local (pour l'entraînement et tester l'inférence).
- **cJSON** (trouvé sur GitHub) : pour parser le JSON (où on exporte les métadonnées de l’architecture).
- **Bmp2Matrix** (écrit par **Mr Chatrie**) : permet de convertir un fichier BMP en matrice, utile pour l’inférence C sur la Raspberry Pi.

---

## **2. Structure de notre contribution dans le projet**
```plain text
Work/
├── training
│   ├── data/
│   │   ├── bmpProcessed/    <– Dossier contenant les images d’entraînement/test (format BMP)
│   │   └── …                <– Autres datasets éventuels
│   ├── models/
│   ├── src/
│   │   └── main.py          <– Script Python principal pour l’entraînement
│   └── utils/
│       ├── export_model.py  <– Fonctions pour exporter le modèle et ses poids
│       └── trainingscript.py <– Fonctions de training/test
└── inference
├── include/
│   ├── Bmp2Matrix.h
│   ├── cJSON.h
│   └── load_weights.h
├── utils/
│   ├── Bmp2Matrix.c
│   ├── cJSON.c
│   └── load_weights.c
├── src/
│   ├── main.c            <– Exécutable principal pour l’inférence
│   ├── reconstruct_architecture.c
│   └── run_inference.c
├── data/
│   └── …                 <– Exemples d’images BMP (28×28×3)
├── models/
│   └── SimpleNNbmp/      <– Dossier où sont exportés les poids (binaires) + model_info.json
└── build/
└── test              <– Binaire généré
```
---

## **3. Fonctionnalités principales**

1. **Entraînement (PC/Desktop)**
   - Fait en Python (PyTorch) avec un dataset de chiffres BMP (28×28×3).
   - Sauvegarde les **poids** au format binaire, plus les métadonnées `model_info.json` (description de l’architecture).

2. **Inference (C)**
   - Compilation C (testée sur macOS, conteneur Ubuntu, Raspberry Pi, etc.).
   - Chargement du `model_info.json` et des fichiers binaires de poids.
   - Lecture d’une image BMP (28×28×3).
   - Exécution d’un MLP (entièrement en C) et affichage du résultat (logits, softmax, label prédict).

3. **Makefile** pour automatiser :
   - `make train` : lance l’entraînement Python et exporte les poids.
   - `make compile` : compile le code C.
   - `make inference` : exécute le binaire sur une image BMP.
   - `make` : enchaîne tout (train → compile → inference).

---

## **4. Prérequis**

### Sur un PC (ou VM) pour l’entraînement

- **Python 3** (testé sur 3.8+).
- **PyTorch**, **torchvision**, et éventuellement **tqdm** pour les barres de progression.
- **argparse** si on veut personnaliser le script via la ligne de commande (généralement déjà inclus).

### Pour l’inférence en C (local ou Raspberry Pi)

- **GCC** ou un autre compilateur C (sur Raspbian : `sudo apt-get install build-essential`).
- **math.h** (standard) + `-lm` pour l’édition de liens.
- (Optionnel) **ImageMagick** pour convertir des images en BMP.

---

## **5. Installation et utilisation**

### 5.1. Entraînement (sur PC)

1. **Cloner le repo** :
   ```bash
   git clone https://github.com/getrichthroughcode/LiteMnistOnRpi.git
   cd LiteMnistOnRpi/Work
2. **Lancer le pipeline complet(train → compile → inference):**
    ```bash
    make
### 5.2 Exécution sur la Raspberry Pi 
Une fois connecté à notre session sur la Pi, placez vous dans le dossier 
```bash
cd student/Work/
make compile
make inference
```
Le binaire lira l’image inference/data/3_1.bmp et utilisera l’architecture exportée dans inference/models/SimpleNNbmp/.
Si vous souhaitez prédire un autre chiffre BMP, il faudra adapter le chemin d’entrée dans le makefile en spécifiant bien 
le nouveau chemin de la nouvelle image au format attendu. 
>Note
Nous utilisons un réseau très simple, nommé **SimpleNN**, défini ainsi (PyTorch) :
```python
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(500, 10)   # 10 classes

    def forward(self, x):
        # on a 28x28 = 784 pixels en entrée
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.ReLU(x)
        return self.fc2(x)
``` 
## **6. Contributeurs**
    - Diallo Abdoulaye (abdoulaye.diallo@bordeaux-inp.fr)
    - Krumm Lorenzo    (lorenzo.krumm@bordeaux-inp.fr)