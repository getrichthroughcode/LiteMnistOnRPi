# **LiteMnistOnRpi**
Ce dépôt propose un **pipeline complet** pour entraîner un modèle (de type MLP) en **Python** sur un PC ou autre machine, puis **compiler et exécuter l’inférence** en **C** (une Raspberry Pi). L’idée est d’avoir un code d’inférence très léger et portable, idéal pour un usage embarqué.
Ce travail a été effectué dans un cadre académique, concernant le cours d'IA embarquée dispensé par **Mr Chatrie** à l'ENSEIRB-MATMECA.

## **Structure du projet**
>
Work/
├── training
│   ├── data/
│   │   ├── bmpProcessed/    <– Dossier contenant les images d’entraînement/test (format BMP)
│   │   └── …              <– Autres datasets éventuels
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
│   └── …               <– Exemples d’images BMP (28x28)
├── models/
│   └── SimpleNNbmp/      <– Dossier où sont exportés les poids (binaires) + model_info.json
├── Makefile
└── build/
└── test              <– Binaire généré 

