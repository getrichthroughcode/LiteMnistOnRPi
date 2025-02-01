# Fichier : Work/Makefile
# But : orchestrer l'entraînement (Python) + compilation + exécution de l'inférence C

# Variables pour la partie C
CC = gcc
CFLAGS = -I./inference/include -lm  # par exemple
SRCS = \
  inference/src/main.c \
  inference/src/reconstruct_architecture.c \
  inference/src/run_inference.c \
  inference/utils/load_weights.c \
  inference/utils/cJSON.c \
  inference/utils/Bmp2Matrix.c
OBJS = $(SRCS:.c=.o)
TARGET = inference/build/test

# Variables pour la partie Python
TRAIN_SCRIPT = training/src/main.py  # ou autre script de training
PYTHON = python3  # commande Python

.PHONY: all train compile inference clean

all: train compile inference
	@echo "Pipeline complet : entraînement -> compilation -> inférence terminé."

#--------------------------------------------------------------------------
# 1) La cible "train" lance le script Python pour l'entraînement
#--------------------------------------------------------------------------
train:
	@echo "===== Entraînement du modèle en Python ====="
	$(PYTHON) $(TRAIN_SCRIPT) \
	  --train_data "training/data/bmpProcessed/train" \
	  --test_data  "training/data/bmpProcessed/test" \
	  --export_dir "inference/models/SimpleNNbmp" \
	  --epochs 5 \
	  --batch_size 10 \
	  --lr 0.001
	@echo "===== Fin de l'entraînement ====="

#--------------------------------------------------------------------------
# 2) La cible "compile" build le code C
#--------------------------------------------------------------------------
compile: $(TARGET)
	@echo "===== Compilation terminée. Binaire : $(TARGET) ====="

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

# Compilation de chaque .c en .o
%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

#--------------------------------------------------------------------------
# 3) La cible "inference" exécute le binaire C
#--------------------------------------------------------------------------
inference: $(TARGET)
	@echo "===== Exécution du binaire d'inférence ====="
	$(TARGET) "inference/models/SimpleNNbmp/model_info.json" "inference/data/3_1.bmp"
	@echo "===== Fin de l'inférence ====="

#--------------------------------------------------------------------------
# 4) Nettoyage
#--------------------------------------------------------------------------
clean:
	rm -f $(OBJS)
	rm -f $(TARGET)
	@echo "Fichiers objets et binaire supprimés."