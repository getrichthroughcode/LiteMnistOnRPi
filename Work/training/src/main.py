import torch 
import torch.nn as nn
import argparse
import json
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.trainingscript import *
from utils.export_model import *
from torch.utils.data import DataLoader
from torchvision import transforms


def main():
    # ---------------------------------------------------
    # 1) Parsing des arguments
    # ---------------------------------------------------
    parser = argparse.ArgumentParser(description="Training and exporting a SimpleNN model.")

    parser.add_argument(
        "--train_data", 
        type=str, 
        default="../data/bmpProcessed/train",
        help="Path to the training data folder (default: ../data/bmpProcessed/train)"
    )
    parser.add_argument(
        "--test_data", 
        type=str, 
        default="../data/bmpProcessed/test",
        help="Path to the test data folder (default: ../data/bmpProcessed/test)"
    )
    parser.add_argument(
        "--export_dir", 
        type=str, 
        default="../../inference/models/SimpleNNbmp",
        help="Path to export the model weights and metadata (default: ../../inference/models/SimpleNNbmp)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=10,
        help="Batch size for training (default: 10)"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001,
        help="Learning rate (default: 0.001)"
    )

    args = parser.parse_args()

    # ---------------------------------------------------
    # 2) Configuration de base
    # ---------------------------------------------------
    print("========== SCRIPT CONFIG ==========")
    print(f"Training data path : {args.train_data}")
    print(f"Test data path     : {args.test_data}")
    print(f"Export directory   : {args.export_dir}")
    print(f"Epochs             : {args.epochs}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Learning rate      : {args.lr}")
    print("===================================")

    # Transforms éventuels
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # ---------------------------------------------------
    # 3) Chargement des datasets
    # ---------------------------------------------------
    print("Loading dataset...")
    train_dataset = CustomMNISTDataset(args.train_data, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = CustomMNISTDataset(args.test_data, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    print(f"  -> {len(train_loader)} training batches")
    print(f"  -> {len(test_loader)} testing batches")

    # ---------------------------------------------------
    # 4) Configuration du modèle, optimizer, etc.
    # ---------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------
    # 5) Boucle d'entraînement
    # ---------------------------------------------------
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Accuracy:  {test_accuracy:.2f}%")

    # ---------------------------------------------------
    # 6) Exportation du modèle
    # ---------------------------------------------------
    print(f"Saving model to : {args.export_dir}")
    export_model_info(model, args.export_dir)
    print("Done.")

if __name__ == "__main__":
    main()