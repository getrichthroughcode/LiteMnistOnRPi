import torch 
import torch.nn as nn
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from utils.trainingscript import *
from utils.export_model import *



# Path to dataset
data_path_train = "../data/database_BMP/train"
data_path_test = "../data/database_BMP/test"


transform = transforms.Compose([
    #transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
    #transforms.Normalize((0.5,), (0.5,))
])


train_dataset = CustomMNISTDataset(data_path_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)


test_dataset = CustomMNISTDataset(data_path_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"{len(test_loader)=}")
print(f"{len(train_loader)=}")


num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
    test_loss, test_accuracy = test(model, test_loader, criterion, device)
    print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

export_model_info(model, "../models/SimpleNN")