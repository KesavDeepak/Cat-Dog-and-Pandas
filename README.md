# Cats-Dogs-Pandas-Image-Classification-using-Transfer-Learning

## Overview
This project implements an **image classification pipeline** using **Transfer Learning** with a pre-trained **ResNet18 model** in PyTorch. The model classifies images into three categories: **cats, dogs, and pandas**. The approach leverages pre-trained weights from ImageNet, freezing early layers and training a new fully connected layer for custom classification.

## Code
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# -----------------------------
# 1. Data Preparation
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder("data/train", transform=transform)
test_data = datasets.ImageFolder("data/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# -----------------------------
# 2. Load Pretrained Model
# -----------------------------
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# -----------------------------
# 3. Training
# -----------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f\"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}\")

torch.save(model.state_dict(), \"best_model.pth\")

# -----------------------------
# 4. Evaluation
# -----------------------------
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=train_data.classes)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# -----------------------------
# 5. Single Image Prediction
# -----------------------------
from PIL import Image

def predict_image(path, model, transform):
    image = Image.open(path)
    image = transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return train_data.classes[pred.item()]

print(\"Prediction:\", predict_image(\"sample_panda.jpg\", model, transform))
