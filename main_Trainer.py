
# Step 2: Train EfficientNet-B0

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np


data = np.load("dataset_processed_clean.npz", allow_pickle=True)
X, y, class_names = data["images"], data["labels"], data["classes"]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


class ImageDataset(Dataset):
    def __init__(self, images, labels):
       
        self.images = torch.tensor(images.transpose(0, 3, 1, 2))
        self.labels = torch.tensor(labels).long()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

train_ds = ImageDataset(X_train, y_train)
val_ds = ImageDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)


device = "cuda" if torch.cuda.is_available() else "cpu"


model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model = model.to(device)


counts = Counter(y_train)
weights = [1.0 / counts[i] for i in range(len(class_names))]
weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


scaler = torch.cuda.amp.GradScaler()


EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    train_loss, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / len(train_ds)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

   
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_correct += (outputs.argmax(1) == labels).sum().item()
    val_acc = val_correct / len(val_ds)
    print(f" → Val Acc: {val_acc:.4f}")
