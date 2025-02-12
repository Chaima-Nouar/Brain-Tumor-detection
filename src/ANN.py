import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define a custom Dataset class to load images from disk
class ImageDataset(Dataset):
    def __init__(self, tumor_dir, notumor_dir, transform=None):
        self.tumor_images = [os.path.join(tumor_dir, f) for f in os.listdir(tumor_dir)]
        self.notumor_images = [os.path.join(notumor_dir, f) for f in os.listdir(notumor_dir)]
        self.images = self.tumor_images + self.notumor_images
        self.labels = [1] * len(self.tumor_images) + [0] * len(self.notumor_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')  # Load image and convert to RGB
        if self.transform:
            image = self.transform(image)
        return image, label

# 1. Data Preprocessing with transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize image to a fixed size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  ])

""" Normalize: Scales pixel values to be between -1 and 1 (using mean=0.5 and std=0.5 for each RGB channel). """

# Paths to your dataset
tumor_dataset_path = r"C:\Users\HP\OneDrive\Bureau\ImageRegionFeatures\data\tumor"
notumor_dataset_path = r"C:\Users\HP\OneDrive\Bureau\ImageRegionFeatures\data\notumor"

# Load dataset using custom Dataset class
dataset = ImageDataset(tumor_dataset_path, notumor_dataset_path, transform=transform)

# Split dataset into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. Define the ANN Model
""" fc1: First input layer with 128 neurons.
fc2: Second hidden layer with 64 neurons.
fc3: Output layer with 2 neurons (for the two classes: tumor and no tumor)."""
class ANN(nn.Module):
    def __init__(self, input_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First layer
        self.fc2 = nn.Linear(128, 64)         # Second layer
        self.fc3 = nn.Linear(64, 2)           # Output layer (2 classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)       # Dropout for regularization


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. Initialize Model, Loss, and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ANN(128 * 128 * 3).to(device)  # Image size is 128x128 with 3 color channels (RGB)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop with Loss Tracking
epochs = 50
train_losses = []  # List to store loss values for plotting
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Flatten images to feed into the ANN
        images = images.view(images.size(0), -1)  # Flatten the 3D image to 1D vector

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() # backpropagation
        optimizer.step() #updates the weights
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# 5. Testing the Model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Flatten images for the ANN
        images = images.view(images.size(0), -1)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
# 30 epochs:  Test Accuracy: 91.70%

# 6. Plot Loss Curve
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss 50')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

torch.save(model.state_dict(), '../models/ann_model50.pth')
print("Model saved as ann_model50.pth")

""" Loss: 0.2866
Test Accuracy: 92.05% """