import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

data_dir = 'C:/Users/HP/OneDrive/Bureau/ImageRegionFeatures/data'

# Transformations: Resizing, normalization
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to a fixed size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
])

dataset = datasets.ImageFolder(data_dir, transform=transform)

# Split into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. Define the CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional Layer 1 (Layer 1)
        # Input: 3 channels (RGB image), Output: 16 feature maps
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # Max Pooling Layer (Layer 2)
        # Reduces spatial dimensions by half (2x2 pooling with stride 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Convolutional Layer 2 (Layer 3)
        # Input: 16 feature maps, Output: 32 feature maps
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Fully Connected Layer 1 (Layer 4)
        # Input: Flattened feature maps, Output: 128 neurons
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Adjust based on input size
        # Fully Connected Layer 2 (Layer 5 - Output Layer)
        # Input: 128 neurons, Output: 2 neurons (binary classification: tumor or notumor)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: tumor, notumor

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Initialize Model, Loss, and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training the Model
epochs = 20
train_losses = []  # List to store loss values for plotting

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
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
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
# 10 epochs: Test Accuracy: 97.95%
# 6. Plot Loss Curve
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

# Save the Model
torch.save(model.state_dict(), '../models/cnn_model20.pth')
print("Model saved as cnn_model20.pth")

"""Loss: 0.0142
Test Accuracy: 98.40% """