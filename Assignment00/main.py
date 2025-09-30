import torch
import os
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transformations: resize all images, convert to tensor, normalize
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # resize images to 128x128
    transforms.ToTensor(),          # convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize between -1 and 1
])

# Load dataset using ImageFolder
dataset = datasets.ImageFolder(root="cats_vs_dogs_dataset", transform=transform)

# Split into training & validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Class names (based on folder names)
class_names = dataset.classes
print("Classes:", class_names)

#The CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)                # reduces image size
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 32 filters
        self.fc1 = nn.Linear(32 * 32 * 32, 64)        # fully connected layer
        self.fc2 = nn.Linear(64, 2)   
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 + relu + pooling
        x = self.pool(F.relu(self.conv2(x)))  # conv2 + relu + pooling
        x = x.view(-1, 32 * 32 * 32)          # flatten
        x = F.relu(self.fc1(x))               # fully connected
        x = self.fc2(x)                       # output layer
        return x
    
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5  # keep it short for demo
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # reset gradients
        outputs = model(images)        # forward pass
        loss = criterion(outputs, labels)  # compute loss
        loss.backward()                # backpropagation
        optimizer.step()               # update weights

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")


model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # add batch dimension
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# Test with one cat and one dog image

print("Prediction:", predict_image("cats_vs_dogs_dataset/cat/cat_12.jpg"))
print("Prediction:", predict_image("cats_vs_dogs_dataset/dog/dog_10.jpg"))

def interactive_predictor_with_display():
    print("\nInteractive Image Predictor (type 'exit' to quit)")
    while True:
        image_path = input("Enter image path: ")
        if image_path.lower() == "exit":
            break
        if not os.path.exists(image_path):
            print("File does not exist. Try again.")
            continue
        
        # Open and display the image
        image = Image.open(image_path)
        image.show()  # This opens the image in a separate window

        # Predict the class
        prediction = predict_image(image_path)
        print(f"Prediction: {prediction}\n")

# Replace the old interactive predictor call
interactive_predictor_with_display()