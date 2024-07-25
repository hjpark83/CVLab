import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
from flood_dataset import FloodSegmentationDataset
from model import FCN
# from VGGNet import VGGNet, FCNs


# Define dataset directories
csv_file = 'dataset/flood-area-segmentation/metadata.csv'
image_dir = 'dataset/flood-area-segmentation/Image'
mask_dir = 'dataset/flood-area-segmentation/Mask'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the dataset
dataset = FloodSegmentationDataset(csv_file=csv_file, image_dir=image_dir, mask_dir=mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FCN().to(device)

# For the VGG-based model
# vgg16 = VGGNet(pretrained=True)
# model = FCNs(vgg16, 2).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 25
train_losses = []

def train():
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device).squeeze(1)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    print('Training complete.')
    return model, train_losses, dataloader, device