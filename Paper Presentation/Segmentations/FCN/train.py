import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
# from model import FCN
from VGGNet import VGGNet, FCNs

# Example function to train the model
def train(dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = FCN().to(device)
    
    vgg16 = VGGNet(pretrained=True)
    model = FCNs(vgg16, 2).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 100
    train_losses = []

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
