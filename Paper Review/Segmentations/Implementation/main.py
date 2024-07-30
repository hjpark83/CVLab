import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from train import train
from load_dataset import SegmentationDataset

def main():
    # Get dataset directory from user input
    base_dir = input("Enter the directory path for the dataset: ")

    # Define dataset directories based on user input
    csv_file = os.path.join(base_dir, 'metadata.csv')
    image_dir = os.path.join(base_dir, 'Image')
    mask_dir = os.path.join(base_dir, 'Mask')

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load the dataset
    dataset = SegmentationDataset(csv_file=csv_file, image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train the model
    model, train_losses, dataloader, device = train(dataloader=dataloader)
    
    # Create output directories if they do not exist
    output_dirs = ['output/original', 'output/ground_truth', 'output/prediction']
    for dir in output_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Visualize all results using OpenCV (save to disk)
    model.eval()
    with torch.no_grad():
        img_num = 0  # Counter to keep track of image index
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device).squeeze(1)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for j in range(images.size(0)):  # Save all images in the batch
                img = images[j].cpu().permute(1, 2, 0).numpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

                mask = masks[j].cpu().numpy()
                pred = preds[j].cpu().numpy()

                # Normalize and convert to uint8 for OpenCV
                img = (img * 255).astype(np.uint8)
                mask = (mask * 255).astype(np.uint8)
                pred = (pred * 255).astype(np.uint8)

                # Save images using OpenCV in separate directories
                cv2.imwrite(f'output/original/original_{img_num}.png', img)
                cv2.imwrite(f'output/ground_truth/ground_truth_{img_num}.png', mask)
                cv2.imwrite(f'output/prediction/prediction_{img_num}.png', pred)

                img_counter += 1  # Increment the image counter

if __name__ == "__main__":
    main()
