import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import os
from train import train

def main():
    # Train the model
    model, train_losses, dataloader, device = train()

    # Plot the loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss over Epochs')
    plt.show()

    # Create output directory if it does not exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # Visualize some results using OpenCV (save to disk)
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device).squeeze(1)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for j in range(min(3, images.size(0))):  # Save up to 3 images per batch
                img = images[j].cpu().permute(1, 2, 0).numpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

                mask = masks[j].cpu().numpy()
                pred = preds[j].cpu().numpy()

                # Normalize and convert to uint8 for OpenCV
                img = (img * 255).astype(np.uint8)
                mask = (mask * 255).astype(np.uint8)
                pred = (pred * 255).astype(np.uint8)

                # Save images using OpenCV
                cv2.imwrite(f'output/original_{i}_{j}.png', img)
                cv2.imwrite(f'output/ground_truth_{i}_{j}.png', mask)
                cv2.imwrite(f'output/prediction_{i}_{j}.png', pred)

            if i >= 2:  # Only save for the first few batches
                break

if __name__ == "__main__":
    main()
