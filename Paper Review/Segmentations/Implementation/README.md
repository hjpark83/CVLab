# Flood Area Segmentation

## Model
### 1. FCN 
### 2. VGG16 (Pre-trained)
### 3. UNet

## Dataset

### Kaggle - [Flood Area Segmentation dataset](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation/data)

## Implementation
 
    > python main.py
      Enter the base directory for the dataset: <directory_path>

## Result

### 1. FCN
|Label|Epoch=25|Epoch=100|
|:-:|:-:|:--:|
|Image|![image](https://github.com/user-attachments/assets/d3fc202b-d875-4351-b9a6-b521e028a419)|![image](https://github.com/user-attachments/assets/d3fc202b-d875-4351-b9a6-b521e028a419)|
|Ground Truth|![image](https://github.com/user-attachments/assets/e917368e-0816-48d9-82ad-f1697aba7d7d)|![image](https://github.com/user-attachments/assets/e917368e-0816-48d9-82ad-f1697aba7d7d)|
|Prediction|![image](https://github.com/user-attachments/assets/5337a4ab-eac0-4e33-9d83-1979623e7683)|![image](https://github.com/user-attachments/assets/e739bc8d-ecd6-4add-b575-d13e56dfbed0)|

### 2. VGG16
|Label|Epoch=25|Epoch=100|
|:-:|:-:|:--:|
|Image|![image](https://github.com/user-attachments/assets/158aba06-59c7-4590-8c74-ef722f7ce4d7)|![image](https://github.com/user-attachments/assets/158aba06-59c7-4590-8c74-ef722f7ce4d7)|
|Ground Truth|![image](https://github.com/user-attachments/assets/b5148b04-51f6-430d-9d42-54312931c236)|![image](https://github.com/user-attachments/assets/b5148b04-51f6-430d-9d42-54312931c236)|
|Prediction|![image](https://github.com/user-attachments/assets/62574978-17c6-444a-bbb2-94d35ebfd5a4)|![image](https://github.com/user-attachments/assets/e8e1a7aa-0fb8-4552-82c3-67b9e4d1c170)|

### 3. UNet
|Label|Epoch=25|Epoch=100|
|:-:|:-:|:--:|
|Image|![image](https://github.com/user-attachments/assets/7efe6ab3-9075-4e3c-ac3d-247c68fe7758)|![image](https://github.com/user-attachments/assets/7efe6ab3-9075-4e3c-ac3d-247c68fe7758)|
|Ground Truth|![image](https://github.com/user-attachments/assets/b2543ece-2f9a-4d60-ac43-f330c682e170)|![image](https://github.com/user-attachments/assets/62b8eb6f-ee3d-4ad2-8ebd-4d44e9c1ed13)|
|Prediction|![image](https://github.com/user-attachments/assets/551c3409-2150-4394-8ef3-be93d79ac073)|![image](https://github.com/user-attachments/assets/d9defb25-c84d-49e2-a339-25641e29cecf)|

## Training

### 1. FCN
|Epoch=25|Epoch=100|
|:--:|:--:|
|![image](https://github.com/user-attachments/assets/affae839-e382-486f-b09b-394458e8fb52)|![image](https://github.com/user-attachments/assets/461a2fa6-17a8-4e4e-99b9-abb0d1f905b6)|

### 2. VGG16
|Epoch=25|Epoch=100|
|:--:|:--:|
|![image](https://github.com/user-attachments/assets/2153e281-4cab-4ad4-8332-7f7f279d2ce7)|![image](https://github.com/user-attachments/assets/ddb46309-ee18-4853-acc3-d3511efc0714)|

### 3. UNet
|Epoch=25|Epoch=100|
|:--:|:--:|
|![image](https://github.com/user-attachments/assets/eae4531a-9a9c-4d04-9910-48ccbcc1abfc)|![image](https://github.com/user-attachments/assets/4fd66ca5-6dde-4ca1-8494-550852fd4715)|
