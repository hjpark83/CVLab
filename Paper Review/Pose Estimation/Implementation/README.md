# Pose Estimation using OpenPose & YOLOv9

## Environment
> Linux
## Model
> - pose_iter_440000.caffemodel
> - YOLOv9s.pt

## Implement
Download pre-trained OpenPose model
```shell
> sh getModels.sh
```
Implement python file
```python
> python PoseEstimation.py --source [input (image/video) name] --output [output (image/video) name]
```

## Output
### Image
![image](https://github.com/user-attachments/assets/31b1ef35-a0f8-448d-9271-ba326899ab96)


### Video
|Soccer|Pole Vault|
|:--:|:--:|
|![EU](https://github.com/user-attachments/assets/a0f8125e-701e-4053-a20c-b321ae1a2400)|![Duplantis](https://github.com/user-attachments/assets/4dc610bd-7571-4724-ab29-7f7fd86aadd1)|
|Baseketball|Baseball|
|![paris](https://github.com/user-attachments/assets/96298cc3-b543-41b3-9fd6-8de5c50e290f)|![kike](https://github.com/user-attachments/assets/616ead8c-ef22-459a-9dd2-13df42f2461e)|
|NewJeans - ETA|PSY - Gangnam Style|
|![eta - Made with Clipchamp (1)](https://github.com/user-attachments/assets/000238c6-f48f-4562-baf6-e6516fa34d2a)|![psy](https://github.com/user-attachments/assets/2b4d3a9c-1beb-4b8c-bd61-00c11cfd6054)|

## reference
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

[YOLOv9](https://github.com/WongKinYiu/yolov9)
