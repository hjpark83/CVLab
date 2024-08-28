# Pose Estimation using OpenPose & YOLOv9

## Environment
> Linux
## Model
> pose_iter_440000.caffemodel,  YOLOv9s.pt

## Implement
Download pre-trained OpenPose model
```shell
> sh getModels.sh
```
Implement python file
```python
> python PoseEstimation.py --source [input image/video name] --output [output image/video name]
```

## Output
### Image
![image](https://github.com/user-attachments/assets/33a3215a-72b1-465c-aff6-2789ead034e3)

### Video
|Soccer|Pole Vault|
|:--:|:--:|
|||
|Baseball|Baseball|
|||
|Basketball|Basketball|
|||

## reference
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

[YOLOv9](https://github.com/WongKinYiu/yolov9)
