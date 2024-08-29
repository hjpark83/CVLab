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
<img src="https://github.com/user-attachments/assets/2016fa5c-3dd9-4e1f-9b9d-ff0f696c7dc7" width="800" height="auto" />

If you want to see image rather than gif, see here.

>[pose_estimated_img.png](https://github.com/hjpark83/CVLab/blob/main/Paper%20Review/Pose%20Estimation/Implementation/result/pose_estimated_img.png)
>
>[pose_estimated_skeleton.png](https://github.com/hjpark83/CVLab/blob/main/Paper%20Review/Pose%20Estimation/Implementation/result/pose_estimated_skeleton.png)

### Video
**Pose Estimated Video**
|Soccer|Pole Vault|
|:--:|:--:|
|![EU](https://github.com/user-attachments/assets/a0f8125e-701e-4053-a20c-b321ae1a2400)|![Duplantis](https://github.com/user-attachments/assets/4dc610bd-7571-4724-ab29-7f7fd86aadd1)|
|Baseketball|Baseball|
|![paris](https://github.com/user-attachments/assets/96298cc3-b543-41b3-9fd6-8de5c50e290f)|![kike](https://github.com/user-attachments/assets/616ead8c-ef22-459a-9dd2-13df42f2461e)|
|NewJeans - ETA|PSY - Gangnam Style|
|![eta - Made with Clipchamp (1)](https://github.com/user-attachments/assets/000238c6-f48f-4562-baf6-e6516fa34d2a)|![psy](https://github.com/user-attachments/assets/2b4d3a9c-1beb-4b8c-bd61-00c11cfd6054)|

**Pose Skeleton**

## reference
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

[YOLOv9](https://github.com/WongKinYiu/yolov9)
