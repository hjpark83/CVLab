# Pose Estimation using OpenPose & YOLOv9

This program is designed to analyze images, videos, or live camera feeds by applying object detection and human pose estimation. 

It is particularly useful for applications that require detailed analysis of human movements or identification of objects within a scene.

The program generates two distinct outputs:

1. A visual representation of the original input with detection bounding boxes and overlaid pose estimations for humans, which is useful for visual inspection and analysis.
2. A separate file containing only the pose skeletons, presented on a black background, which can be used for further processing or analysis of human movement.

Pose estimation is applied exclusively to humans, while object detection works on any object with a confidence score above 0.5.

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
|![EU](https://github.com/user-attachments/assets/04bc651b-2d22-4d3e-805e-a44dcd4e9009)|![bar](https://github.com/user-attachments/assets/b34612f1-a06a-4ee9-a12a-3dab3a78b3ff)|
|Baseketball|Baseball|
|![paris](https://github.com/user-attachments/assets/96298cc3-b543-41b3-9fd6-8de5c50e290f)|![kike](https://github.com/user-attachments/assets/616ead8c-ef22-459a-9dd2-13df42f2461e)|
|NewJeans - ETA|PSY - Gangnam Style|
|![eta - Made with Clipchamp (1)](https://github.com/user-attachments/assets/000238c6-f48f-4562-baf6-e6516fa34d2a)|![psy](https://github.com/user-attachments/assets/2b4d3a9c-1beb-4b8c-bd61-00c11cfd6054)|

**Pose Skeleton**
|Soccer|Pole Vault|
|:--:|:--:|
|![EU_skel](https://github.com/user-attachments/assets/2c9ff82d-550b-407e-91e3-ad10b4e6ddad)|![skel](https://github.com/user-attachments/assets/0c019f32-2114-4ed3-aa8c-3670d1cfa850)|
|Baseketball|Baseball|
|||
|NewJeans - ETA|PSY - Gangnam Style|
|||

## reference
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

[YOLOv9](https://github.com/WongKinYiu/yolov9)
