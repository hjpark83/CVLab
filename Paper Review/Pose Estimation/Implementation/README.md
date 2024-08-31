# Pose Estimation using OpenPose & YOLOv9

This program is designed to analyze images, videos, or live camera feeds by applying object detection and human pose estimation. 
It is particularly useful for applications that require detailed analysis of human movements or identification of objects within a scene.

The program generates two distinct outputs:

1. A visual representation of the original input with detection bounding boxes and overlaid pose estimations for humans, which is useful for visual inspection and analysis.
2. A separate file containing only the pose skeletons, presented on a black background, which can be used for further processing or analysis of human movement.

Pose estimation is applied exclusively to humans, while object detection works on any object with a confidence score above 0.5.

## Environment
> - Linux
## Model
> - pose_iter_440000.caffemodel
> - YOLOv9s.pt

## Implement
Download pre-trained OpenPose model
```shell
> sh getModels.sh
```
Implement
```python
> python main.py --source [input (image/video) name or 'camera'] --output [output (image/video) name]
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
|![bask](https://github.com/user-attachments/assets/855e36d8-4b9b-4771-b537-0b010ec59e31)|![ohtani](https://github.com/user-attachments/assets/1dc1e713-daa1-41ea-b03d-d827f1387251)|
|NewJeans - ETA|BTS - Dynamite|
|![eta](https://github.com/user-attachments/assets/7e6bdeb3-8a70-4281-99c6-502fe0f86d25)|![dynamite](https://github.com/user-attachments/assets/dbf3eb07-ec69-492b-af18-ff1b91683830)|

**Pose Skeleton**
|Soccer|Pole Vault|
|:--:|:--:|
|![EU_skel](https://github.com/user-attachments/assets/2c9ff82d-550b-407e-91e3-ad10b4e6ddad)|![skel](https://github.com/user-attachments/assets/0c019f32-2114-4ed3-aa8c-3670d1cfa850)|
|Baseketball|Baseball|
|![12](https://github.com/user-attachments/assets/4fee5b08-115d-46dd-8a27-428525ebaa45)|![shohei](https://github.com/user-attachments/assets/3447465a-dafa-49a4-a7f8-233bfb555c05)|
|NewJeans - ETA|BTS - Dynamite|
|![eta - Made with Clipchamp](https://github.com/user-attachments/assets/9031b4b8-f0c0-49f2-a9ff-703a2e7b3372)|![dynamite_s](https://github.com/user-attachments/assets/669f01e9-f210-46b8-9fbb-6ea68b1a7067)|

## reference
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

[YOLOv9](https://github.com/WongKinYiu/yolov9)
