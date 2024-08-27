# Pose Estimation using OpenPose & YOLOv9

## Environment
> Linux
## Model
> pose_iter_440000.caffemodel,  YOLOv9s.pt

## Implement
Download pre-trained OpenPose model
```shell
sh getModels.sh
```
Implement python file
```python
python PoseEstimation.py --source [input image/video name] --output [output image/video name]
```

## Output
### Image
![image](https://github.com/user-attachments/assets/e07b6632-232b-4acd-bd15-fd2f4b9a7c87)

### Video
|-|-|
|:--:|:--:|
|![EURO2024](https://github.com/user-attachments/assets/ba405671-7761-49ea-8314-bd3b0b40e1c0)|![duplantis](https://github.com/user-attachments/assets/e1e3007a-1100-4e1a-a972-a7b84f2eef19)|
|![ohtani (1)](https://github.com/user-attachments/assets/f70da5f7-5171-4015-9e9f-122275003cee)|![kike - Made with Clipchamp](https://github.com/user-attachments/assets/66abf01d-3f1d-4663-be4f-65d34d876298)|
|![basket - Made with Clipchamp (1)](https://github.com/user-attachments/assets/5a7bc22b-b428-4ac9-ad59-d612608433e1)||

## reference
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

[YOLOv9](https://github.com/WongKinYiu/yolov9)
