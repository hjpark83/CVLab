from ultralytics import YOLO
import glob

model = YOLO('best.pt')

# jpg 파일 리스트 생성
image_files = glob.glob('Image/test/images/*.jpg')

# 이미지 파일 리스트를 source에 전달
model.predict(source=image_files, imgsz=640, conf=0.5, show=True, save=True, save_dir='output')
