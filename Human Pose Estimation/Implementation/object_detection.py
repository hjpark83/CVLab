import cv2
import random
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path='yolov9s.pt'):
        self.model = YOLO(model_path).to('cuda')
        self.colors = {}


    def detect_objects(self, image):
        results = self.model.predict(image)
        boxes = []
        labels = []
        confidences = []

        for r in results:
            for box in r.boxes:
                confidence = float(box.conf)
                if confidence > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = self.model.names[int(box.cls)]
                    boxes.append([x1, y1, x2, y2])
                    labels.append(label)
                    confidences.append(confidence)

                    if label not in self.colors:
                        self.colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return boxes, labels, confidences


    def draw_boxes(self, image, boxes, labels=None, confidences=None):
        for i, (startX, startY, endX, endY) in enumerate(boxes):
            label = labels[i] if labels is not None else ''
            color = self.colors.get(label, (0, 255, 0))

            cv2.rectangle(image, (startX, startY), (endX, endY), color, 5)

            confidence = confidences[i] if confidences is not None else ''
            text = f"{label}: {confidence:.2f}" if confidence else label

            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 1)
            y = max(startY, text_height + 10)

            cv2.rectangle(image, (startX, y - text_height - 10),
                          (startX + text_width, y + baseline - 10),
                          color, cv2.FILLED)
            cv2.putText(image, text, (startX, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image
    
    
    def detect_faces_haar(self, image):
        face_cascade = cv2.CascadeClassifier('models/face/haarcascade_frontalface_alt.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces


    def process_face_with_haar(self, image):
        faces = self.detect_faces_haar(image)
        for (x, y, w, h) in faces:
            roi = image[y:y+h, x:x+w]
            
            if len(roi.shape) == 2 or roi.shape[2] == 1:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

            face_keypoints, _ = self.process_pose(roi, 0, 0, w, h, roi.shape)