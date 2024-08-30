import os
import argparse
import cv2
import yaml
import random
import numpy as np
from ultralytics import YOLO

def load_config(config_file='pose.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return (
        config['BODY_PARTS'],
        config['POSE_PAIRS'],
        config['colors'],
        config['pose_protoFile'],
        config['pose_weightsFile'],
        config['face_protoFile'],
        config['face_weightsFile'],
        config['hand_protoFile'],
        config['hand_weightsFile'],
        config['image_path'],
        config['video_path'],
        config['output_image_path'],
        config['output_video_path']
    )


def initialize_video_writer(cap, output_path):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    base_dir = os.path.dirname(os.path.dirname(output_path))
    video_output_dir = os.path.join(base_dir, 'video')
    os.makedirs(video_output_dir, exist_ok=True)

    video_output_path = os.path.join(video_output_dir, os.path.basename(output_path))
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    pose_output_dir = os.path.join(base_dir, 'pose/video')
    os.makedirs(pose_output_dir, exist_ok=True)

    pose_output_path = os.path.join(pose_output_dir, os.path.basename(output_path))
    out_pose = cv2.VideoWriter(pose_output_path, fourcc, fps, (width, height))

    return out, out_pose, (width, height)



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


class PoseEstimation:
    def __init__(self, yolo_model, body_parts, pose_pairs, colors, 
                 pose_proto_file, pose_weights_file, face_proto_file, face_weights_file, hand_proto_file, hand_weights_file):
        
        self.yolo_model = yolo_model
        self.body_parts = body_parts
        self.pose_pairs = pose_pairs
        self.colors = colors
        self.net_pose = cv2.dnn.readNetFromCaffe(pose_proto_file, pose_weights_file)
        self.net_face = cv2.dnn.readNetFromCaffe(face_proto_file, face_weights_file)
        self.net_hand = cv2.dnn.readNetFromCaffe(hand_proto_file, hand_weights_file)


    def get_points(self, input, output, startX, startY, H, W, imageHeight, imageWidth):
        points = []
        for i in range(len(self.body_parts)):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = int(startX + (imageWidth * point[0]) / W)
            y = int(startY + (imageHeight * point[1]) / H)
            if prob > 0.05: 
                points.append((x, y))
            else:
                points.append(None) 
        return points
    
    def draw_pose(self, image, points, pose_pairs):
        for idx, pair in enumerate(pose_pairs):
            partA = points[self.body_parts[pair[0]]]
            partB = points[self.body_parts[pair[1]]]
            if partA and partB:
                color = self.colors[idx % len(self.colors)]
                cv2.line(image, partA, partB, color, 8)
                

    def process_pose(self, input, startX, startY, endX, endY, frame_size):
        roi = input[startY:endY, startX:endX]
        imageHeight, imageWidth, _ = roi.shape

        if roi.dtype != np.uint8:
            roi = roi.astype(np.uint8)

        inpBlob = cv2.dnn.blobFromImage(roi, 1.0 / 255, (imageWidth, imageHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        self.net_pose.setInput(inpBlob)
        output = self.net_pose.forward()

        H = output.shape[2]
        W = output.shape[3]

        points = self.get_points(input, output, startX, startY, H, W, imageHeight, imageWidth)
        
        self.draw_pose(input, points, self.pose_pairs)
        blank_frame = np.zeros(frame_size, dtype=np.uint8)
        self.draw_pose(blank_frame, points, self.pose_pairs)

        return points, blank_frame


    def save_PE(self, frame, points, pose_pairs, frame_size):
        blank_frame = np.zeros(frame_size, dtype=np.uint8)  
        for idx, pair in enumerate(pose_pairs):
            partA = points[self.body_parts[pair[0]]] 
            partB = points[self.body_parts[pair[1]]] 
            if partA and partB:
                color = self.colors[idx % len(self.colors)]
                cv2.line(blank_frame, partA, partB, color, 8)  
                
        for point in points:
            if point:
                cv2.circle(blank_frame, point, 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

        return blank_frame


    def process_media(self, source, output_path):
        if source == "camera":
            self.process_camera(output_path)
        elif source.endswith((".png", ".jpg")):
            self.process_image(source, output_path)
        elif source.endswith((".mp4", ".avi")):
            self.process_video(source, output_path)
        else:
            print(f"Unsupported file format: {source}")


    def process_image(self, image_path, output_path):
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        boxes, labels, confidences = self.yolo_model.detect_objects(image)
        image = self.yolo_model.draw_boxes(image, boxes, labels, confidences)

        blank_frame = np.zeros((height, width, 3), dtype=np.uint8)

        for (startX, startY, endX, endY) in boxes:
            _, pose_image = self.process_pose(image, startX, startY, endX, endY, (height, width, 3))
            blank_frame = cv2.add(blank_frame, pose_image)
            
        base_dir = os.path.dirname(os.path.dirname(output_path)) 
        pose_output_dir = os.path.join(base_dir, 'pose/image')
        os.makedirs(pose_output_dir, exist_ok=True) 
        
        pose_image_path = os.path.join(pose_output_dir, os.path.basename(output_path))
        cv2.imwrite(pose_image_path, blank_frame)
        
        cv2.imwrite(output_path, image)
        
        print(f"Finished processing image: {image_path}")


    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)

        out, out_pose, frame_size = initialize_video_writer(cap, output_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {video_path}, Total frames: {frame_count}, Resolution: {frame_size[0]}x{frame_size[1]}, FPS: {cap.get(cv2.CAP_PROP_FPS)}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            boxes, labels, confidences = self.yolo_model.detect_objects(frame)
            frame = self.yolo_model.draw_boxes(frame, boxes, labels, confidences)

            blank_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

            for (startX, startY, endX, endY) in boxes:
                _, pose_frame = self.process_pose(frame, startX, startY, endX, endY, (frame_size[1], frame_size[0], 3))
                blank_frame = cv2.add(blank_frame, pose_frame)

            out.write(frame)
            out_pose.write(blank_frame)

        cap.release()
        out.release()
        out_pose.release()
        
        print(f"Finished processing video: {video_path}")


    def process_camera(self, output_path):
        cap = cv2.VideoCapture(0)

        out, out_pose, frame_size = initialize_video_writer(cap, output_path)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        out_pose = cv2.VideoWriter(out_pose, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        print(f"Camera opened: Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, FPS: {cap.get(cv2.CAP_PROP_FPS)}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            boxes, labels, confidences = self.yolo_model.detect_objects(frame)
            frame = self.yolo_model.draw_boxes(frame, boxes, labels, confidences)

            blank_frame = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)

            for (startX, startY, endX, endY) in boxes:
                _, pose_frame = self.process_pose(frame, startX, startY, endX, endY, (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3))
                blank_frame = cv2.add(blank_frame, pose_frame)

            self.process_face_with_haar(frame)

            out.write(frame)
            out_pose.write(blank_frame)
            cv2.imshow('Pose Estimation', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        out_pose.release()
        cv2.destroyAllWindows()
        
        print(f"Finished processing camera stream. Video saved to: {out} and Pose-only video saved to: {out_pose}")


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


def parse_args():
    parser = argparse.ArgumentParser(description="Pose Estimation using OpenPose model")
    parser.add_argument('--source', type=str, required=True, help="Name of the input file (image or video)")
    parser.add_argument('--output', type=str, required=True, help="Name of the output file (image or video)")

    args = parser.parse_args()
    return args


def main():
    (
        BODY_PARTS, 
        POSE_PAIRS, 
        colors, 
        protoFile, 
        weightsFile, 
        face_protoFile, 
        face_weightsFile, 
        hand_protoFile, 
        hand_weightsFile, 
        image_path, 
        video_path, 
        output_image_path, 
        output_video_path
    ) = load_config()

    args = parse_args()

    yolo_model = YOLOModel(model_path='yolov9s.pt')
    PE = PoseEstimation(yolo_model, BODY_PARTS, POSE_PAIRS, colors, protoFile, weightsFile, face_protoFile, face_weightsFile, hand_protoFile, hand_weightsFile)

    if args.source == "camera":
        source_file_path = args.source
        output_file_path = args.output
    else:
        source_file_path = os.path.join(image_path if args.source.endswith((".png", ".jpg")) else video_path, args.source)
        output_file_path = os.path.join(output_image_path if args.source.endswith((".png", ".jpg")) else output_video_path, args.output)


    PE.process_media(source_file_path, output_file_path)

if __name__ == "__main__":
    main()