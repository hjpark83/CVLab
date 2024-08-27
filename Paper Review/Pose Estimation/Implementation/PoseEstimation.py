import os
import argparse
import cv2
import yaml
import random
from ultralytics import YOLO

def load_config(config_file='pose.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    return (
        config['BODY_PARTS'],
        config['POSE_PAIRS'],
        config['colors'],
        config['protoFile'],
        config['weightsFile'],
        config['image_path'],
        config['video_path'],
        config['output_image_path'],
        config['output_video_path']
    )


class YOLOModel:
    def __init__(self, model_path='yolov9s.pt'):
        self.model = YOLO(model_path)
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
    def __init__(self, yolo_model, body_parts, pose_pairs, colors, proto_file, weights_file):
        self.yolo_model = yolo_model
        self.body_parts = body_parts
        self.pose_pairs = pose_pairs
        self.colors = colors
        self.net_pose = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    def process_image(self, image, output_path, image_name):
        boxes, labels, confidences = self.yolo_model.detect_objects(image)
        image = self.yolo_model.draw_boxes(image, boxes, labels, confidences)

        for (startX, startY, endX, endY) in boxes:
            roi = image[startY:endY, startX:endX]

            imageHeight, imageWidth, _ = roi.shape
            inpBlob = cv2.dnn.blobFromImage(roi, 1.0/255, (imageWidth, imageHeight),
                                            (0, 0, 0), swapRB=False, crop=False)

            self.net_pose.setInput(inpBlob)
            output = self.net_pose.forward()

            H = output.shape[2]
            W = output.shape[3]

            points = []
            for i in range(len(self.body_parts)):
                probMap = output[0, i, :, :]
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                x = (startX + (imageWidth * point[0]) / W)
                y = (startY + (imageHeight * point[1]) / H)
                if prob > 0.1:
                    points.append((int(x), int(y)))
                else:
                    points.append(None)

            for idx, pair in enumerate(self.pose_pairs):
                partA = self.body_parts[pair[0]]
                partB = self.body_parts[pair[1]]
                if points[partA] and points[partB]:
                    color = self.colors[idx % len(self.colors)]
                    cv2.line(image, points[partA], points[partB], color, 8)

        cv2.imwrite(output_path, image)

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {video_path}, Total frames: {frame_count}, Resolution: {width}x{height}, FPS: {fps}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            boxes, labels, confidences = self.yolo_model.detect_objects(frame)
            frame = self.yolo_model.draw_boxes(frame, boxes, labels, confidences)

            for (startX, startY, endX, endY) in boxes:
                roi = frame[startY:endY, startX:endX]
                imageHeight, imageWidth, _ = roi.shape
                inpBlob = cv2.dnn.blobFromImage(roi, 1.0 / 255, (imageWidth, imageHeight),
                                                (0, 0, 0), swapRB=False, crop=False)
                self.net_pose.setInput(inpBlob)
                output = self.net_pose.forward()

                H = output.shape[2]
                W = output.shape[3]

                points = []
                for i in range(len(self.body_parts)):
                    probMap = output[0, i, :, :]
                    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                    x = (startX + (imageWidth * point[0]) / W)
                    y = (startY + (imageHeight * point[1]) / H)
                    if prob > 0.1:
                        points.append((int(x), int(y)))
                    else:
                        points.append(None)

                for idx, pair in enumerate(self.pose_pairs):
                    partA = self.body_parts[pair[0]]
                    partB = self.body_parts[pair[1]]
                    if points[partA] and points[partB]:
                        color = self.colors[idx % len(self.colors)]
                        cv2.line(frame, points[partA], points[partB], color, 5)
            out.write(frame)

        cap.release()
        out.release()
        print(f"Finished processing video: {video_path}, output saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pose Estimation using OpenPose model")
    parser.add_argument('--source', type=str, required=True, help="Name of the input file (image or video)")
    parser.add_argument('--output', type=str, required=True, help="Name of the output file (image or video)")

    args = parser.parse_args()
    return args


def main():
    (   BODY_PARTS, 
        POSE_PAIRS, 
        colors, 
        protoFile, 
        weightsFile, 
        image_path, 
        video_path, 
        output_image_path, 
        output_video_path
    ) = load_config()

    args = parse_args()

    image_file_path = os.path.join(image_path, args.source)
    video_file_path = os.path.join(video_path, args.source)

    if args.source.endswith(".png") or args.source.endswith(".jpg"):
        output_file_path = os.path.join(output_image_path, args.output)
    elif args.source.endswith(".mp4") or args.source.endswith(".avi"):
        output_file_path = os.path.join(output_video_path, args.output)
    else:
        output_file_path = None

    yolo_model = YOLOModel(model_path='yolov9s.pt')
    PE = PoseEstimation(yolo_model, BODY_PARTS, POSE_PAIRS, colors, protoFile, weightsFile)

    if os.path.isfile(image_file_path):
        if image_file_path.endswith(".png") or image_file_path.endswith(".jpg"):
            image = cv2.imread(image_file_path)
            PE.process_image(image, output_file_path, os.path.basename(image_file_path))
        else:
            print(f"Unsupported image file format: {os.path.basename(image_file_path)}")
    elif os.path.isfile(video_file_path):
        if video_file_path.endswith(".mp4") or video_file_path.endswith(".avi"):
            PE.process_video(video_file_path, output_file_path)
        else:
            print(f"Unsupported video file format: {os.path.basename(video_file_path)}")
    else:
        print(f"Source file {args.source} does not exist in either image or video paths.")

if __name__ == "__main__":
    main()

