import os
import cv2
import numpy as np

def initialize_video_writer(cap, output_path):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # video
    base_dir = os.path.dirname(os.path.dirname(output_path))
    video_output_dir = os.path.join(base_dir, 'video')
    os.makedirs(video_output_dir, exist_ok=True)

    video_output_path = os.path.join(video_output_dir, os.path.basename(output_path))
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    # pose
    pose_output_dir = os.path.join(base_dir, 'pose/video')
    os.makedirs(pose_output_dir, exist_ok=True)

    pose_output_path = os.path.join(pose_output_dir, os.path.basename(output_path))
    out_pose = cv2.VideoWriter(pose_output_path, fourcc, fps, (width, height))

    return out, out_pose, (width, height)


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