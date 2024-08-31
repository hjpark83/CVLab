import os
import cv2
import numpy as np
from pose_estimation import PoseEstimation, initialize_video_writer

class Process_Media:
    def __init__(self, yolo_model, body_parts, pose_pairs, colors, 
                 pose_proto_file, pose_weights_file, face_proto_file, face_weights_file, hand_proto_file, hand_weights_file):
        
        self.yolo_model = yolo_model
        self.pose_estimator = PoseEstimation(
            yolo_model,
            body_parts,
            pose_pairs,
            colors,
            pose_proto_file,
            pose_weights_file,
            face_proto_file,
            face_weights_file,
            hand_proto_file,
            hand_weights_file
        )


    def make_output(self, source, output_path):
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
            _, pose_image = self.pose_estimator.process_pose(image, startX, startY, endX, endY, (height, width, 3))
            blank_frame = cv2.add(blank_frame, pose_image)
        
        # Save output images
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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            boxes, labels, confidences = self.yolo_model.detect_objects(frame)
            frame = self.yolo_model.draw_boxes(frame, boxes, labels, confidences)

            blank_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

            for (startX, startY, endX, endY) in boxes:
                _, pose_frame = self.pose_estimator.process_pose(frame, startX, startY, endX, endY, (frame_size[1], frame_size[0], 3))
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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            boxes, labels, confidences = self.yolo_model.detect_objects(frame)
            frame = self.yolo_model.draw_boxes(frame, boxes, labels, confidences)

            blank_frame = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)

            for (startX, startY, endX, endY) in boxes:
                _, pose_frame = self.pose_estimator.process_pose(frame, startX, startY, endX, endY, blank_frame.shape)
                blank_frame = cv2.add(blank_frame, pose_frame)

            self.pose_estimator.process_face_with_haar(frame)

            out.write(frame)
            out_pose.write(blank_frame)
            cv2.imshow('Pose Estimation', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        out_pose.release()
        cv2.destroyAllWindows()
        print(f"Finished processing camera stream.")
