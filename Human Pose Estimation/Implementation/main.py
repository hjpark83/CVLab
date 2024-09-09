import os
import argparse
from pose_estimation import PoseEstimation
from object_detection import YOLOModel
from process import Process_Media
from config import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Pose Estimation using OpenPose model")
    parser.add_argument('--source', type=str, required=True, help="Name of the input file (image or video)")
    parser.add_argument('--output', type=str, required=True, help="Name of the output file (image or video)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    yolo_model = YOLOModel(model_path='yolov9s.pt')
    PE = PoseEstimation(
        yolo_model,
        config.BODY_PARTS,
        config.POSE_PAIRS,
        config.colors,
        config.pose_protoFile,
        config.pose_weightsFile,
        config.face_protoFile,
        config.face_weightsFile,
        config.hand_protoFile,
        config.hand_weightsFile
    )

    process_media = Process_Media(
        yolo_model,
        config.BODY_PARTS,
        config.POSE_PAIRS,
        config.colors,
        config.pose_protoFile,
        config.pose_weightsFile,
        config.face_protoFile,
        config.face_weightsFile,
        config.hand_protoFile,
        config.hand_weightsFile
    )

    if args.source == "camera":
        source_file_path = args.source
        output_file_path = args.output
    else:
        source_file_path = os.path.join(config.image_path if args.source.endswith((".png", ".jpg")) else config.video_path, args.source)
        output_file_path = os.path.join(config.output_image_path if args.source.endswith((".png", ".jpg")) else config.output_video_path, args.output)

    process_media.make_output(source_file_path, output_file_path)
    
if __name__ == "__main__":
    main()
