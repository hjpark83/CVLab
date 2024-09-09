import json

class Config:
    def __init__(self, config_file='pose.json'):
        with open(config_file, 'r') as file:
            config = json.load(file)
        
        self.BODY_PARTS = config['BODY_PARTS']
        self.POSE_PAIRS = config['POSE_PAIRS']
        self.colors = config['colors']
        self.pose_protoFile = config['pose_protoFile']
        self.pose_weightsFile = config['pose_weightsFile']
        self.face_protoFile = config['face_protoFile']
        self.face_weightsFile = config['face_weightsFile']
        self.hand_protoFile = config['hand_protoFile']
        self.hand_weightsFile = config['hand_weightsFile']
        self.image_path = config['image_path']
        self.video_path = config['video_path']
        self.output_image_path = config['output_image_path']
        self.output_video_path = config['output_video_path']
