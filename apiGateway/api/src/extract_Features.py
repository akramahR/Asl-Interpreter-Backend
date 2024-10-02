import torch
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu
import numpy as np
from videoGenerator import preprocess_video, load_video
from UTILS import Identity
import os

# Define constants
TARGET_SIZE = (224, 224)
FPS = 32

# Load SlowFast model configuration
args = parse_args()
cfg = load_config(args)
cfg = assert_and_infer_cfg(cfg)

# Build and load the pre-trained model
model = build_model(cfg)
cu.load_test_checkpoint(cfg, model)

# Replace model's head with identity to extract features
model.head = Identity()
model.eval()
model.to(device="cuda:0")

# Function to extract features from a single video
def extract_features_from_video(video_path, start_time=None, end_time=None, fps=FPS):
    """
    Extracts features from a given video.

    Args:
        video_path (str): Path to the video file.
        start_time (float): Start time of the video segment (in seconds).
        end_time (float): End time of the video segment (in seconds).
        fps (int): Frames per second to process the video.

    Returns:
        numpy.ndarray: Extracted features as a numpy array.
    """
    # Load and preprocess the video
    video = load_video(video_path, target_size=TARGET_SIZE, target_fps=fps, start_time=start_time, end_time=end_time)
    video_tensor = torch.tensor(video, dtype=torch.float32).to(device="cuda:0")

    # Extract features using the pre-trained model
    with torch.no_grad():
        features = model(video_tensor.unsqueeze(0))  # Add batch dimension
        features = features.cpu().numpy()  # Move features to CPU and convert to NumPy

    return features

# Example usage: Extract features from a video
video_path = 'path_to_your_video.mp4'  # Replace with the path to your video
start_time = 0  # Optional: Set the start time of the segment
end_time = 5  # Optional: Set the end time of the segment (in seconds)

# Extract features from the video
features = extract_features_from_video(video_path, start_time=start_time, end_time=end_time)

# Print or return the features
print("Extracted features:", features)
