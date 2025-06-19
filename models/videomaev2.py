from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig, VideoMAEConfig
import numpy as np
import torch
import json
from typing import Tuple
import os
import cv2
import numpy as np
import torch
import torchvision
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
import torchaudio
# Initialize model and processor
config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Base", trust_remote_code=True)
processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Base")
model = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-Base', config=config, trust_remote_code=True).cuda()

# Create a longer video (250 frames  # [250, 3, 224, 224]

# def read_video_fast(path: str):
#     frames = []
#     for frame in sorted(os.listdir(path)):
#         if frame.endswith(".jpg"):
#             img_path = os.path.join(path, frame)
#             img = cv2.imread(img_path)
#             if img is None:
#                 print(f"Warning: Failed to read {img_path}")
#                 continue
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, (224, 224))  # Resize to 224x224
#             frames.append(img)
#         print(frame)
#     if not frames:
#         raise ValueError(f"No frames found in {path}")
#     video = np.stack(frames, axis=0)
#     video = rearrange(video, 'T H W C -> T C H W')
#     return torch.from_numpy(video).float() / 255


# read_video_fast("/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/silent_videos/9yhsja3clq/real/1_5")

def process_video_in_chunks(video, chunk_size=16):
        num_frames = video.shape[1]
        num_chunks = num_frames // chunk_size
        
        all_features = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            
            chunk = video[:, start_idx:end_idx]
            
            # Process chunk
            # inputs = self.video_processor(list(chunk[0]), return_tensors="pt")
            # inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4).cuda()
            chunk = chunk.permute(0, 2, 1, 3, 4)
            with torch.no_grad():
                outputs = model(chunk.float())
            print(outputs.shape)
            all_features.append(outputs)
        
        all_features = torch.stack(all_features)
        print(all_features.shape)
        all_features = all_features.type_as(video)
        
        final_features = all_features.mean(dim=0)
        
        return final_features

video = np.random.rand(2, 150, 3, 224, 224)
video = torch.from_numpy(video).cuda()
# # video = video.permute(0, 1, 3, 4, 2)
# inputs = processor(video.float()/255, return_tensors="pt")
features = process_video_in_chunks(video.float())
print(features.shape)
# def process_video_in_chunks(video, chunk_size=16):
#     # Calculate number of chunks
#     num_frames = video.shape[0]
#     num_chunks = num_frames // chunk_size
    
#     all_features = []
    
#     # Process each chunk
#     for i in range(num_chunks):
#         start_idx = i * chunk_size
#         end_idx = start_idx + chunk_size
        
#         # Get chunk of frames
#         chunk = video[start_idx:end_idx]
        
#         # Process chunk
#         # print(chunk.shape)
#         # inputs = processor(list(chunk), return_tensors="pt")
#         # inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4)
#         # chunk = chunk.unsqueeze(0).permute(0, 2, 1, 3, 4)
#         chunk = chunk.permute(0, 2, 1, 3, 4)
#         # print(inputs['pixel_values'].shape)
#         with torch.no_grad():
#             outputs = model(chunk.float())
#             # Get features for this chunk
#             all_features.append(outputs)
    
#     # Stack all chunk features
#     all_features = torch.stack(all_features)
    
#     # Average features across chunks
#     final_features = all_features.mean(dim=0)
    
#     return final_features

# # Process the video
# features = process_video_in_chunks(video)
# print("Final feature shape:", features.shape)
# print("Number of chunks processed:", video.shape[0] // 16)