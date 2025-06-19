import os
import random
import cv2
import json
import shutil
from tqdm import tqdm
import concurrent.futures
from insightface.app import FaceAnalysis
import multiprocessing
import psutil
import time

def extract_faces(video_path, output_dir, app=None):
    """
    Extract faces from a video file and save them as individual images
    """
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        if faces:
            for i, face in enumerate(faces):
                bbox = face.bbox.astype(int)

                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                face_crop = frame[y1:y2, x1:x2]

                # Save face crop as image
                output_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
                face_crop = cv2.resize(face_crop, (224, 224))
                cv2.imwrite(output_path, face_crop)
                face_count += 1

        frame_idx += 1

    cap.release()
    return face_count

def get_silent_identity(folder_name):
    """
    Extract identity from folder name
    Example: subject_0_2msdhgqawh_vid_0_0 -> 2msdhgqawh
    """
    return folder_name.split('_')[2]

def process_identity_folder(identity_data, output_base_dir, app):
    """
    Process an identity's videos, extracting frames from real and fake videos
    """
    identity, real_folders, fake_folders = identity_data
    output_identity_dir = os.path.join(output_base_dir, 'silent_videos', identity)
    
    # Process real videos
    for folder_path in real_folders:
        real_video = os.path.join(folder_path, 'real.mp4')
        real_json = os.path.join(folder_path, 'real.json')
        
        if os.path.exists(real_video):
            video_name = os.path.basename(folder_path)
            video_name = video_name.split('vid_')[1]
            output_dir = os.path.join(output_identity_dir, 'real', video_name)
            frame_count = extract_faces(real_video, output_dir, app)
            if os.path.exists(real_json):
                shutil.copy(real_json, os.path.join(output_dir, "real.json"))
    
    # Process fake videos
    for folder_path in fake_folders:
        fake_video = os.path.join(folder_path, 'fake.mp4')
        fake_json = os.path.join(folder_path, 'fake.json')
        
        if os.path.exists(fake_video):
            video_name = os.path.basename(folder_path)
            video_name = video_name.split('vid_')[1]
            output_dir = os.path.join(output_identity_dir, 'fake', video_name)
            frame_count = extract_faces(fake_video, output_dir, app)
            if os.path.exists(fake_json):
                shutil.copy(fake_json, os.path.join(output_dir, "fake.json"))

def main():
    # Directory containing silent_videos in extracted_frames
    base_dir = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/silent_videos"

    # Initialize face detection
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(224, 224))

    # Traverse all identities
    for identity in tqdm(os.listdir(base_dir), desc="Identities"):
        identity_path = os.path.join(base_dir, identity)
        if not os.path.isdir(identity_path):
            continue
        for label in ['real', 'fake']:
            label_path = os.path.join(identity_path, label)
            if not os.path.isdir(label_path):
                continue
            for subfolder in os.listdir(label_path):
                subfolder_path = os.path.join(label_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue
                # Look for json files in this subfolder
                for fname in os.listdir(subfolder_path):
                    if fname.endswith('.json'):
                        json_path = os.path.join(subfolder_path, fname)
                        # Read the JSON file to get the relative video path
                        with open(json_path, 'r') as f:
                            meta = json.load(f)
                        rel_video_path = meta.get('file')
                        if not rel_video_path:
                            print(f"No 'file' attribute in {json_path}, skipping.")
                            continue
                        # Resolve to absolute path
                        video_path = os.path.join("/home/csgrad/susimmuk/acmdeepfake/data/AV-Deepfake1M-PlusPlus/train", rel_video_path)
                        if not os.path.exists(video_path):
                            print(f"Video file not found for {json_path} (expected at {video_path}), skipping.")
                            continue
                        # Output directory is the same as the json file's directory
                        output_dir = subfolder_path
                        # print(f"Extracting faces from {video_path} to {output_dir}")
                        extract_faces(video_path, output_dir, app)

if __name__ == "__main__":
    main() 