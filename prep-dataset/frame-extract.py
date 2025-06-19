import os
import random
import cv2
import json
import shutil
from tqdm import tqdm
import concurrent.futures
from insightface.app import FaceAnalysis


def extract_faces(video_path, output_dir, json_path=None, app=None):

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
                cv2.imwrite(output_path, face_crop)
                face_count += 1

        frame_idx += 1

    cap.release()
    return face_count

def get_video_type(video_name, dataset_name, subfolder_name):
    """
    Determine if a video is real or fake based on its name and dataset
    """
    if dataset_name == 'lrs3':
        # Real videos in lrs3 have parent_folder_p.mp4 format
        if "_p" in video_name and "fake" not in video_name.lower():
            return "real"
        elif "fake" in video_name.lower():
            return "fake"
    elif dataset_name in ['silent_videos', 'vox_celeb_2']:
        # Real videos are named real.mp4
        if video_name == "real.mp4":
            return "real"
        elif "fake" in video_name.lower():
            return "fake"
    
    return None  # Unknown type

def process_identity_folder(identity_path, output_base_dir, app):
    """
    Process an identity folder, selecting random subfolders and extracting frames from real and fake videos
    """
    identity_name = os.path.basename(identity_path)
    dataset_name = os.path.basename(os.path.dirname(identity_path))
    
    # Get all subfolders (0008, 0009, etc.)
    output_identity_dir = os.path.join(output_base_dir, dataset_name, identity_name)
    if dataset_name == 'silent_videos':
        identity_name = identity_path[0]
        dataset_name = 'silent_videos'
        # Check for real.mp4 and fake.mp4 directly in the folder
        real_video = os.path.join(identity_path[1], 'real.mp4')
        fake_video = os.path.join(identity_path[2], 'fake.mp4')
        real_json = os.path.join(identity_path[1], 'real.json')
        fake_json = os.path.join(identity_path[2], 'fake.json')
        
        # Process real video if it exists
        if (os.path.exists(real_video) and os.path.exists(fake_video)) and (os.path.exists(real_json) and os.path.exists(fake_json)):
            video_name = 'real'
            output_dir = os.path.join(output_identity_dir, 'real')
            frame_count = extract_faces(real_video, output_dir, real_json if os.path.exists(real_json) else None, app)
            if os.path.exists(real_json):
                shutil.copy(real_json, os.path.join(output_dir, f"{video_name}.json"))
            # print(f"Extracted {frame_count} frames from {real_video}")
        
            video_name = 'fake'
            output_dir = os.path.join(output_identity_dir, 'fake')
            frame_count = extract_faces(fake_video, output_dir, fake_json if os.path.exists(fake_json) else None, app)
            if os.path.exists(fake_json):
                shutil.copy(fake_json, os.path.join(output_dir, f"{video_name}.json"))
            # print(f"Extracted {frame_count} frames from {fake_video}")

    elif dataset_name == 'vox_celeb_2':
        # Get all video folders (like 1ZyRE-CrDUk)
        video_folders = [f for f in os.listdir(identity_path) if os.path.isdir(os.path.join(identity_path, f))]
        if not video_folders:
            print(f"Warning: {identity_path} doesn't have any video folders")
            return
            
        # Select 5 random video folders
        selected_video_folders = random.sample(video_folders, min(3, len(video_folders)))
        
        for video_folder in selected_video_folders:
            video_folder_path = os.path.join(identity_path, video_folder)
            
            # Get all segment folders (like 00006)
            segment_folders = [f for f in os.listdir(video_folder_path) if os.path.isdir(os.path.join(video_folder_path, f))]
            if not segment_folders:
                continue
                
            # Select 2 random segment folders
            selected_segments = random.sample(segment_folders, min(2, len(segment_folders)))
            
            for segment in selected_segments:
                segment_path = os.path.join(video_folder_path, segment)

                real_video = os.path.join(segment_path, 'real.mp4')
                real_json = os.path.join(segment_path, 'real.json')

                fake_videos = [
                    ('fake_video_fake_audio.mp4', 'fake_video_fake_audio.json'),
                    ('fake_video_real_audio.mp4', 'fake_video_real_audio.json'),
                    ('real_video_fake_audio.mp4', 'real_video_fake_audio.json')
                ]
                fake_videos = random.sample(fake_videos, min(1, len(fake_videos)))
                fake_video = os.path.join(segment_path, fake_videos[0][0])
                fake_json = os.path.join(segment_path, fake_videos[0][1])

                if os.path.exists(real_video) and os.path.exists(fake_video):
                    video_name = f"{video_folder}_{segment}"
                    output_dir = os.path.join(output_identity_dir, 'real', video_name)
                    frame_count = extract_faces(real_video, output_dir, real_json if os.path.exists(real_json) else None, app)
                    if os.path.exists(real_json):
                        shutil.copy(real_json, os.path.join(output_dir, "real.json"))
                    # print(f"Extracted {frame_count} frames from {real_video}")

                    video_name = f"{video_folder}_{segment}"
                    output_dir = os.path.join(output_identity_dir, 'fake', video_name)
                    frame_count = extract_faces(fake_video, output_dir, fake_json if os.path.exists(fake_json) else None, app)
                    if os.path.exists(fake_json):
                        shutil.copy(fake_json, os.path.join(output_dir, f"{fake_videos[0][1]}"))
                
                # Process fake video

                # fake_videos = random.sample(fake_videos, min(1, len(fake_videos)))

                # for fake_video_name, fake_json_name in fake_videos:
                #     fake_video = os.path.join(segment_path, fake_video_name)
                #     fake_json = os.path.join(segment_path, fake_json_name)
                #     if os.path.exists(fake_video):
                #         video_name = f"{video_folder}_{segment}_{fake_video_name.replace('.mp4', '')}"
                #         output_dir = os.path.join(output_identity_dir, 'fake', video_name)
                #         frame_count = extract_faces(fake_video, output_dir, fake_json if os.path.exists(fake_json) else None, app)
                #         if os.path.exists(fake_json):
                #             shutil.copy(fake_json, os.path.join(output_dir, f"{video_name}.json"))
                #         print(f"Extracted {frame_count} frames from {fake_video}")

    else:
        subfolders = [f for f in os.listdir(identity_path) if os.path.isdir(os.path.join(identity_path, f))]

        if not subfolders:
            print(f"Warning: {identity_path} doesn't have any subfolders")
            return
        
        selected_subfolders = random.sample(subfolders, min(2, len(subfolders)))
        output_identity_dir = os.path.join(output_base_dir, dataset_name, identity_name)

        
        for subfolder in selected_subfolders:
            subfolder_path = os.path.join(identity_path, subfolder)
            
            # Find all videos in this subfolder
            videos = [f for f in os.listdir(subfolder_path) if f.endswith('.mp4')]
            
            real_videos = []
            fake_videos = []

            for video in videos:
                video_type = get_video_type(video, dataset_name, subfolder)
                if video_type == "real":
                    video_path = os.path.join(subfolder_path, video)
                    json_path = os.path.join(subfolder_path, video.replace('.mp4', '.json'))
                    if not os.path.exists(json_path):
                        json_path = None
                    real_videos.append((video_path, json_path))
                elif video_type == "fake":
                    video_path = os.path.join(subfolder_path, video)
                    json_path = os.path.join(subfolder_path, video.replace('.mp4', '.json'))
                    if not os.path.exists(json_path):
                        json_path = None
                    fake_videos.append((video_path, json_path))
            
            # Extract frames from real videos
        selected_real_videos = random.sample(real_videos, min(2, len(real_videos)))
        selected_fake_videos = random.sample(fake_videos, min(2, len(fake_videos)))
        
        # Extract frames from selected real videos
        for video_path, json_path in selected_real_videos:
            video_name = os.path.basename(video_path).replace('.mp4', '')
            output_dir = os.path.join(output_identity_dir, 'real', video_name)
            frame_count = extract_faces(video_path, output_dir, json_path, app)
            # Copy the JSON file for reference if it exists
            if json_path and os.path.exists(json_path):
                shutil.copy(json_path, os.path.join(output_dir, f"{video_name}.json"))
            # print(f"Extracted {frame_count} frames from {video_path}")
        
        # Extract frames from selected fake videos
        for video_path, json_path in selected_fake_videos:
            video_name = os.path.basename(video_path).replace('.mp4', '')
            output_dir = os.path.join(output_identity_dir, 'fake', video_name)
            frame_count = extract_faces(video_path, output_dir, json_path, app)
            # Copy the JSON file for reference if it exists
            if json_path and os.path.exists(json_path):
                shutil.copy(json_path, os.path.join(output_dir, f"{video_name}.json"))
            # print(f"Extracted {frame_count} frames from {video_path}")

def get_subject_id(folder_name):
    """
    Extract subject ID from folder name
    Example: from 'subject_0_2msdhgqawh_vid_0_0' returns 'subject_0_2msdhgqawh'
    """
    parts = folder_name.split('_vid_')
    return parts[0] if len(parts) > 0 else folder_name

def get_silent_identity(folder_name):
    # Example: subject_0_2msdhgqawh_vid_0_0 -> 2msdhgqawh
    return folder_name.split('_')[2]

def main():
    # Base directories
    base_dir = "/home/csgrad/susimmuk/acmdeepfake/data/AV-Deepfake1M-PlusPlus/testA"
    output_base_dir = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames"

    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(96, 96))
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each dataset
    # datasets = ['vox_celeb_2']
    # datasets = ['lrs3']/
    datasets = ['vox_celeb_2', 'lrs3']
    
    for dataset in datasets:
        dataset_dir = os.path.join(base_dir, dataset)
        if not os.path.exists(dataset_dir):
            print(f"Warning: Dataset directory not found: {dataset_dir}")
            continue
        
        print(f"Processing dataset: {dataset}")
        
        # Get all identity folders
        if dataset == 'silent_videos':
            all_folders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
            # Group folders by identity
            identity_groups = {}
            for folder in all_folders:
                identity = get_silent_identity(folder)
                if identity not in identity_groups:
                    identity_groups[identity] = []
                identity_groups[identity].append(folder)
            # For each identity, process only if at least 1 real and 1 fake exist

            identity_folders = []
            
            for identity, folders in identity_groups.items():
                real_candidates = []
                fake_candidates = []
                for folder in folders:
                    folder_path = os.path.join(dataset_dir, folder)
                    if os.path.exists(os.path.join(folder_path, 'real.mp4')):
                        real_candidates.append(folder_path)
                    if os.path.exists(os.path.join(folder_path, 'fake.mp4')):
                        fake_candidates.append(folder_path)
                if real_candidates and fake_candidates:
                    # Store tuple: (identity, real_candidates, fake_candidates)
                    selected_real = random.sample(real_candidates, min(5, len(real_candidates)))
                    selected_fake = random.sample(fake_candidates, min(5, len(fake_candidates)))
                    identity_folders.append((identity, selected_real, selected_fake))
                
            
        else:
            extracted_dir = os.path.join(output_base_dir, dataset)

            # List all identity folders in source and extracted
            # all_identities = set(
            #     d for d in os.listdir(dataset_dir)
            #     if os.path.isdir(os.path.join(dataset_dir, d))
            # )
            # already_extracted = set()
            # if os.path.exists(extracted_dir):
            #     already_extracted = set(
            #         d for d in os.listdir(extracted_dir)
            #         if os.path.isdir(os.path.join(extracted_dir, d))
            #     )
            # # Only process those not already extracted
            # to_process = sorted(list(all_identities - already_extracted))

            # identity_folders = [os.path.join(dataset_dir, d) for d in to_process]
            # print(f"Identity folders: {len(identity_folders)}")
            # Original logic for other datasets
            identity_folders = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir) 
                              if os.path.isdir(os.path.join(dataset_dir, d))]
        
        # Process each identity folder with progress tracking
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(
                executor.map(
                    lambda folder: process_identity_folder(folder, output_base_dir, app), 
                    identity_folders
                ),
                total=len(identity_folders),
                desc=f"Processing {dataset}"
            ))

if __name__ == "__main__":
    main()