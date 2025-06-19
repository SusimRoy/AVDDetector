import os
import random
import json
from collections import defaultdict
import cv2
from insightface.app import FaceAnalysis
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import shutil
import os
import random
import json
from collections import defaultdict
import cv2
from insightface.app import FaceAnalysis
from tqdm import tqdm
import concurrent.futures
from functools import partial
import shutil

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
                face_crop = cv2.resize(face_crop, (224, 224))
                cv2.imwrite(output_path, face_crop)
                face_count += 1

        frame_idx += 1

    cap.release()
    return face_count

def find_missing_subdirectories(base_source_dir, base_extracted_dir, dataset_name):
    """Find subdirectories that exist in source but not in extracted directories"""
    missing_by_identity = defaultdict(list)
    
    if not os.path.exists(base_source_dir):
        print(f"Source directory not found: {base_source_dir}")
        return missing_by_identity
    
    identity_folders = [d for d in os.listdir(base_source_dir) 
                       if os.path.isdir(os.path.join(base_source_dir, d))]
    
    for identity_id in identity_folders:
        source_identity_path = os.path.join(base_source_dir, identity_id)
        extracted_identity_path = os.path.join(base_extracted_dir, identity_id)
        
        # Get all subdirectories in source for this identity
        if dataset_name == 'lrs3':
            # LRS3 structure: identity/segment (direct segments under identity)
            source_subdirs = set(d for d in os.listdir(source_identity_path) 
                               if os.path.isdir(os.path.join(source_identity_path, d)))
        elif dataset_name == 'vox_celeb_2':
            # VoxCeleb2 structure: identity/video_folder/segment
            source_subdirs = set()
            for video_folder in os.listdir(source_identity_path):
                video_path = os.path.join(source_identity_path, video_folder)
                if os.path.isdir(video_path):
                    for segment in os.listdir(video_path):
                        segment_path = os.path.join(video_path, segment)
                        if os.path.isdir(segment_path):
                            source_subdirs.add(f"{video_folder}_{segment}")
        else:
            source_subdirs = set(d for d in os.listdir(source_identity_path) 
                               if os.path.isdir(os.path.join(source_identity_path, d)))
        
        # Get extracted subdirectories for this identity
        extracted_subdirs = set()
        if os.path.exists(extracted_identity_path):
            for subfolder_type in ['fake']:
                subfolder_path = os.path.join(extracted_identity_path, subfolder_type)
                if os.path.exists(subfolder_path):
                    extracted_subdirs.update(d for d in os.listdir(subfolder_path) 
                                           if os.path.isdir(os.path.join(subfolder_path, d)))
        
        # Find missing subdirectories
        missing_subdirs = source_subdirs - extracted_subdirs
        if missing_subdirs:
            missing_by_identity[identity_id] = list(missing_subdirs)
    
    return dict(missing_by_identity)

def find_fake_videos_in_folder(folder_path):
    """Find fake videos and their corresponding JSON files in a folder"""
    fake_videos = []
    
    if not os.path.exists(folder_path):
        return fake_videos
    
    for file in os.listdir(folder_path):
        if file.endswith('.mp4') and 'fake' in file.lower():
            video_path = os.path.join(folder_path, file)
            # Look for corresponding JSON file
            json_file = file.replace('.mp4', '.json')
            json_path = os.path.join(folder_path, json_file)
            
            if os.path.exists(json_path):
                fake_videos.append((video_path, json_path))
            else:
                fake_videos.append((video_path, None))
    
    return fake_videos

def process_single_video(video_info, app):
    """Process a single video - this function will be called by each worker process"""
    video_path, json_path, output_dir = video_info
    
    try:        
        face_count = extract_faces(video_path, output_dir, json_path, app)
        
        # Copy JSON file if it exists
        if json_path and os.path.exists(json_path):
            json_output_path = os.path.join(output_dir, os.path.basename(json_path))
            shutil.copy2(json_path, json_output_path)
        
        return {
            'success': True,
            'video_path': video_path,
            'face_count': face_count,
        }
        
    except Exception as e:
        return {
            'success': False,
            'video_path': video_path,
            'error': str(e),
        }

def collect_videos_to_process(missing_by_identity, base_source_dir, base_extracted_dir, dataset_name='lrs3'):
    """Collect all videos that need to be processed"""
    videos_to_process = []
    
    for identity_id, missing_subdirs in missing_by_identity.items():
        # Select up to 2 random subdirectories
        selected_subdirs = random.sample(missing_subdirs, min(2, len(missing_subdirs)))
        
        for subdir in selected_subdirs:
            # Construct source folder path
            if dataset_name == 'lrs3':
                # LRS3: identity/segment (direct path)
                source_folder = os.path.join(base_source_dir, identity_id, subdir)
            elif dataset_name == 'vox_celeb_2':
                # VoxCeleb2: identity/video_folder/segment
                parts = subdir.split('_')
                if len(parts) >= 2:
                    video_folder = '_'.join(parts[:-1])
                    segment = parts[-1]
                    source_folder = os.path.join(base_source_dir, identity_id, video_folder, segment)
                else:
                    continue
            else:
                source_folder = os.path.join(base_source_dir, identity_id, subdir)
            
            # Find fake videos in this folder
            fake_videos = find_fake_videos_in_folder(source_folder)
            
            if not fake_videos:
                continue
            
            # Select up to 2 random fake videos
            selected_videos = random.sample(fake_videos, min(3, len(fake_videos)))
            
            for video_path, json_path in selected_videos:
                # Create output directory
                output_dir = os.path.join(base_extracted_dir, identity_id, 'fake', subdir)
                videos_to_process.append((video_path, json_path, output_dir))
    
    return videos_to_process

def process_missing_videos_parallel(missing_by_identity, base_source_dir, base_extracted_dir, dataset_name='vox_celeb_2', num_workers=None):
    """Process videos using multiprocessing"""
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Use up to 8 cores
    
    print(f"Using {num_workers} worker processes")
    
    # Collect all videos that need to be processed
    videos_to_process = collect_videos_to_process(missing_by_identity, base_source_dir, base_extracted_dir, dataset_name)
    
    if not videos_to_process:
        print("No videos to process!")
        return
    
    print(f"Total videos to process: {len(videos_to_process)}")
    # print(f"Sample video to process: {videos_to_process[0][0]}")
    # print(f"Sample output directory: {videos_to_process[0][2]}")
    # print(f"Sample JSON path: {videos_to_process[0][1]}")
    
    # Process videos in parallel
    successful_count = 0
    failed_count = 0

    app = FaceAnalysis(providers=['CUDAExecutionProvider'])  # Use CPU for stability in multiprocessing
    app.prepare(ctx_id=0, det_size=(96, 96))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks and get results with progress bar
        results = list(tqdm(
            executor.map(
                lambda video_info: process_single_video(video_info, app),
                videos_to_process
            ),
            total=len(videos_to_process),
            desc="Processing videos"
        ))
    
    # Process results
    for result in results:
        if result['success']:
            successful_count += 1
        else:
            failed_count += 1
            print(f"Failed to process {result['video_path']}: {result['error']}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful_count} videos")
    print(f"Failed: {failed_count} videos")

def main():
    # Set multiprocessing start method (important for some systems)
    mp.set_start_method('spawn', force=True)
    
    base_source_dir = "/home/csgrad/susimmuk/acmdeepfake/data/AV-Deepfake1M-PlusPlus/train/vox_celeb_2"
    base_extracted_dir = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2"
    
    # Find missing subdirectories
    missing_subdirs = find_missing_subdirectories(base_source_dir, base_extracted_dir, 'vox_celeb_2')
    
    if not missing_subdirs:
        print("No missing subdirectories found!")
        return
    
    # print(f"Found missing subdirectories for {len(missing_subdirs)} identities")
    # for identity, subdirs in missing_subdirs.items():
    #     print(f"Identity {identity} has {subdirs} missing subdirectories")
    #     break
    
    # Process the missing videos in parallel
    num_workers = 100  # Adjust based on your system capabilities
    process_missing_videos_parallel(missing_subdirs, base_source_dir, base_extracted_dir, 'vox_celeb_2', num_workers)

if __name__ == "__main__":
    main()