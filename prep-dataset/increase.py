import os
import json
import random
import cv2
import shutil
from collections import defaultdict
from insightface.app import FaceAnalysis
from tqdm import tqdm
import concurrent.futures

def extract_paths_from_json_files(base_extracted_dir):
    """Extract file paths from JSON files in fake subfolders"""
    used_paths = set()
    
    if not os.path.exists(base_extracted_dir):
        print(f"Extracted directory not found: {base_extracted_dir}")
        return used_paths
    
    for identity in os.listdir(base_extracted_dir):
        identity_path = os.path.join(base_extracted_dir, identity)
        if not os.path.isdir(identity_path):
            continue
            
        fake_path = os.path.join(identity_path, 'fake')
        if not os.path.exists(fake_path) or not os.path.isdir(fake_path):
            continue
            
        # Go through each subfolder under fake
        for subfolder in os.listdir(fake_path):
            subfolder_path = os.path.join(fake_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
                
            # Look for JSON files in this subfolder
            for file in os.listdir(subfolder_path):
                if file.endswith('.json'):
                    json_path = os.path.join(subfolder_path, file)
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            if 'file' in data:
                                file_path = data['file']
                                # Split by "/" and take first two parts
                                path_parts = file_path.split('/')
                                if len(path_parts) >= 2:
                                    used_subdir_path = '/'.join(path_parts[:3])
                                    used_paths.add((identity, used_subdir_path))
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Error reading {json_path}: {e}")
    
    return used_paths

def find_alternative_subdirectories(base_source_dir, used_paths, max_alternatives=2):
    """Find alternative subdirectories that haven't been used"""
    alternatives_by_identity = defaultdict(list)
    
    if not os.path.exists(base_source_dir):
        print(f"Source directory not found: {base_source_dir}")
        return alternatives_by_identity
    
    # Convert used_paths to a dict for easier lookup
    used_by_identity = defaultdict(set)
    for identity, used_path in used_paths:
        used_by_identity[identity].add(used_path)
    
    # Go through each identity in the source directory
    for identity in os.listdir(base_source_dir):
        identity_path = os.path.join(base_source_dir, identity)
        if not os.path.isdir(identity_path):
            continue
            
        # Get all available subdirectories for this identity
        available_subdirs = []
        for subdir in os.listdir(identity_path):
            subdir_path = os.path.join(identity_path, subdir)
            if os.path.isdir(subdir_path):
                subdir_relative_path = f"{identity}/{subdir}"
                # Check if this subdirectory hasn't been used
                if subdir_relative_path not in used_by_identity[identity]:
                    available_subdirs.append(subdir)
        
        # Randomly select up to max_alternatives subdirectories
        if available_subdirs:
            selected_count = min(max_alternatives, len(available_subdirs))
            selected_subdirs = random.sample(available_subdirs, selected_count)
            
            # Create full paths
            for subdir in selected_subdirs:
                full_path = os.path.join(base_source_dir, identity, subdir)
                alternatives_by_identity[identity].append(full_path)
    
    return dict(alternatives_by_identity)

def extract_faces(video_path, output_dir, json_path=None, app=None):
    """Extract faces from a video and save as images"""
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

def check_fake_videos_in_paths(alternatives):
    """Check how many alternative paths contain at least one fake video"""
    paths_with_fake_videos = []
    paths_without_fake_videos = []
    
    total_fake_videos = 0
    
    for identity, paths in alternatives.items():
        for path in paths:
            if not os.path.exists(path):
                continue
                
            fake_videos = []
            try:
                for file in os.listdir(path):
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) and 'fake' in file.lower():
                        video_path = os.path.join(path, file)
                        
                        # Look for corresponding JSON file
                        json_file = file.replace('.mp4', '.json').replace('.avi', '.json').replace('.mov', '.json').replace('.mkv', '.json')
                        json_path = os.path.join(path, json_file)
                        
                        if os.path.exists(json_path):
                            fake_videos.append((video_path, json_path))
                        else:
                            fake_videos.append((video_path, None))
                        
                        total_fake_videos += 1
                
                if fake_videos:
                    paths_with_fake_videos.append((path, fake_videos))
                else:
                    paths_without_fake_videos.append(path)
                    
            except OSError as e:
                print(f"Error accessing {path}: {e}")
                paths_without_fake_videos.append(path)
    
    return paths_with_fake_videos, paths_without_fake_videos, total_fake_videos

def process_single_video(video_info, app):
    """Process a single video and extract faces"""
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
            'output_dir': output_dir
        }
        
    except Exception as e:
        return {
            'success': False,
            'video_path': video_path,
            'error': str(e)
        }

def collect_videos_to_process(paths_with_fake, base_extracted_dir):
    """Collect videos to process from paths with fake videos"""
    videos_to_process = []
    
    for path, fake_videos in paths_with_fake:
        # Extract identity from path
        # Path format: /path/to/source/identity/segment
        path_parts = path.split('/')
        identity = path_parts[-2]  # Second to last part is identity
        segment = path_parts[-1]   # Last part is segment
        
        # Select up to 2 random fake videos
        selected_videos = random.sample(fake_videos, min(2, len(fake_videos)))
        
        for i, (video_path, json_path) in enumerate(selected_videos):
            # Create unique output directory name
            # Format: segment_videoname_idx
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_subdir = f"{segment}_{video_name}_{i+1}"
            
            # Create output directory path
            output_dir = os.path.join(base_extracted_dir, identity, 'fake', output_subdir)
            
            videos_to_process.append((video_path, json_path, output_dir))
    
    return videos_to_process

def process_videos_parallel(videos_to_process, num_workers=None):
    """Process videos using ThreadPoolExecutor"""
    
    if num_workers is None:
        num_workers = min(os.cpu_count(), 8)  # Use up to 8 threads
    
    print(f"Using {num_workers} worker threads")
    
    if not videos_to_process:
        print("No videos to process!")
        return
    
    print(f"Total videos to process: {len(videos_to_process)}")
    
    # Initialize face analysis app
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(96, 96))
    
    # Process videos in parallel
    successful_count = 0
    failed_count = 0
    
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
            print(f"✓ Processed: {result['video_path']} -> {result['output_dir']} ({result['face_count']} faces)")
        else:
            failed_count += 1
            print(f"✗ Failed: {result['video_path']} - {result['error']}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful_count} videos")
    print(f"Failed: {failed_count} videos")

def main():
    # Base directories
    base_extracted_dir = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/lrs3"
    base_source_dir = "/home/csgrad/susimmuk/acmdeepfake/data/AV-Deepfake1M-PlusPlus/train/lrs3"
    
    print("Step 1: Extracting used paths from JSON files...")
    used_paths = extract_paths_from_json_files(base_extracted_dir)
    print(f"Found {len(used_paths)} used subdirectory paths")
    # how to get first set object
    # if not used_paths:
    #     print("No used paths found!")
    #     return [], []
    # used_paths = list(used_paths)  # Convert to list for easier indexing
    # print(f"Example used path: {used_paths[0]}")    
    # print(used_paths[0])
    print(used_paths['uSDSzEToJQ4'])
    # print("\nStep 2: Finding alternative subdirectories...")
    alternatives = find_alternative_subdirectories(base_source_dir, used_paths, max_alternatives=2)
    
    print(f"Found alternatives for {len(alternatives)} identities")
    for key,item in alternatives.items():
        print(f"{key}: {item} alternatives")
        break
    # print("\nStep 3: Checking for fake videos in alternative paths...")
    # paths_with_fake, paths_without_fake, total_fake_videos = check_fake_videos_in_paths(alternatives)
    
    # print(f"\nFake video analysis:")
    # print(f"Paths with at least 1 fake video: {len(paths_with_fake)}")
    # print(f"Paths without fake videos: {len(paths_without_fake)}")
    # print(f"Total fake videos found: {total_fake_videos}")
    # print(paths_with_fake[0])  # Show first 5 paths with fake videos
    # if paths_with_fake:
    #     print("\nStep 4: Processing fake videos...")
    #     videos_to_process = collect_videos_to_process(paths_with_fake, base_extracted_dir)
    #     process_videos_parallel(videos_to_process, num_workers=8)
    # else:
    #     print("No fake videos found to process!")
    
    return alternatives, paths_with_fake

if __name__ == "__main__":
    alternatives, paths_with_fake = main()