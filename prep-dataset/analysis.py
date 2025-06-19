# import cv2
# import os
# import tempfile
# import shutil
# from tqdm import tqdm

# def change_fps_keep_all_frames(video_path, target_fps=25.0):
#     """Change FPS of a video while keeping all frames"""
#     try:
#         # Open original video
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise IOError(f"Failed to open video: {video_path}")

#         # Get original properties
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         codec = cv2.VideoWriter_fourcc(*'mp4v')
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         original_fps = cap.get(cv2.CAP_PROP_FPS)

#         print(f"Processing {video_path}")
#         print(f"  Original FPS: {original_fps}, Target FPS: {target_fps}, Frames: {total_frames}")

#         # Create temporary file in the same directory as the target file
#         # This ensures we're on the same filesystem for the rename operation
#         video_dir = os.path.dirname(video_path)
#         tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mp4', dir=video_dir)
#         os.close(tmp_fd)  # We'll let OpenCV write to it

#         out = cv2.VideoWriter(tmp_path, codec, target_fps, (width, height))

#         frame_count = 0
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             out.write(frame)
#             frame_count += 1

#         cap.release()
#         out.release()

#         # Use shutil.move instead of os.replace for cross-filesystem compatibility
#         shutil.move(tmp_path, video_path)
#         print(f"  Successfully rewritten with {frame_count} frames at {target_fps} FPS")
#         return True

#     except Exception as e:
#         print(f"  Error processing {video_path}: {e}")
#         # Clean up temporary file if it exists
#         try:
#             if 'tmp_path' in locals() and os.path.exists(tmp_path):
#                 os.remove(tmp_path)
#         except:
#             pass
#         return False

# def find_magnified_videos(base_dir, dataset_name):
#     """Find all magnified video files in a dataset directory"""
#     magnified_videos = []
    
#     if not os.path.exists(base_dir):
#         print(f"Directory {base_dir} does not exist!")
#         return magnified_videos
    
#     print(f"Scanning {dataset_name} directory: {base_dir}")
    
#     for identity in os.listdir(base_dir):
#         identity_path = os.path.join(base_dir, identity)
#         if not os.path.isdir(identity_path):
#             continue
        
#         # Check both real and fake folders
#         for folder_type in ['real', 'fake']:
#             type_path = os.path.join(identity_path, folder_type)
            
#             if not os.path.exists(type_path) or not os.path.isdir(type_path):
#                 continue
            
#             # Check each subfolder
#             for subfolder in os.listdir(type_path):
#                 subfolder_path = os.path.join(type_path, subfolder)
                
#                 if not os.path.isdir(subfolder_path):
#                     continue
                
#                 # Look for magnified video files
#                 for file in os.listdir(subfolder_path):
#                     if ('magnified' in file.lower() and 
#                         file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))):
#                         video_path = os.path.join(subfolder_path, file)
#                         magnified_videos.append({
#                             'path': video_path,
#                             'dataset': dataset_name,
#                             'identity': identity,
#                             'type': folder_type,
#                             'subfolder': subfolder,
#                             'file': file
#                         })
    
#     print(f"Found {len(magnified_videos)} magnified videos in {dataset_name}")
#     return magnified_videos

# def process_all_magnified_videos(target_fps=25.0):
#     """Process all magnified videos in LRS3, VoxCeleb2, and Silent Videos"""
    
#     # Define base directories
#     datasets = {
#         'lrs3': '/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/lrs3',
#         'vox_celeb_2': '/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2',
#         'silent_videos': '/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/silent_videos'
#     }
    
#     all_magnified_videos = []
    
#     # Find all magnified videos across all datasets
#     for dataset_name, base_dir in datasets.items():
#         videos = find_magnified_videos(base_dir, dataset_name)
#         all_magnified_videos.extend(videos)
    
#     if not all_magnified_videos:
#         print("No magnified videos found in any dataset!")
#         return
    
#     print(f"\nTotal magnified videos found: {len(all_magnified_videos)}")
    
#     # Group by dataset for reporting
#     by_dataset = {}
#     for video in all_magnified_videos:
#         dataset = video['dataset']
#         if dataset not in by_dataset:
#             by_dataset[dataset] = 0
#         by_dataset[dataset] += 1
    
#     for dataset, count in by_dataset.items():
#         print(f"  {dataset}: {count} videos")
    
#     # Ask for confirmation
#     response = input(f"\nDo you want to change FPS to {target_fps} for all these videos? (y/N): ")
    
#     if response.lower() != 'y':
#         print("Operation cancelled.")
#         return
    
#     # Process all videos
#     successful_count = 0
#     failed_count = 0
    
#     print(f"\nProcessing {len(all_magnified_videos)} magnified videos...")
    
#     for video in tqdm(all_magnified_videos, desc="Processing videos"):
#         success = change_fps_keep_all_frames(video['path'], target_fps)
#         if success:
#             successful_count += 1
#         else:
#             failed_count += 1
    
#     print(f"\n=== PROCESSING COMPLETE ===")
#     print(f"Successfully processed: {successful_count} videos")
#     print(f"Failed: {failed_count} videos")
#     print(f"Total processed: {successful_count + failed_count}")

# def process_specific_dataset(dataset_name, target_fps=25.0):
#     """Process magnified videos for a specific dataset only"""
    
#     datasets = {
#         'lrs3': '/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/lrs3',
#         'vox_celeb_2': '/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2',
#         'silent_videos': '/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/silent_videos'
#     }
    
#     if dataset_name not in datasets:
#         print(f"Unknown dataset: {dataset_name}")
#         print(f"Available datasets: {list(datasets.keys())}")
#         return
    
#     base_dir = datasets[dataset_name]
#     magnified_videos = find_magnified_videos(base_dir, dataset_name)
    
#     if not magnified_videos:
#         print(f"No magnified videos found in {dataset_name}!")
#         return
    
#     print(f"\nFound {len(magnified_videos)} magnified videos in {dataset_name}")
    
#     # Ask for confirmation
#     response = input(f"Do you want to change FPS to {target_fps} for these videos? (y/N): ")
    
#     if response.lower() != 'y':
#         print("Operation cancelled.")
#         return
    
#     # Process videos
#     successful_count = 0
#     failed_count = 0
    
#     print(f"\nProcessing {len(magnified_videos)} magnified videos...")
    
#     for video in tqdm(magnified_videos, desc=f"Processing {dataset_name}"):
#         success = change_fps_keep_all_frames(video['path'], target_fps)
#         if success:
#             successful_count += 1
#         else:
#             failed_count += 1
    
#     print(f"\n=== PROCESSING COMPLETE ===")
#     print(f"Successfully processed: {successful_count} videos")
#     print(f"Failed: {failed_count} videos")

# def check_fps_of_magnified_videos():
#     """Check the current FPS of all magnified videos without changing them"""
    
#     datasets = {
#         'lrs3': '/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/lrs3',
#         'vox_celeb_2': '/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2',
#         'silent_videos': '/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/silent_videos'
#     }
    
#     fps_stats = {}
    
#     for dataset_name, base_dir in datasets.items():
#         videos = find_magnified_videos(base_dir, dataset_name)
#         fps_list = []
        
#         print(f"\nChecking FPS for {dataset_name} videos...")
        
#         for video in tqdm(videos, desc=f"Checking {dataset_name}"):
#             cap = cv2.VideoCapture(video['path'])
#             if cap.isOpened():
#                 fps = cap.get(cv2.CAP_PROP_FPS)
#                 fps_list.append(fps)
#                 cap.release()
        
#         if fps_list:
#             fps_stats[dataset_name] = {
#                 'count': len(fps_list),
#                 'unique_fps': list(set(fps_list)),
#                 'avg_fps': sum(fps_list) / len(fps_list)
#             }
    
#     # Report FPS statistics
#     print("\n=== FPS ANALYSIS ===")
#     for dataset, stats in fps_stats.items():
#         print(f"\n{dataset}:")
#         print(f"  Total videos: {stats['count']}")
#         print(f"  Unique FPS values: {stats['unique_fps']}")
#         print(f"  Average FPS: {stats['avg_fps']:.2f}")

# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) < 2:
#         print("Usage:")
#         print("  python analysis.py all [fps]          - Process all datasets")
#         print("  python analysis.py <dataset> [fps]    - Process specific dataset (lrs3, vox_celeb_2, silent_videos)")
#         print("  python analysis.py check              - Check current FPS without changing")
#         sys.exit(1)
    
#     command = sys.argv[1].lower()
#     target_fps = 25.0
    
#     if len(sys.argv) >= 3:
#         try:
#             target_fps = float(sys.argv[2])
#         except ValueError:
#             print("Invalid FPS value, using default 25.0")
    
#     if command == 'all':
#         process_all_magnified_videos(target_fps)
#     elif command == 'check':
#         check_fps_of_magnified_videos()
#     elif command in ['lrs3', 'vox_celeb_2', 'silent_videos']:
#         process_specific_dataset(command, target_fps)
#     else:
#         print(f"Unknown command: {command}")
#         print("Available commands: all, check, lrs3, vox_celeb_2, silent_videos")


import os
import json
from tqdm import tqdm
from collections import defaultdict

def count_json_entries(json_path):
    """Count real and fake videos in the JSON file"""
    print(f"Analyzing JSON file: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    total_entries = len(data)
    fake_entries = 0
    real_entries = 0
    
    for entry in tqdm(data, desc="Processing JSON entries"):
        # Check if it has fake periods > 0
        if len(entry['fake_segments']) > 0:
        # if 'fake' in entry and 'periods' in entry['fake'] and entry['fake']['periods'] and len(entry['fake']['periods']) > 0:
            fake_entries += 1
        else:
            real_entries += 1
    
    print(f"\nJSON Analysis Results:")
    print(f"  Total entries: {total_entries}")
    print(f"  Real videos: {real_entries}")
    print(f"  Fake videos: {fake_entries}")
    
    return {
        'total': total_entries,
        'real': real_entries,
        'fake': fake_entries
    }

def count_physical_folders():
    """Count real and fake subfolders in the extracted_frames directories and check for JSON files"""
    datasets = {
        'lrs3': '/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/lrs3',
        'vox_celeb_2': '/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2',
        'silent_videos': '/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/silent_videos'
    }
    
    total_stats = {
        'identities': 0,
        'real_folders': 0,
        'real_with_json': 0,
        'fake_folders': 0,
        'fake_with_json': 0
    }
    
    dataset_stats = {}

    real_without_json_paths = []
    fake_without_json_paths = []
    
    for dataset_name, base_dir in datasets.items():
        if not os.path.exists(base_dir):
            print(f"Directory not found: {base_dir}")
            continue
        
        print(f"Scanning {dataset_name} directory: {base_dir}")
        
        identity_count = 0
        real_folders = 0
        real_with_json = 0
        fake_folders = 0
        fake_with_json = 0

        dataset_real_without_json = []
        dataset_fake_without_json = []
        
        # Iterate through each identity folder
        for identity in tqdm(os.listdir(base_dir), desc=f"Processing {dataset_name}"):
            identity_path = os.path.join(base_dir, identity)
            if not os.path.isdir(identity_path):
                continue
            
            identity_count += 1
            
            # Count real folders
            real_path = os.path.join(identity_path, 'real')
            if os.path.exists(real_path) and os.path.isdir(real_path):
                real_subfolders = [d for d in os.listdir(real_path) 
                                 if os.path.isdir(os.path.join(real_path, d))]
                real_folders += len(real_subfolders)
                
                # Check for JSON files in real subfolders
                for subfolder in real_subfolders:
                    subfolder_path = os.path.join(real_path, subfolder)
                    has_json = any(file.endswith('.json') for file in os.listdir(subfolder_path))
                    if has_json:
                        real_with_json += 1
                    else:
                        # Store path of folder without JSON
                        full_path = os.path.join(dataset_name, identity, 'real', subfolder)
                        dataset_real_without_json.append(full_path)
                        real_without_json_paths.append(full_path)
            
            # Count fake folders
            fake_path = os.path.join(identity_path, 'fake')
            if os.path.exists(fake_path) and os.path.isdir(fake_path):
                fake_subfolders = [d for d in os.listdir(fake_path) 
                                 if os.path.isdir(os.path.join(fake_path, d))]
                fake_folders += len(fake_subfolders)
                
                # Check for JSON files in fake subfolders
                for subfolder in fake_subfolders:
                    subfolder_path = os.path.join(fake_path, subfolder)
                    has_json = any(file.endswith('.json') for file in os.listdir(subfolder_path))
                    if has_json:
                        fake_with_json += 1
                    else:
                        # Store path of folder without JSON
                        full_path = os.path.join(dataset_name, identity, 'fake', subfolder)
                        dataset_fake_without_json.append(full_path)
                        fake_without_json_paths.append(full_path)
        
        dataset_stats[dataset_name] = {
                'identities': identity_count,
                'real_folders': real_folders,
                'real_with_json': real_with_json,
                'real_without_json': real_folders - real_with_json,
                'real_without_json_paths': dataset_real_without_json,
                'fake_folders': fake_folders,
                'fake_with_json': fake_with_json,
                'fake_without_json': fake_folders - fake_with_json,
                'fake_without_json_paths': dataset_fake_without_json,
                'total_folders': real_folders + fake_folders,
                'total_with_json': real_with_json + fake_with_json,
                'total_without_json': (real_folders - real_with_json) + (fake_folders - fake_with_json)
            }
            
        total_stats['identities'] += identity_count
        total_stats['real_folders'] += real_folders
        total_stats['real_with_json'] += real_with_json
        total_stats['fake_folders'] += fake_folders
        total_stats['fake_with_json'] += fake_with_json
    
    # Print results for each dataset
    print("\nFolder Structure Analysis:")
    for dataset_name, stats in dataset_stats.items():
        print(f"\n  {dataset_name}:")
        print(f"    Identities: {stats['identities']}")
        print(f"    Real folders: {stats['real_folders']}")
        print(f"      - With JSON: {stats['real_with_json']} ({stats['real_with_json']/stats['real_folders']*100:.1f}% if real_folders > 0)")
        print(f"      - Without JSON: {stats['real_without_json']}")
        print(f"    Fake folders: {stats['fake_folders']}")
        print(f"      - With JSON: {stats['fake_with_json']} ({stats['fake_with_json']/stats['fake_folders']*100:.1f}% if fake_folders > 0)")
        print(f"      - Without JSON: {stats['fake_without_json']}")
        print(f"    Total folders: {stats['total_folders']}")
        print(f"      - With JSON: {stats['total_with_json']} ({stats['total_with_json']/stats['total_folders']*100:.1f}% if total_folders > 0)")
        print(f"      - Without JSON: {stats['total_without_json']}")

        if stats['real_without_json_paths']:
                print(f"\n    Sample real folders without JSON (showing up to 3):")
                for path in stats['real_without_json_paths'][:3]:
                    print(f"      - {path}")
                if len(stats['real_without_json_paths']) > 3:
                    print(f"      - ... and {len(stats['real_without_json_paths'])-3} more")
                    
        if stats['fake_without_json_paths']:
                print(f"\n    Sample fake folders without JSON (showing up to 3):")
                for path in stats['fake_without_json_paths'][:3]:
                    print(f"      - {path}")
                if len(stats['fake_without_json_paths']) > 3:
                    print(f"      - ... and {len(stats['fake_without_json_paths'])-3} more")
    
    # Print overall results
    print("\nOverall Folder Statistics:")
    print(f"  Total identities: {total_stats['identities']}")
    print(f"  Total real folders: {total_stats['real_folders']}")
    print(f"    - With JSON: {total_stats['real_with_json']} ({total_stats['real_with_json']/total_stats['real_folders']*100:.1f}% if real_folders > 0)")
    print(f"    - Without JSON: {total_stats['real_folders'] - total_stats['real_with_json']}")
    print(f"  Total fake folders: {total_stats['fake_folders']}")
    print(f"    - With JSON: {total_stats['fake_with_json']} ({total_stats['fake_with_json']/total_stats['fake_folders']*100:.1f}% if fake_folders > 0)")
    print(f"    - Without JSON: {total_stats['fake_folders'] - total_stats['fake_with_json']}")
    print(f"  Total folders: {total_stats['real_folders'] + total_stats['fake_folders']}")
    print(f"    - With JSON: {total_stats['real_with_json'] + total_stats['fake_with_json']} ({(total_stats['real_with_json'] + total_stats['fake_with_json'])/(total_stats['real_folders'] + total_stats['fake_folders'])*100:.1f}%)")
    print(f"    - Without JSON: {(total_stats['real_folders'] - total_stats['real_with_json']) + (total_stats['fake_folders'] - total_stats['fake_with_json'])}")
    
    return {
        'dataset_stats': dataset_stats,
        'total_stats': total_stats,
        'real_without_json_paths': real_without_json_paths,
        'fake_without_json_paths': fake_without_json_paths
    }

def main():
    print("=== VIDEO DATASET ANALYSIS ===\n")
    
    # 1. Analyze JSON file
    json_path = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/train_metadata_with_audio.json"
    json_stats = count_json_entries(json_path)

    print(f"JSON file entries: {json_stats['total']}")
    print(f"  - Real: {json_stats['real']}")
    print(f"  - Fake: {json_stats['fake']}")

    json_path = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/train_metadata.json"
    json_stats = count_json_entries(json_path)
    
    # 2. Analyze folder structure
    # folder_stats = count_physical_folders()
    
    # 3. Compare results
    print("\n=== COMPARISON ===")
    print(f"JSON file entries: {json_stats['total']}")
    print(f"  - Real: {json_stats['real']}")
    print(f"  - Fake: {json_stats['fake']}")
    # print(f"Physical folders: {folder_stats['total_stats']['real_folders'] + folder_stats['total_stats']['fake_folders']}")
    # print(f"  - Real: {folder_stats['total_stats']['real_folders']}")
    # print(f"  - Fake: {folder_stats['total_stats']['fake_folders']}")

if __name__ == "__main__":
    main()