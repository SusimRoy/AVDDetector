# import os
# import shutil
# from tqdm import tqdm

# def has_magnified_file(folder_path):
#     """Check if folder contains any file with 'magnified' in its name"""
#     try:
#         for file in os.listdir(folder_path):
#             if 'magnified' in file.lower():
#                 return True
#         return False
#     except OSError:
#         return False

# def delete_folders_without_magnified():
#     """Delete subfolders that don't contain files with 'magnified' in the name"""
#     base_path = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2"
    
#     if not os.path.exists(base_path):
#         print(f"Base path {base_path} does not exist!")
#         return
    
#     deleted_count = 0
    
#     # Iterate through each identity folder
#     for identity in tqdm(os.listdir(base_path)):
#         identity_path = os.path.join(base_path, identity)
        
#         if not os.path.isdir(identity_path):
#             continue
            
#         # print(f"Processing identity: {identity}")
        
#         # Check both 'real' and 'fake' folders
#         for folder_type in ['real', 'fake']:
#             type_path = os.path.join(identity_path, folder_type)
            
#             if not os.path.exists(type_path) or not os.path.isdir(type_path):
#                 # print(f"  Skipping {folder_type} - folder doesn't exist")
#                 continue
                
#             # print(f"  Checking {folder_type} folder...")
            
#             # Check each subfolder
#             for subfolder in os.listdir(type_path):
#                 subfolder_path = os.path.join(type_path, subfolder)
                
#                 if not os.path.isdir(subfolder_path):
#                     continue
                    
#                 if not has_magnified_file(subfolder_path):
#                     # print(f"    Deleting: {subfolder_path}")
#                     try:
#                         shutil.rmtree(subfolder_path)
#                         deleted_count += 1
#                     except Exception as e:
#                         print(f"    Error deleting {subfolder_path}: {e}")
#                     # print(f"    Keeping: {subfolder_path}")
    
#     # print(f"\nOperation completed. Deleted {deleted_count} folders.")

# if __name__ == "__main__":
#     # Ask for confirmation before proceeding
#     response = input("This will delete folders that don't contain 'magnified' files. Continue? (y/N): ")
#     if response.lower() == 'y':
#         delete_folders_without_magnified()
#     else:
#         print("Operation cancelled.")


import os
import cv2
from tqdm import tqdm

def get_video_length(video_path):
    """Get the length of a video file in seconds"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        cap.release()
        
        if fps > 0:
            duration = frame_count / fps
            return duration
        else:
            return 0
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return 0

def check_magnified_video_lengths():
    """Check the length of magnified video files in VoxCeleb extracted frames"""
    base_path = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2"
    
    if not os.path.exists(base_path):
        print(f"Base path {base_path} does not exist!")
        return
    
    zero_length_videos = []
    valid_videos = []
    total_videos = 0
    
    print("Checking magnified video lengths...")
    
    # Iterate through each identity folder
    for identity in tqdm(os.listdir(base_path), desc="Processing identities"):
        identity_path = os.path.join(base_path, identity)
        
        if not os.path.isdir(identity_path):
            continue
        
        # Check fake folders
        fake_path = os.path.join(identity_path, 'fake')
        
        if not os.path.exists(fake_path) or not os.path.isdir(fake_path):
            continue
        
        # Check each subfolder in fake
        for subfolder in os.listdir(fake_path):
            subfolder_path = os.path.join(fake_path, subfolder)
            
            if not os.path.isdir(subfolder_path):
                continue
            
            # Look for magnified video files
            for file in os.listdir(subfolder_path):
                if 'magnified' in file.lower() and file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(subfolder_path, file)
                    total_videos += 1
                    
                    # Get video length
                    length = get_video_length(video_path)
                    
                    if length <= 0:
                        zero_length_videos.append({
                            'path': video_path,
                            'identity': identity,
                            'subfolder': subfolder,
                            'file': file,
                            'length': length
                        })
                    else:
                        valid_videos.append({
                            'path': video_path,
                            'identity': identity,
                            'subfolder': subfolder,
                            'file': file,
                            'length': length
                        })
    
    # Print results
    print(f"\n=== MAGNIFIED VIDEO LENGTH ANALYSIS ===")
    print(f"Total magnified videos found: {total_videos}")
    print(f"Videos with length > 0: {len(valid_videos)}")
    print(f"Videos with length = 0: {len(zero_length_videos)}")
    
    if zero_length_videos:
        print(f"\nVideos with zero length ({len(zero_length_videos)}):")
        for video in zero_length_videos:
            print(f"  {video['identity']}/{video['subfolder']}/{video['file']}")
    
    if valid_videos:
        lengths = [v['length'] for v in valid_videos]
        avg_length = sum(lengths) / len(lengths)
        print(f"\nValid videos statistics:")
        print(f"  Average length: {avg_length:.2f} seconds")
        print(f"  Min length: {min(lengths):.2f} seconds")
        print(f"  Max length: {max(lengths):.2f} seconds")
    
    return {
        'total_videos': total_videos,
        'valid_videos': valid_videos,
        'zero_length_videos': zero_length_videos
    }

def delete_zero_length_magnified_videos():
    """Delete magnified video files that have zero length"""
    results = check_magnified_video_lengths()
    
    zero_length_videos = results['zero_length_videos']
    
    if not zero_length_videos:
        print("No zero-length magnified videos found!")
        return
    
    print(f"\nFound {len(zero_length_videos)} zero-length magnified videos.")
    response = input("Do you want to delete these zero-length videos? (y/N): ")
    
    if response.lower() == 'y':
        deleted_count = 0
        for video in zero_length_videos:
            try:
                os.remove(video['path'])
                print(f"Deleted: {video['path']}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {video['path']}: {e}")
        
        print(f"\nDeleted {deleted_count} zero-length magnified videos.")
    else:
        print("Deletion cancelled.")

def check_specific_identity_magnified_videos(identity_name):
    """Check magnified video lengths for a specific identity"""
    base_path = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2"
    identity_path = os.path.join(base_path, identity_name)
    
    if not os.path.exists(identity_path):
        print(f"Identity {identity_name} does not exist!")
        return
    
    fake_path = os.path.join(identity_path, 'fake')
    
    if not os.path.exists(fake_path):
        print(f"No fake folder found for identity {identity_name}")
        return
    
    print(f"Checking magnified videos for identity: {identity_name}")
    
    for subfolder in os.listdir(fake_path):
        subfolder_path = os.path.join(fake_path, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        print(f"\n  Subfolder: {subfolder}")
        
        magnified_found = False
        for file in os.listdir(subfolder_path):
            if 'magnified' in file.lower() and file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                magnified_found = True
                video_path = os.path.join(subfolder_path, file)
                length = get_video_length(video_path)
                print(f"    {file}: {length:.2f} seconds")
        
        if not magnified_found:
            print(f"    No magnified videos found")

if __name__ == "__main__":
    # Check all magnified video lengths
    results = check_magnified_video_lengths()
    
    # Optionally delete zero-length videos
    # if results['zero_length_videos']:
    #     delete_zero_length_magnified_videos()
    
    # Example: Check specific identity
    # check_specific_identity_magnified_videos("id00012")