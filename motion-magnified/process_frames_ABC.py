"""
Process all extracted frames in the dataset to create frameA, frameB, and frameC directories.
This script will process both real and fake videos for each identity in parallel.
"""
import os
import sys
from tqdm import tqdm
import shutil
import concurrent.futures
from functools import partial

def process_frames(directory, image_format='jpg'):
    """
    Process frames in the given directory to create frameA, frameB, and frameC
    Args:
        directory (str): Path to directory containing frames
        image_format (str): Image format (default: jpg)
    """
    
    # Create frame directories
    os.makedirs(os.path.join(directory, 'frameA'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'frameC'), exist_ok=True)
    
    # Get sorted list of frame files
    files = sorted([f for f in os.listdir(directory) 
                   if os.path.splitext(f)[1] == f'.{image_format}'],
                  key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not files:
        return
    
    # Copy files to frameA and frameC
    for f in files:
        src = os.path.join(directory, f)
        dst_a = os.path.join(directory, 'frameA', f)
        dst_c = os.path.join(directory, 'frameC', f)
        shutil.copy2(src, dst_a)  # Using shutil.copy2 instead of os.system
        shutil.copy2(src, dst_c)
    
    # Remove last frame from frameA and first frame from frameC
    os.remove(os.path.join(directory, 'frameA', files[-1]))
    os.remove(os.path.join(directory, 'frameC', files[0]))
    
    # Rename frames in frameC to match frameB
    for f in sorted(os.listdir(os.path.join(directory, 'frameC')), 
                   key=lambda x: int(x.split('_')[-1].split('.')[0])):
        old_path = os.path.join(directory, 'frameC', f)
        frame_num = int(f.split('_')[-1].split('.')[0])
        new_name = f'frame_{frame_num-1:05d}.{image_format}'
        new_path = os.path.join(directory, 'frameC', new_name)
        os.rename(old_path, new_path)
    
    # Create frameB as copy of frameC
    shutil.copytree(os.path.join(directory, 'frameC'), os.path.join(directory, 'frameB'))

def has_magnified_file(folder_path):
    """Check if folder contains any file with 'magnified' in its name"""
    try:
        for file in os.listdir(folder_path):
            if 'magnified' in file.lower():
                return True
        return False
    except OSError:
        return False

def process_single_video_dir(video_path):
    """Process a single video directory - wrapper for parallel processing"""
    try:
        process_frames(video_path)
        return {'success': True, 'path': video_path}
    except Exception as e:
        return {'success': False, 'path': video_path, 'error': str(e)}

def collect_video_directories(base_dir):
    """Collect all video directories that need processing"""
    video_dirs = []
    
    for identity in os.listdir(base_dir):
        identity_path = os.path.join(base_dir, identity)
        if not os.path.isdir(identity_path):
            continue
        
        # Process fake videos
        fake_path = os.path.join(identity_path, 'fake')
        if os.path.exists(fake_path):
            for video_dir in os.listdir(fake_path):
                video_path = os.path.join(fake_path, video_dir)
                if (os.path.isdir(video_path) and 
                    not has_magnified_file(video_path) and 
                    not os.path.exists(os.path.join(video_path, 'frameA'))):
                    video_dirs.append(video_path)
        
        # Uncomment to process real videos as well
        # real_path = os.path.join(identity_path, 'real')
        # if os.path.exists(real_path):
        #     for video_dir in os.listdir(real_path):
        #         video_path = os.path.join(real_path, video_dir)
        #         if (os.path.isdir(video_path) and 
        #             not os.path.exists(os.path.join(video_path, 'frameA')) and 
        #             not os.path.exists(os.path.join(video_path, 'frameB')) and 
        #             not os.path.exists(os.path.join(video_path, 'frameC'))):
        #             video_dirs.append(video_path)
    
    return video_dirs

def process_dataset_parallel(base_dir, num_workers=None):
    """
    Process all frames in the dataset using parallel processing
    Args:
        base_dir (str): Base directory containing the extracted frames
        num_workers (int): Number of worker threads (default: CPU count)
    """
    if num_workers is None:
        num_workers = min(os.cpu_count(), 16)  # Use up to 16 workers
    
    print(f"Using {num_workers} worker threads")
    
    # Collect all video directories that need processing
    print("Collecting video directories to process...")
    video_dirs = collect_video_directories(base_dir)
    
    if not video_dirs:
        print("No video directories found to process!")
        return
    
    print(f"Found {len(video_dirs)} video directories to process")
    
    # Process directories in parallel
    successful_count = 0
    failed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks and get results with progress bar
        results = list(tqdm(
            executor.map(process_single_video_dir, video_dirs),
            total=len(video_dirs),
            desc="Processing frame directories"
        ))
    
    # Process results
    for result in results:
        if result['success']:
            successful_count += 1
        else:
            failed_count += 1
            print(f"Failed to process {result['path']}: {result['error']}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful_count} directories")
    print(f"Failed: {failed_count} directories")

def delete_frame_abc_folders(base_dir):
    """
    Recursively delete all frameA, frameB, and frameC folders under base_dir.
    Args:
        base_dir (str): Base directory to search for frame folders
    """
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    deleted_count = 0
    frame_folders = ['frameA', 'frameB', 'frameC']
    
    print(f"Searching for frame folders in {base_dir}...")
    
    for root, dirs, files in tqdm(os.walk(base_dir), desc="Scanning directories"):
        # Create a copy of dirs to iterate over, since we'll be modifying the original
        dirs_to_check = dirs.copy()
        
        for folder_name in dirs_to_check:
            if folder_name in frame_folders:
                folder_path = os.path.join(root, folder_name)
                try:
                    shutil.rmtree(folder_path)
                    deleted_count += 1
                    print(f"Deleted: {folder_path}")
                    # Remove from dirs list to prevent os.walk from descending into it
                    dirs.remove(folder_name)
                except Exception as e:
                    print(f"Error deleting {folder_path}: {e}")
    
    print(f"\nDeletion complete! Deleted {deleted_count} frame folders.")

def delete_frame_abc_folders_confirm(base_dir):
    """
    Delete frameA, frameB, and frameC folders with user confirmation.
    Args:
        base_dir (str): Base directory to search for frame folders
    """
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # First, count how many folders will be deleted
    count = 0
    frame_folders = ['frameA', 'frameB', 'frameC']
    
    print(f"Scanning {base_dir} for frame folders...")
    for root, dirs, files in os.walk(base_dir):
        for folder_name in dirs:
            if folder_name in frame_folders:
                count += 1
    
    if count == 0:
        print("No frameA, frameB, or frameC folders found.")
        return
    
    print(f"Found {count} frame folders to delete.")
    response = input("Do you want to proceed with deletion? (y/N): ")
    
    if response.lower() != 'y':
        print("Deletion cancelled.")
        return
    
    # Proceed with deletion
    delete_frame_abc_folders(base_dir)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python process_frames_ABC.py <extracted_frames_directory> [action] [num_workers]")
        print("Actions: process (default), delete, delete-confirm")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        sys.exit(1)
    
    # Check for action parameter
    action = 'process'
    if len(sys.argv) >= 3:
        action = sys.argv[2].lower()
    
    if action == 'delete':
        delete_frame_abc_folders(base_dir)
    elif action == 'delete-confirm':
        delete_frame_abc_folders_confirm(base_dir)
    else:
        # Default: process frames
        num_workers = 100
        if len(sys.argv) >= 4:
            try:
                num_workers = int(sys.argv[3])
            except ValueError:
                print("Warning: Invalid number of workers specified, using default")
        
        process_dataset_parallel(base_dir, num_workers)