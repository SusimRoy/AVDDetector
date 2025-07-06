# def process_frames(directory, image_format='jpg'):
#     """
#     Process frames in the given directory to create frameA, frameB, and frameC
#     Args:
#         directory (str): Path to directory containing frames
#         image_format (str): Image format (default: jpg)
#     """
    
#     # Create frame directories
#     os.makedirs(os.path.join(directory, 'frameA'), exist_ok=True)
#     os.makedirs(os.path.join(directory, 'frameC'), exist_ok=True)
    
#     # Get sorted list of frame files
#     files = sorted([f for f in os.listdir(directory) 
#                    if os.path.splitext(f)[1] == f'.{image_format}'],
#                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
#     if not files:
#         return
    
#     # Copy files to frameA and frameC
#     for f in files:
#         src = os.path.join(directory, f)
#         dst_a = os.path.join(directory, 'frameA', f)
#         dst_c = os.path.join(directory, 'frameC', f)
#         shutil.copy2(src, dst_a)  # Using shutil.copy2 instead of os.system
#         shutil.copy2(src, dst_c)
    
#     # Remove last frame from frameA and first frame from frameC
#     os.remove(os.path.join(directory, 'frameA', files[-1]))
#     os.remove(os.path.join(directory, 'frameC', files[0]))
    
#     # Rename frames in frameC to match frameB
#     for f in sorted(os.listdir(os.path.join(directory, 'frameC')), 
#                    key=lambda x: int(x.split('_')[-1].split('.')[0])):
#         old_path = os.path.join(directory, 'frameC', f)
#         frame_num = int(f.split('_')[-1].split('.')[0])
#         new_name = f'frame_{frame_num-1:05d}.{image_format}'
#         new_path = os.path.join(directory, 'frameC', new_name)
#         os.rename(old_path, new_path)
    
#     # Create frameB as copy of frameC
#     shutil.copytree(os.path.join(directory, 'frameC'), os.path.join(directory, 'frameB'))

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from tqdm import tqdm
import shutil
import argparse
import concurrent.futures
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_frames(directory, image_format='jpg'):
    """
    Process frames in the given directory to create frameA, frameB, and frameC
    Args:
        directory (str): Path to directory containing frames
        image_format (str): Image format (default: jpg)
    """

    files = sorted([f for f in os.listdir(directory) 
                   if os.path.splitext(f)[1] == f'.{image_format}'],
                  key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not files:
        return
    
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

def is_video_folder(folder_path):
    return os.path.isdir(folder_path) and os.path.basename(folder_path).isdigit()

def main():
    parser = argparse.ArgumentParser(description="Process video frame folders to create frameA, frameB, frameC.")
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing video folders')
    args = parser.parse_args()

    base_dir = args.base_dir
    video_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if is_video_folder(os.path.join(base_dir, d))
    ]
    video_dirs.sort()
    max_workers = os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_frames, fname) for fname in video_dirs]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            pass

if __name__ == "__main__":
    main()