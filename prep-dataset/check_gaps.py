import shutil
import os
import re
from tqdm import tqdm
import argparse


def get_frame_indexes(folder):
    indexes = []
    for fname in os.listdir(folder):
        match = re.match(r"frame_(\d{5})\.jpg", fname)
        if match:
            indexes.append(int(match.group(1)))
    return sorted(indexes)

def find_missing_indexes(indexes):
    if not indexes:
        return []
    expected = set(range(indexes[0], indexes[-1] + 1))
    actual = set(indexes)
    return sorted(expected - actual)

def remove_gaps_in_folder(folder):
    # Get all frame files and their indexes
    frame_files = []
    for fname in os.listdir(folder):
        match = re.match(r"frame_(\d{5})\.jpg", fname)
        if match:
            frame_files.append((int(match.group(1)), fname))
    if not frame_files:
        return
    # Sort by index
    frame_files.sort()
    # Renumber frames consecutively
    for new_idx, (old_idx, fname) in enumerate(frame_files):
        new_fname = f"frame_{new_idx:05d}.jpg"
        old_path = os.path.join(folder, fname)
        new_path = os.path.join(folder, new_fname)
        if old_path != new_path:
            os.rename(old_path, new_path)

def is_video_folder(folder_path):
    return os.path.isdir(folder_path) and os.path.basename(folder_path).isdigit()

def find_and_fix_folders_with_gaps(base_dir):
    ctr1, ctr2 = 0, 0
    digit_folders = [folder for folder in os.listdir(base_dir) if is_video_folder(os.path.join(base_dir, folder))]
    digit_folders = sorted(digit_folders, key=lambda x: int(x))
    for folder in tqdm(digit_folders):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            has_frame = any(
                re.match(r"frame_(\d{5})\.jpg", fname)
                for fname in os.listdir(folder_path)
            )
            if has_frame:
                indexes = get_frame_indexes(folder_path)
                missing = find_missing_indexes(indexes)
                ctr1 += 1
                if missing:
                    remove_gaps_in_folder(folder_path)

def find_folders_with_no_frames(base_dir):
    """
    Returns a list of folder names (full paths) among the first 400,000 digit-named folders
    that do not contain any frame_XXXXX.jpg files.
    """
    no_frame_folders = []
    digit_folders = [folder for folder in os.listdir(base_dir) if folder.isdigit()]
    digit_folders = sorted(digit_folders, key=lambda x: int(x))
    for folder in digit_folders:
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            has_frame = any(
                re.match(r"frame_(\d{5})\.jpg", fname)
                for fname in os.listdir(folder_path)
            )
            if not has_frame:
                no_frame_folders.append(folder_path)
    return no_frame_folders

def count_folders_with_magnified(base_dir):
    count = 0
    digit_folders = [folder for folder in os.listdir(base_dir) if folder.isdigit()]
    for folder in tqdm(digit_folders):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            if any('magnified' in fname.lower() for fname in os.listdir(folder_path)):
                count += 1
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove frame gaps")
    parser.add_argument('--output_base_dir', type=str, required=True, help='Base directory to save extracted faces')
    args = parser.parse_args()
    base_dir = args.output_base_dir
    find_and_fix_folders_with_gaps(base_dir)