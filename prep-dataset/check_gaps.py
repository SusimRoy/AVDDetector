import os
import re
import shutil
from collections import defaultdict

def get_frame_indexes(folder):
    indexes = []
    for fname in os.listdir(folder):
        match = re.search(r"(\d+)", fname)
        if match:
            indexes.append(int(match.group(1)))
    return sorted(indexes)

def has_magnified_file(folder_path):
    """Check if folder contains any file with 'magnified' in its name"""
    try:
        for file in os.listdir(folder_path):
            if 'magnified' in file.lower():
                return True
        return False
    except OSError:
        return False

def find_missing_indexes(indexes):
    if not indexes:
        return []
    expected = set(range(indexes[0], indexes[-1] + 1))
    actual = set(indexes)
    return sorted(expected - actual)

root_dir = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2"
result = defaultdict(lambda: {'fake': 0, 'real': 0})

for identity in os.listdir(root_dir):
    identity_path = os.path.join(root_dir, identity)
    if not os.path.isdir(identity_path):
        continue
    for label in ['fake']:
        label_path = os.path.join(identity_path, label)
        if not os.path.isdir(label_path):
            continue
        for subfolder in os.listdir(label_path):
            subfolder_path = os.path.join(label_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            if has_magnified_file(subfolder_path):
                continue

            indexes = get_frame_indexes(subfolder_path)
            missing = find_missing_indexes(indexes)
            if missing:
                result[identity][label] += 1
                # print(f"Deleting {subfolder_path} (missing indexes: {missing})")
                # shutil.rmtree(subfolder_path)  # This deletes the folder and all its contents

# Print the summary
for identity in result:
    print(f"Identity: {identity}")
    for label in ['fake', 'real']:
        print(f"  {label}: {result[identity][label]} subfolders deleted (had gaps)")

print(len(result))


# import os

# root_dir = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2"

# only_real = []
# only_fake = []
# both = []

# for identity in os.listdir(root_dir):
#     identity_path = os.path.join(root_dir, identity)
#     if not os.path.isdir(identity_path):
#         continue

#     has_real = os.path.isdir(os.path.join(identity_path, 'real'))
#     has_fake = os.path.isdir(os.path.join(identity_path, 'fake'))

#     if has_real and not has_fake:
#         only_real.append(identity)
#     elif has_fake and not has_real:
#         only_fake.append(identity)
#     elif has_real and has_fake:
#         both.append(identity)

# print(f"Identities with only 'real' folder: {len(only_real)}")
# print(f"Identities with only 'fake' folder: {len(only_fake)}")
# print(f"Identities with both 'real' and 'fake' folders: {len(both)}") 

# If you want to see which identities are in each group, uncomment below:
# print("Only real:", only_real)
# print("Only fake:", only_fake)
# print("Both:", both)


# import os

# root_dir = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/silent_videos"

# for identity in os.listdir(root_dir):
#     identity_path = os.path.join(root_dir, identity)
#     if not os.path.isdir(identity_path):
#         continue
#     for label in ['real', 'fake']:
#         label_path = os.path.join(identity_path, label)
#         if not os.path.isdir(label_path):
#             continue
#         for subfolder in os.listdir(label_path):
#             subfolder_path = os.path.join(label_path, subfolder)
#             if not os.path.isdir(subfolder_path):
#                 continue
#             for fname in os.listdir(subfolder_path):
#                 if fname.lower().endswith('.jpg'):
#                     file_path = os.path.join(subfolder_path, fname)
#                     print(f"Deleting {file_path}")
#                     os.remove(file_path)

# import os

# root_dir = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2"
# missing = []
# deleted = []

# for identity in os.listdir(root_dir):
#     identity_path = os.path.join(root_dir, identity)
#     if not os.path.isdir(identity_path):
#         continue

#     for label in ['real', 'fake']:
#         label_path = os.path.join(identity_path, label)
#         if not os.path.isdir(label_path):
#             continue
#         for subfolder in os.listdir(label_path):
#             subfolder_path = os.path.join(label_path, subfolder)
#             if not os.path.isdir(subfolder_path):
#                 continue
#             # find if magnified video exists
#             magnified_video = os.path.join(subfolder_path, f"{subfolder}_magnified.mp4")
#             if os.path.exists(magnified_video) and not os.path.exists(os.path.join(subfolder_path, 'frameA')) and not os.path.exists(os.path.join(subfolder_path, 'frameB')) and not os.path.exists(os.path.join(subfolder_path, 'frameC')):
#                 missing.append(subfolder_path)
#                 continue
            
#             # if it does, delete the subfolder
#             # shutil.rmtree(subfolder_path)
#             # Check for all three frame folders
#             # frameA = os.path.isdir(os.path.join(subfolder_path, 'frameA'))
#             # frameB = os.path.isdir(os.path.join(subfolder_path, 'frameB'))
#             # frameC = os.path.isdir(os.path.join(subfolder_path, 'frameC'))
#             # if not (frameA and frameB and frameC):
#             #     missing.append(subfolder_path)

#             # Find and delete all *_magnified.mp4 files
#             for fname in os.listdir(subfolder_path):
#                 if fname.endswith('_magnified.mp4'):
#                     file_path = os.path.join(subfolder_path, fname)
#                     print(f"Deleting {file_path}")
#                     os.remove(file_path)
#                     deleted.append(file_path)

# print("Subfolders missing at least one of frameA, frameB, or frameC:")
# for path in missing:
#     print(path)
# print(f"Total: {len(missing)}")

# print(f"Total magnified videos deleted: {len(deleted)}")


# import os
# import shutil

# root_dir = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2"
# deleted = []

# for identity in os.listdir(root_dir):
#     identity_path = os.path.join(root_dir, identity)
#     if not os.path.isdir(identity_path):
#         continue

#     for label in ['real', 'fake']:
#         label_path = os.path.join(identity_path, label)
#         if not os.path.isdir(label_path):
#             continue
#         for subfolder in os.listdir(label_path):
#             subfolder_path = os.path.join(label_path, subfolder)
#             if not os.path.isdir(subfolder_path):
#                 continue
#             for frame_folder in ['frameA', 'frameB', 'frameC']:
#                 frame_path = os.path.join(subfolder_path, frame_folder)
#                 if os.path.isdir(frame_path):
#                     print(f"Deleting {frame_path}")
#                     shutil.rmtree(frame_path)
#                     deleted.append(frame_path)

# print(f"Total frame folders deleted: {len(deleted)}")