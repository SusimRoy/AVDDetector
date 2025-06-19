import os

base_dir = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/silent_videos"
missing = []

for root, dirs, files in os.walk(base_dir):
    # Only check leaf directories (those that might contain frameA/B/C)
    if any(sub in dirs for sub in ['frameA', 'frameB', 'frameC']):
        for sub in ['frameA', 'frameB', 'frameC']:
            if sub not in dirs:
                missing.append(root)
                break

# Remove duplicates and print
for folder in sorted(set(missing)):
    print(folder)