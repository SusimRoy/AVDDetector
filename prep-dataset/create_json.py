# # import os
# # import json
# # from tqdm import tqdm

# # def update_json_with_magnified_path(base_dir):
# #     for identity in tqdm(os.listdir(base_dir)):
# #         identity_path = os.path.join(base_dir, identity)
# #         if not os.path.isdir(identity_path):
# #             continue

# #         for realfake in ['fake']:
# #             realfake_path = os.path.join(identity_path, realfake)
# #             if not os.path.isdir(realfake_path):
# #                 continue

# #             for subfolder in os.listdir(realfake_path):
# #                 subfolder_path = os.path.join(realfake_path, subfolder)
# #                 if not os.path.isdir(subfolder_path):
# #                     continue

# #                 # Find all json files in this subfolder
# #                 for file in os.listdir(subfolder_path):
# #                     if file.endswith('.json'):
# #                         json_path = os.path.join(subfolder_path, file)
# #                         magnified_video = f"{subfolder}_magnified.mp4"
# #                         magnified_path = os.path.join(subfolder_path, magnified_video)

# #                         # Load, update, and save the JSON
# #                         with open(json_path, 'r') as f:
# #                             data = json.load(f)
# #                         data['magnifiedfile'] = magnified_path
# #                         with open(json_path, 'w') as f:
# #                             json.dump(data, f, indent=2)
# #                         # print(f"Updated {json_path} with magnifiedfile: {magnified_path}")

# # if __name__ == "__main__":
# #     base_dir = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2"
# #     update_json_with_magnified_path(base_dir)

# import os
# import json

# base_dir = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2"
# output_json = os.path.join(base_dir, "train_metadata_new.json")

# all_entries = []

# for identity in os.listdir(base_dir):
#     identity_path = os.path.join(base_dir, identity)
#     if not os.path.isdir(identity_path):
#         continue
#     for realfake in ["real", "fake"]:
#         realfake_path = os.path.join(identity_path, realfake)
#         if not os.path.isdir(realfake_path):
#             continue
#         for subfolder in os.listdir(realfake_path):
#             subfolder_path = os.path.join(realfake_path, subfolder)
#             if not os.path.isdir(subfolder_path):
#                 continue
#             # Find the json file (real.json or fake.json)
#             for fname in os.listdir(subfolder_path):
#                 if fname.endswith(".json"):
#                     json_path = os.path.join(subfolder_path, fname)
#                     with open(json_path, "r") as f:
#                         data = json.load(f)
#                         # Optionally, add identity/realfake/subfolder info
#                         # data["_identity"] = identity
#                         # data["_type"] = realfake
#                         # data["_subfolder"] = subfolder
#                         all_entries.append(data)

# with open(output_json, "w") as f:
#     json.dump(all_entries, f, indent=2)

# print(f"Combined {len(all_entries)} JSON entries into {output_json}")


# import json

# # Load existing data from destination (if it exists)
# try:
#     with open('/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/train_metadata.json', 'r') as dest_file:
#         dest_data = json.load(dest_file)
# except FileNotFoundError:
#     dest_data = {}

# # Load source data
# with open('/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2/train_metadata.json', 'r') as source_file:
#     source_data = json.load(source_file)

# # Merge source_data into dest_data (assuming they're dictionaries)
# dest_data.extend(source_data)

# # Write merged data back to destination.json
# with open('/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/train_metadata.json', 'w') as dest_file:
#     json.dump(dest_data, dest_file, indent=4)



# import json

# import json

# def find_top_n_fake_segments(json_path, n=50):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
    
#     # Filter entries with non-empty fake_segments
#     fake_entries = [entry for entry in data if len(entry.get('fake_segments', [])) > 0]
#     print(f"Total entries: {len(data)}")
#     print(f"Entries with fake_segments > 0: {len(fake_entries)}")
    
#     # Sort filtered entries by video_frames in descending order
#     sorted_entries = sorted(
#         fake_entries, 
#         key=lambda entry: entry.get("video_frames", 0), 
#         reverse=True
#     )
    
#     print(f"Top {n} videos with fake segments, sorted by number of frames:")
#     for i, entry in enumerate(sorted_entries[:n]):
#         print(f"{i+1}. {entry.get('file', 'Unknown')} - {entry.get('video_frames', 0)} frames - {len(entry.get('fake_segments', []))} fake segments")
    
#     return sorted_entries[:n]

# # Usage
# json_path = "/home/csgrad/susimmuk/acmdeepfake/data/AV-Deepfake1M-PlusPlus/train_metadata.json"
# find_top_n_fake_segments(json_path, n=200)


import json 

def find_top_n_fake_segments(json_path, n=50, sort_by_last_fake=True):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter entries with non-empty fake_segments
    fake_entries = [entry for entry in data if len(entry.get('fake_segments', [])) > 0]
    print(f"Total entries: {len(data)}")
    print(f"Entries with fake_segments > 0: {len(fake_entries)}")
    
    # Calculate last fake timestamp for each entry
    for entry in fake_entries:
        fake_segments = entry.get('fake_segments', [])
        last_fake_timestep = 0
        
        for segment in fake_segments:
            if isinstance(segment, dict) and 'end' in segment:
                last_fake_timestep = max(last_fake_timestep, segment['end'])
            elif isinstance(segment, list) and len(segment) >= 2:
                last_fake_timestep = max(last_fake_timestep, segment[1])
        
        # Store the calculated value in the entry
        entry['last_fake_timestamp'] = last_fake_timestep
    
    # Sort entries by last fake timestamp if requested, otherwise by video_frames
    if sort_by_last_fake:
        sorted_entries = sorted(
            fake_entries, 
            key=lambda entry: entry.get('last_fake_timestamp', 0),
            reverse=True
        )
        sort_criteria = "last fake timestamp"
    else:
        sorted_entries = sorted(
            fake_entries, 
            key=lambda entry: entry.get('video_frames', 0),
            reverse=True
        )
        sort_criteria = "number of frames"
    
    print(f"Top {n} videos with fake segments, sorted by {sort_criteria}:")
    for i, entry in enumerate(sorted_entries[:n]):
        last_fake_timestep = entry.get('last_fake_timestamp', 0)
        frames = entry.get('video_frames', 0)
        fake_segments = entry.get('fake_segments', [])
        
        # Calculate percentage of video where fake content appears
        percentage = (last_fake_timestep / frames * 100) if frames > 0 else 0
        
        print(f"{i+1}. {entry.get('file', 'Unknown')} - {frames} frames - "
              f"{len(fake_segments)} fake segments - Last fake at: {last_fake_timestep} "
              f"({percentage:.1f}% of video)")
    
    return sorted_entries[:n]

# Usage
json_path = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/train_metadata_with_audio.json"
find_top_n_fake_segments(json_path, n=50, sort_by_last_fake=True)

# import json 

# def find_top_n_fake_segments(json_path, n=50):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
    
#     # Filter entries with non-empty fake_segments
#     fake_entries = [entry for entry in data if len(entry.get('fake_segments', [])) > 0]
#     print(f"Total entries: {len(data)}")
#     print(f"Entries with fake_segments > 0: {len(fake_entries)}")
    
#     # Sort filtered entries by video_frames in descending order
#     sorted_entries = sorted(
#         fake_entries, 
#         key=lambda entry: entry.get("video_frames", 0), 
#         reverse=True
#     )
    
#     print(f"Top {n} videos with fake segments, sorted by number of frames:")
#     for i, entry in enumerate(sorted_entries[:n]):
#         # Find the last timestep where a fake segment occurs
#         fake_segments = entry.get('fake_segments', [])
#         last_fake_timestep = 0
        
#         for segment in fake_segments:
#             # Assuming each segment has 'end' attribute
#             # If it has a different structure, this needs to be adjusted
#             if isinstance(segment, dict) and 'end' in segment:
#                 last_fake_timestep = max(last_fake_timestep, segment['end'])
#             elif isinstance(segment, list) and len(segment) >= 2:
#                 last_fake_timestep = max(last_fake_timestep, segment[1])  # Assuming [start, end] format
        
#         print(f"{i+1}. {entry.get('file', 'Unknown')} - {entry.get('video_frames', 0)} frames - "
#               f"{len(fake_segments)} fake segments - Last fake at: {last_fake_timestep}")
    
#     return sorted_entries[:n]

# # Usage
# json_path = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/train_metadata_with_audio.json"
# find_top_n_fake_segments(json_path, n=50)
 
# import json

# def count_nonempty_fake_segments(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     count = sum(1 for entry in data if len(entry.get('fake_segments', [])) > 0)
#     print(f"Number of entries with len(fake_segments) > 0: {count}")
#     return count

# # Example usage:
# json_path = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2/train_metadata_new_with_audio.json"
# json_path = "/home/csgrad/susimmuk/acmdeepfake/data/AV-Deepfake1M-PlusPlus/vox_celeb_2/train_metadata_new.json"
# # i want to divide the count by the total number of entries in the json file
# with open(json_path, 'r') as f:
#     data = json.load(f)
# print(len(data))
# print((len(data) - count_nonempty_fake_segments(json_path)))
# print(count_nonempty_fake_segments(json_path))

##New Dataset
# Total 16917
# Real 8880
# Fake 8037
##Old Dataset
# Real 297509
# Fake 801708


# import json
# import shutil

# def remove_vox_celeb_2_entries_from_file_attr(json_path, output_path=None, create_backup=True):
#     """
#     Remove entries from JSON file that contain 'vox_celeb_2' in the 'file' attribute.
#     """
    
#     # Create backup if requested
#     if create_backup:
#         backup_path = json_path + '.backup'
#         shutil.copy2(json_path, backup_path)
#         print(f"Backup created: {backup_path}")
    
#     # Load the JSON data
#     with open(json_path, 'r') as f:
#         data = json.load(f)
    
#     original_count = len(data)
    
#     # Filter out entries containing 'vox_celeb_2' in the 'file' attribute
#     filtered_data = []
#     removed_count = 0
    
#     for entry in data:
#         should_remove = False
        
#         if isinstance(entry, dict) and 'file' in entry:
#             if isinstance(entry['file'], str) and 'vox_celeb_2' in entry['file']:
#                 should_remove = True
        
#         if should_remove:
#             removed_count += 1
#         else:
#             filtered_data.append(entry)
    
#     remaining_count = len(filtered_data)
    
#     # Save the filtered data
#     if output_path is None:
#         output_path = json_path
    
#     with open(output_path, 'w') as f:
#         json.dump(filtered_data, f, indent=2)
    
#     print(f"Original entries: {original_count}")
#     print(f"Removed entries with 'vox_celeb_2' in 'file' attribute: {removed_count}")
#     print(f"Remaining entries: {remaining_count}")
    
#     return filtered_data

# def count_nonempty_fake_segments(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     count = sum(1 for entry in data if len(entry.get('fake_segments', [])) > 0)
#     print(f"Number of entries with len(fake_segments) > 0: {count}")
#     return count

# def merge_json_files(main_json_path, additional_json_path, output_path=None):
#     """
#     Merge two JSON files by appending entries from additional_json_path to main_json_path
#     """
#     # Load main JSON file
#     with open(main_json_path, 'r') as f:
#         main_data = json.load(f)
    
#     # Load additional JSON file
#     with open(additional_json_path, 'r') as f:
#         additional_data = json.load(f)
    
#     # Merge the data
#     merged_data = main_data + additional_data
    
#     # Save merged data
#     if output_path is None:
#         output_path = main_json_path
    
#     with open(output_path, 'w') as f:
#         json.dump(merged_data, f, indent=2)
    
#     print(f"Merged {len(main_data)} + {len(additional_data)} = {len(merged_data)} entries")
#     print(f"Saved to: {output_path}")
    
#     return merged_data

# def main():
#     # File paths
#     main_json_path = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/train_metadata_with_audio.json"
#     additional_json_path = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/train_metadata_new_with_audio.json"
    
#     print("=== BEFORE PROCESSING ===")
    
#     # Show stats for main file before processing
#     print(f"\nMain file: {main_json_path}")
#     with open(main_json_path, 'r') as f:
#         main_data = json.load(f)
#     print(f"Total entries: {len(main_data)}")
#     main_fake_count = count_nonempty_fake_segments(main_json_path)
#     print(f"Real entries: {len(main_data) - main_fake_count}")
#     print(f"Fake entries: {main_fake_count}")
    
#     # Show stats for additional file
#     print(f"\nAdditional file: {additional_json_path}")
#     with open(additional_json_path, 'r') as f:
#         additional_data = json.load(f)
#     print(f"Total entries: {len(additional_data)}")
#     additional_fake_count = count_nonempty_fake_segments(additional_json_path)
#     print(f"Real entries: {len(additional_data) - additional_fake_count}")
#     print(f"Fake entries: {additional_fake_count}")
    
#     print("\n=== PROCESSING ===")
    
#     # Step 1: Remove vox_celeb_2 entries from main file
#     print("\nStep 1: Removing vox_celeb_2 entries from main file...")
#     filtered_main_data = remove_vox_celeb_2_entries_from_file_attr(main_json_path)
    
#     # Step 2: Merge with additional file
#     print("\nStep 2: Merging with additional file...")
#     merged_data = merge_json_files(main_json_path, additional_json_path)
    
#     print("\n=== AFTER PROCESSING ===")
    
#     # Show final stats
#     print(f"\nFinal merged file: {main_json_path}")
#     print(f"Total entries: {len(merged_data)}")
#     final_fake_count = sum(1 for entry in merged_data if len(entry.get('fake_segments', [])) > 0)
#     print(f"Real entries: {len(merged_data) - final_fake_count}")
#     print(f"Fake entries: {final_fake_count}")
    
#     print(f"\nNumber of entries with len(fake_segments) > 0: {final_fake_count}")

# if __name__ == "__main__":
#     main()