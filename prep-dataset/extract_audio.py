import json
import os
import subprocess
from pathlib import Path
from tqdm import tqdm


def extract_audio(video_path):
    """Extract audio from video file using ffmpeg."""
    video_path = Path(video_path)
    audio_path = video_path.with_suffix('.wav')
    
    # Skip if audio file already exists
    if audio_path.exists():
        return str(audio_path)
    
    # Check if video exists
    if not video_path.exists():
        # print(f"Warning: Video file does not exist: {video_path}")
        return None
    
    # First check if video has audio stream
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        has_audio = bool(result.stdout.strip())
        
        if not has_audio:
            print(f"Warning: No audio stream found in {video_path}")
            # Create an empty audio file with the correct format
            cmd = [
                'ffmpeg',
                '-f', 'lavfi',  # Use lavfi input
                '-i', 'anullsrc=r=16000:cl=mono',  # Generate silent audio
                '-t', '1',  # Duration of 1 second
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                str(audio_path)
            ]
        else:
            # Normal audio extraction
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                str(audio_path)
            ]
        
        # Run ffmpeg command with stderr output
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # print(f"Error processing {video_path}:")
            # print(f"FFmpeg error: {result.stderr}")
            return None
        return str(audio_path)
    except Exception as e:
        # print(f"Error processing {video_path}: {str(e)}")
        return None

def process_metadata(json_path):
    """Process all videos in metadata and update JSON with audio paths."""
    # Read metadata
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # Process each entry
    total = len(metadata)
    success = 0
    failed = 0
    
    for i, entry in tqdm(enumerate(metadata, 1)):
        video_path = entry.get('file')
        if entry.get('audiofile'):
            continue
        vpath = os.path.join("/home/csgrad/susimmuk/acmdeepfake/data/AV-Deepfake1M-PlusPlus/train", video_path)
        if not vpath:
            continue
            
        # print(f"\nProcessing {i}/{total}: {vpath}")
        
        # Extract audio
        audio_path = extract_audio(vpath)
        if audio_path:
            entry['audiofile'] = audio_path
            success += 1
            print(f"Success: {vpath} -> {audio_path}")
        else:
            failed += 1
            print(f"Failed to process: {vpath}")
    
    # Save updated metadata
    output_path = json_path.replace('.json', '_with_audio.json')
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nProcessing complete:")
    print(f"Total files: {total}")
    print(f"Successful: {success}")
    print(f"Failed: {failed}") 
    print(f"Updated metadata saved to: {output_path}")

if __name__ == "__main__":
    json_path = "/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2/train_metadata_new.json"
    process_metadata(json_path) 