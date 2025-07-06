import json
import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser(description="Extract faces from videos")
parser.add_argument('--video_root', type=str, required=True, help='Directory containing video files')
parser.add_argument('--output_base_dir', type=str, required=True, help='Base directory to save extracted faces')
args = parser.parse_args()
video_root = args.video_root
audio_output_root = args.output_base_dir

def extract_audio(video_path):
    """Extract audio from video file using ffmpeg."""
    video_path = Path(video_path)
    audio_path = video_path.with_suffix('.wav')
    
    if not video_path.exists():
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
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        return str(audio_path)
    except Exception as e:
        return None


def process_video(fname):
    if fname.endswith('.mp4'):
        video_path = os.path.join(video_root, fname)
        audio_path = extract_audio(video_path)
        if audio_path:
            video_name = os.path.splitext(fname)[0]
            target_dir = os.path.join(audio_output_root, video_name)
            os.makedirs(target_dir, exist_ok=True)
            target_audio_path = os.path.join(target_dir, f"{video_name}.wav")
            if audio_path != target_audio_path:
                os.rename(audio_path, target_audio_path)
        else:
            print(f"Failed to extract audio for {video_path}")

if __name__ == "__main__":
    video_files = [f for f in os.listdir(video_root) if f.endswith('.mp4')]
    num_workers = min(16,os.cpu_count())
    with ThreadPoolExecutor(max_workers=num_workers) as executor:  # Adjust max_workers as needed
        futures = [executor.submit(process_video, fname) for fname in video_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Extracting audio"):
            pass