import torch
import audiossl.methods.atstframe.embedding as embedding
embedding.N_BLOCKS=1
from audiossl.methods.atstframe.embedding import load_model,get_scene_embedding,get_timestamp_embedding
# import torchvision
# import numpy as np
# import librosa
# import librosa.display

model = load_model("/home/csgrad/susimmuk/acmdeepfake/audio-extraction/atstframe_base.ckpt")

# # Set parameters to read more of the video
# video_path = "/home/csgrad/susimmuk/acmdeepfake/data/AV-Deepfake1M-PlusPlus/train/lrs3/ukgnU2aXM2c/00004/00004_p4.mp4"
# audio_path = "/home/csgrad/susimmuk/acmdeepfake/data/AV-Deepfake1M-PlusPlus/train/vox_celeb_2/id00830/s1ewU7d_-ls/00147/fake_video_fake_audio.wav"
# # Option 1: Read the entire video (no time limits)
# video, audio, info = torchvision.io.read_video(video_path, pts_unit='sec')

# # Option 2 (alternative): Read a specific segment
# # video, audio, info = torchvision.io.read_video(video_path, start_pts=0, end_pts=10, pts_unit='sec')

# audio = audio.float()
# sample_rate = info["audio_fps"]  # Get actual sample rate from metadata

# # Calculate and print duration information
# duration_seconds = len(audio[0]) / sample_rate
# print(f"Video shape: {video.shape}, Audio shape: {audio.shape}")
# print(f"Audio sample rate: {sample_rate} Hz")
# print(f"Audio duration: {duration_seconds:.2f} seconds")
# print(f"Number of frames: {len(video)} at {info.get('video_fps', 'unknown')} fps")

# emb_scene = get_scene_embedding(audio,model)

# # Enhanced audio visualization
# import matplotlib.pyplot as plt
# plt.figure(figsize=(15, 10))

# # 1. Display video frame
# plt.subplot(2, 2, 1)
# plt.imshow(video[0].numpy())
# plt.title("Video Frame")

# # 2. Display audio waveform
# plt.subplot(2, 2, 2)
# audio_np = audio.numpy().flatten()
# time_axis = np.arange(0, len(audio_np)) / sample_rate
# plt.plot(time_axis, audio_np)
# plt.title(f"Audio Waveform ({duration_seconds:.2f}s)")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")

# # 3. Display spectrogram
# plt.subplot(2, 2, 3)
# D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_np)), ref=np.max)
# librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram')

# # 4. Display mel spectrogram
# plt.subplot(2, 2, 4)
# S = librosa.feature.melspectrogram(y=audio_np, sr=sample_rate)
# S_dB = librosa.power_to_db(S, ref=np.max)
# librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel-frequency Spectrogram')

# plt.tight_layout()
# plt.savefig("/home/csgrad/susimmuk/acmdeepfake/audio-extraction/audio_visualization.png")
# # plt.show()

# # Optional: Save audio as WAV file for external playback
# import scipy.io.wavfile as wav
# wav.write("/home/csgrad/susimmuk/acmdeepfake/audio-extraction/extracted_audio.wav", sample_rate, audio_np)

def read_audio(path: str):
    audio, rate = torchaudio.load(path)
    # audio = audio.permute(1, 0)
    if audio.shape[0] == 0:
        audio = torch.zeros(1, 1)
    return audio, rate
    
import torchaudio
audio_path = "/home/csgrad/susimmuk/acmdeepfake/data/AV-Deepfake1M-PlusPlus/train/silent_videos/subject_17_ddvyjnjfad_vid_1_4/real.wav"
audio_path = "/home/csgrad/susimmuk/acmdeepfake/data/AV-Deepfake1M-PlusPlus/train/vox_celeb_2/id04265/31ACB9gZfDs/00002/real_video_fake_audio.wav"
waveform, sample_rate = read_audio(audio_path)
emb_scene = get_scene_embedding(waveform,model)
print(emb_scene.shape)
emb_scene, t = get_timestamp_embedding(waveform,model)
print(waveform.shape, sample_rate)
print(emb_scene.shape)
print(t.shape)
