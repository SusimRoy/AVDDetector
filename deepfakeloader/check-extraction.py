from utils import read_video, read_video_fast

video = read_video("/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2/id07995/fake/84UbI2isyB4_00003/84UbI2isyB4_00003_magnified.mp4")
video2 = read_video_fast("/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2/id07995/fake/84UbI2isyB4_00003/")

print(video.shape)
print(video2.shape)
# print(audio.shape)
# print(info)

# print(video2.shape)