# AVDDetector

## Inference

Suppose /data contains the testB video files and the testB_files.txt
### Step 1
Run the prep-dataset/create-test.py file using the command:

```bash
python prep-dataset/create-test.py --video_dir /path/to/videos --output_base_dir /path/to/output
```

This will create the same number of directories as the number of input videos in the output directory. Each such directory has the face extracted frames.

### Step 2
Run the prep-dataset/check-gaps.py file using the command:
```bash
python check-gaps.py --output_base_dir /path/to/output
```

This file checks if there was any frame missing while extracting the face frames and renames the frames to make them consecutive.

### Step 3
Run the motion-magnified/process_frames_ABC.py file using the command:
```bash
python process_frames_ABC.py --base_dir /path/to/output
```

This file creates the frameA, frameB and frameC folders needed for motion-magnification of the video frames.
### Step 4
Run the motion-magnified/apply_magnification.py file using the command:
```bash
python apply_magnification.py --base_dir /path/to/ouput --mag_weight_path /path/to/magnification_weights
```

This file will create the motion magnified video and store it in the same folder as the corresponding video folder where it's frames are stored.

### Step 5
Run the prep-dataset/extract_audio.py using the command:

```bash
python extract_audio.py --video_root /path/to/data --output_base_dir /path/to/output
```

This command will extract the audio from the videos and store them as a ``.wav`` file in the corresponding video directory in the /path/to/output

### Step 6
Run the test.py in the root directory using the command:
```bash 
python test.py --data_root /path/to/output --text_file /path/to/data  --resume /path/to/av-classifier_checkpoint --model av-classifier
```

This file will run the videos through the model and store the results in prediction.txt in the required format.
