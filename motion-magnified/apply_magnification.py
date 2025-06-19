"""
Apply motion magnification using MagNet model on frames from frameA, frameB, frameC folders.
The magnified frames will be stored in the same directory as the input frames.
"""
import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from magnet import MagNet
from dataloader import get_gen_ABC, unit_postprocessing, numpy2cuda
import time

class Config(object):
    def __init__(self):
        # General
        self.epochs = 12
        self.batch_size = 6
        self.date = '0510'
        
        # Data
        self.data_dir = '../../../datasets/mm'
        self.dir_train = os.path.join(self.data_dir, 'train')
        self.dir_test = os.path.join(self.data_dir, 'test')
        self.frames_train = 'coco100000'
        self.cursor_end = int(self.frames_train.split('coco')[-1])
        if os.path.exists(os.path.join(self.dir_train, 'train_mf.txt')):
            self.coco_amp_lst = np.loadtxt(os.path.join(self.dir_train, 'train_mf.txt'))[:self.cursor_end]
        else:
            # print('Please load train_mf.txt if you want to do training.')
            self.coco_amp_lst = None
        self.videos_train = []
        self.load_all = False

        # Training
        self.lr = 1e-4
        self.betas = (0.9, 0.999)
        self.batch_size_test = 1
        self.preproc = ['poisson']
        self.pretrained_weights = ''

        # Callbacks
        self.num_val_per_epoch = 10
        self.save_dir = 'weights_date{}'.format(self.date)
        self.time_st = time.time()
        self.losses = []

def apply_magnification(frame_dir, model, amp_factor=2.5, output_name="magnified_video.mp4", fps=30):
    """
    Apply motion magnification to frames using MagNet model
    Args:
        frame_dir (str): Directory containing frameA, frameB, frameC folders
        model (MagNet): Loaded MagNet model
        amp_factor (float): Amplification factor
        output_name (str): Name of the output video file
        fps (int): Frames per second for the output video
    """
    # Create a temporary config for this directory
    config = Config()
    config.dir_test = frame_dir
    
    # Get data loader
    data_loader = get_gen_ABC(config, mode='test_on_testset')
    # print('Number of test image couples:', data_loader.data_len)
    
    if data_loader.data_len == 0:
        # print(f"No frames found in {frame_dir}")
        return
    
    # Get video dimensions from first frame
    vid_size = cv2.imread(data_loader.paths[0]).shape[:2][::-1]
    
    # Create video writer
    output_path = os.path.join(frame_dir, output_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, vid_size)
    
    # print(f"Applying magnification to frames in {frame_dir}")
    # print(f"Output video will be saved as: {output_path}")
    
    # Process frames in batches
    frames = []
    for idx_load in range(0, data_loader.data_len, data_loader.batch_size):
        # if (idx_load+1) % 100 == 0:
        #     print('{}'.format(idx_load+1), end=', ')
            
        # Get batch of frames
        batch_A, batch_B = data_loader.gen_test()
        
        # Prepare amplification factor
        amp_factor_tensor = numpy2cuda(amp_factor)
        for _ in range(len(batch_A.shape) - len(amp_factor_tensor.shape)):
            amp_factor_tensor = amp_factor_tensor.unsqueeze(-1)
        
        # Apply model
        with torch.no_grad():
            # batch_A = torch.nn.functional.interpolate(batch_A, size=(224, 224), mode='bilinear')
            # batch_B = torch.nn.functional.interpolate(batch_B, size=(224, 224), mode='bilinear')
            y_hats = model(batch_A, batch_B, None, None, amp_factor_tensor, mode='evaluate')
        
        # Process output frames
        for y_hat in y_hats:
            y_hat = unit_postprocessing(y_hat, vid_size=vid_size)
            frames.append(y_hat)
            if len(frames) >= data_loader.data_len:
                break
                
        if len(frames) >= data_loader.data_len:
            break
    
    # Add first frame
    data_loader = get_gen_ABC(config, mode='test_on_testset')
    frames = [unit_postprocessing(data_loader.gen_test()[0], vid_size=vid_size)] + frames
    
    # Write frames to video
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    # Release video writer
    out.release()
        # print(f"Video creation completed: {output_path}")

def process_directory(base_dir, model, amp_factor=2.5):
    """
    Process all video directories in the base directory
    Args:
        base_dir (str): Base directory containing video directories
        model (MagNet): Loaded MagNet model
        amp_factor (float): Amplification factor
    """
    for identity in tqdm(os.listdir(base_dir)):
        identity_path = os.path.join(base_dir, identity)
        if not os.path.isdir(identity_path):
            continue
            
        # print(f"\nProcessing identity: {identity}")
        
        # Process real videos
        # real_path = os.path.join(identity_path, 'real')
        # if os.path.exists(real_path):
        #     # print(f"Processing real videos for {identity}")
        #     for video_dir in os.listdir(real_path):
        #         video_path = os.path.join(real_path, video_dir)
        #         if os.path.isdir(video_path) and not os.path.exists(os.path.join(video_path, f"{video_dir}_magnified.mp4")):
        #             apply_magnification(video_path, model, amp_factor, f"{video_dir}_magnified.mp4")
        
        # Process fake videos
        fake_path = os.path.join(identity_path, 'fake')
        if os.path.exists(fake_path):
            # print(f"Processing fake videos for {identity}")
            for video_dir in os.listdir(fake_path):
                video_path = os.path.join(fake_path, video_dir)
                if os.path.isdir(video_path) and not os.path.exists(os.path.join(video_path, f"{video_dir}_magnified.mp4")):
                    apply_magnification(video_path, model, amp_factor, f"{video_dir}_magnified.mp4")

def main():
    # if len(sys.argv) != 2:
    #     print("Usage: python apply_magnification.py <extracted_frames_directory>")
    #     sys.exit(1)
    
    # base_dir = sys.argv[1]
    # if not os.path.exists(base_dir):
    #     print(f"Error: Directory {base_dir} does not exist")
    #     sys.exit(1)
    
    # # Initialize and load model
    weights_path = '/home/csgrad/susimmuk/acmdeepfake/motion-magnified/magnet_epoch12_loss7.28e-02.pth'
    model = MagNet().cuda()
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # # Process the directory
    # process_directory(base_dir, model)
    apply_magnification(
        frame_dir='/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/vox_celeb_2/id00561/fake/8snVgZMIzRY_00032',
        model=model,
        amp_factor=2.5,
        output_name="8snVgZMIzRY_00032_magnified.mp4",
        fps=30
    )

if __name__ == '__main__':
    main() 