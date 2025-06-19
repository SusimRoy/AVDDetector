import os
import shutil
import tempfile
import numpy as np
from types import SimpleNamespace
from typing import List, Dict, Tuple, Optional, Union, Any
import torch
from torch.utils.data import DataLoader

from .loader import AVDeepfake1mPlusPlusImages
from .dataloader import get_gen_ABC, DataGen
from .utils import read_json, read_video, resize_video, iou_with_anchors, \
    read_video_fast, read_video, iou_1d


class MagNetFrameDataset(AVDeepfake1mPlusPlusImages):
    """
    Dataset that extracts frames from videos and organizes them for use with MagNet.
    Extends AVDeepfake1mPlusPlusImages to add MagNet compatibility.
    """
    
    def __init__(self, *args, temp_dir: Optional[str] = None, 
                 batch_size: int = 4, **kwargs):
        """
        Initialize the MagNet frame dataset.
        
        Args:
            *args: Arguments to pass to AVDeepfake1mPlusPlusImages
            temp_dir: Directory to store temporary organized frames (None for auto-generated)
            batch_size: Batch size for MagNet processing
            **kwargs: Additional arguments for AVDeepfake1mPlusPlusImages
        """
        super().__init__(*args, **kwargs)
        
        self.batch_size = batch_size
        self.temp_dir = temp_dir
        
        # If temp_dir is not specified, create one
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        else:
            os.makedirs(self.temp_dir, exist_ok=True)
            
        # Create necessary subdirectories for MagNet
        self._create_magnet_dirs()
        
    def _create_magnet_dirs(self):
        """Create directories required by MagNet"""
        os.makedirs(os.path.join(self.temp_dir, 'frameA'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'frameB'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'frameC'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'amplified'), exist_ok=True)
    
    def process_video_frames(self, 
                             video_frames: List[torch.Tensor], 
                             save_frames: bool = True) -> Tuple[str, Dict[str, List[str]]]:
        """
        Organize frames from a single video into the format required by MagNet.
        
        Args:
            video_frames: List of video frame tensors
            save_frames: If True, save frames to disk, otherwise just organize in memory
            
        Returns:
            Tuple of (organized directory path, dictionary of frame sets)
        """
        frame_sets = {'frameA': [], 'frameB': [], 'frameC': [], 'amplified': []}
        
        # Process frames in A, B, C pattern
        for i, frame in enumerate(video_frames):
            set_idx = i % 3
            
            if set_idx == 0:
                set_name = 'frameA'
            elif set_idx == 1:
                set_name = 'frameB'
            else:
                set_name = 'frameC'
                
            frame_path = os.path.join(self.temp_dir, set_name, f"{i//3}.jpg")
            
            if save_frames:
                # Convert tensor to image and save
                frame_img = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                import cv2
                cv2.imwrite(frame_path, cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))
            
            frame_sets[set_name].append(frame_path)
            
            # Create amplified version for A frames
            if set_name == 'frameA':
                amp_path = os.path.join(self.temp_dir, 'amplified', f"{i//3}.jpg")
                if save_frames:
                    shutil.copy(frame_path, amp_path)
                frame_sets['amplified'].append(amp_path)
        
        return self.temp_dir, frame_sets
    
    def get_magnet_generator(self, video_idx: Optional[int] = None, 
                            mode: str = 'train',
                            num_frames: Optional[int] = None) -> DataGen:
        """
        Get a MagNet-compatible data generator for a specific video.
        
        Args:
            video_idx: Index of the video to extract frames from (None = extract all)
            mode: 'train' or 'test' mode for the generator
            num_frames: Optional limit on number of frames to process
            
        Returns:
            MagNet DataGen object
        """
        # Clear existing files
        self._create_magnet_dirs()
        
        # Extract frames from the video
        if video_idx is not None:
            meta = self.metadata[video_idx]
            video_frames = []
            
            # Read the video and extract frames
            video = read_video_fast(os.path.join(self.data_root, self.subset, meta.file))
            if self.image_size != 224:
                video = resize_video(video, (self.image_size, self.image_size))
                
            # Limit number of frames if needed
            if num_frames is not None:
                video = video[:num_frames]
                
            # Process frames for MagNet
            self.process_video_frames(video)
        
        # Create MagNet configuration
        config = SimpleNamespace()
        config.dir_train = self.temp_dir
        config.dir_test = self.temp_dir
        config.data_dir = os.path.dirname(self.temp_dir)
        config.batch_size = self.batch_size
        config.batch_size_test = self.batch_size
        config.cursor_end = 100  # signifies the length of the video(no. of frames you want to take from each video)
        config.videos_train = []
        config.skip = 0
        config.preproc = []  # No preprocessing
        config.load_all = False
        config.coco_amp_lst = [1.0] * (len(os.listdir(os.path.join(self.temp_dir, 'frameA'))))
        
        # Get data generator
        data_gen = get_gen_ABC(config, mode=mode)
        return data_gen

    def iter_videos_as_magnet(self, batch_size: int = 4, 
                             mode: str = 'train',
                             max_frames_per_video: Optional[int] = None):
        """
        Iterator that yields MagNet DataGen objects for each video.
        
        Args:
            batch_size: Batch size for MagNet
            mode: 'train' or 'test' mode
            max_frames_per_video: Maximum number of frames to process per video
        
        Yields:
            Tuples of (video_index, DataGen object)
        """
        for idx in range(len(self.metadata)):
            yield idx, self.get_magnet_generator(video_idx=idx, mode=mode, num_frames=max_frames_per_video)

    @staticmethod
    def create_dataloader_from_frames(frames: List[torch.Tensor], 
                                     output_dir: str,
                                     batch_size: int = 4, 
                                     mode: str = 'train'):
        """
        Static utility to create a MagNet data generator directly from a list of frames.
        
        Args:
            frames: List of frame tensors
            output_dir: Directory to store organized frames
            batch_size: Batch size for processing
            mode: 'train' or 'test' mode
            
        Returns:
            DataGen object that can be used with MagNet
        """
        # Create directories
        os.makedirs(os.path.join(output_dir, 'frameA'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'frameB'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'frameC'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'amplified'), exist_ok=True)
        
        # Save frames in A, B, C pattern
        import cv2
        for i, frame in enumerate(frames):
            set_idx = i % 3
            if set_idx == 0:
                set_name = 'frameA'
            elif set_idx == 1:
                set_name = 'frameB'
            else:
                set_name = 'frameC'
                
            # Convert tensor to image and save
            frame_img = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frame_path = os.path.join(output_dir, set_name, f"{i//3}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))
            
            # Create amplified version for A frames
            if set_name == 'frameA':
                amp_path = os.path.join(output_dir, 'amplified', f"{i//3}.jpg")
                shutil.copy(frame_path, amp_path)
        
        # Create MagNet configuration
        config = SimpleNamespace()
        config.dir_train = output_dir
        config.dir_test = output_dir
        config.data_dir = os.path.dirname(output_dir)
        config.batch_size = batch_size
        config.batch_size_test = batch_size
        config.cursor_end = 1
        config.videos_train = []
        config.skip = 0
        config.preproc = []
        config.load_all = False
        config.coco_amp_lst = [1.0] * len(frames)
        
        # Get data generator
        return get_gen_ABC(config, mode=mode)
