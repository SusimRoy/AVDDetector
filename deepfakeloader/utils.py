import json
from typing import Tuple
import os
import cv2
import numpy as np
import torch
import torchvision
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
import torchaudio

def read_json(path: str, object_hook=None):
    with open(path, 'r') as f:
        return json.load(f, object_hook=object_hook)


def read_audio(path: str):
    audio, rate = torchaudio.load(path)
    # audio = audio.permute(1, 0)
    if audio.shape[0] == 0:
        audio = torch.zeros(1, 1)
    return audio, rate

def read_video(path: str):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    video = np.stack(frames, axis=0)
    video = rearrange(video, 'T H W C -> T C H W')
    return torch.from_numpy(video) / 255

def read_video_fast(path: str):
    frames = []
    for frame in sorted(os.listdir(path)):
        if frame.endswith(".jpg"):
            img_path = os.path.join(path, frame)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to read {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != 224 or img.shape[1] != 224:
                img = cv2.resize(img, (224, 224))  # Resize to 224x224
            frames.append(img)
    if not frames:
        raise ValueError(f"No frames found in {path}")
    video = np.stack(frames, axis=0)
    video = rearrange(video, 'T H W C -> T C H W')
    return torch.from_numpy(video).float() / 255


def resize_video(tensor: Tensor, size: Tuple[int, int], resize_method: str = "bicubic") -> Tensor:
    return F.interpolate(tensor, size=size, mode=resize_method)


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors."""

    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    iou = inter_len / (union_len + 1e-8)
    return iou


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors + 1e-8)
    return scores


def iou_1d(proposal, target) -> Tensor:
    """
    Calculate 1D IOU for N proposals with L labels.

    Args:
        proposal (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The predicted array with [M, 2]. First column is
            beginning, second column is end.
        target (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The label array with [N, 2]. First column is
            beginning, second column is end.

    Returns:
        :class:`~torch.Tensor`: The iou result with [M, N].
    """
    if type(proposal) is np.ndarray:
        proposal = torch.from_numpy(proposal)

    if type(target) is np.ndarray:
        target = torch.from_numpy(target)

    proposal_begin = proposal[:, 0].unsqueeze(0).T
    proposal_end = proposal[:, 1].unsqueeze(0).T
    target_begin = target[:, 0]
    target_end = target[:, 1]

    inner_begin = torch.maximum(proposal_begin, target_begin)
    inner_end = torch.minimum(proposal_end, target_end)
    outer_begin = torch.minimum(proposal_begin, target_begin)
    outer_end = torch.maximum(proposal_end, target_end)

    inter = torch.clamp(inner_end - inner_begin, min=0.)
    union = outer_end - outer_begin
    return inter / union