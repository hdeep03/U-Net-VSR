from typing import Tuple
import os
from torch.utils.data import IterableDataset
import cv2
import torch
import numpy as np
from collections import deque
from utils import downsample, bicubic_upsample

class VSRDataset(IterableDataset):
    def __init__(self, data_dir, buffer_size=10):
        self.data_dir = data_dir
        self.videos = [f for f in os.listdir(data_dir) if f.endswith(".webm")]
        self.buffer_size = buffer_size

    def process_buffer(self, buffer) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.tensor(np.array(buffer))
        y = y.permute(0, 3, 1, 2)
        x = bicubic_upsample(downsample(y))
        return x, y

    def __iter__(self):
        for video in self.videos:
            buffer = deque([])
            vidcap = cv2.VideoCapture(os.path.join(self.data_dir, video))
            success = True
            while success:
                success, image = vidcap.read()
                buffer.append(image)
                if len(buffer) < self.buffer_size:
                    continue
                yield self.process_buffer(buffer)
                buffer.popleft()
