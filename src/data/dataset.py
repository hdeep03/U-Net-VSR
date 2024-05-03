from typing import Tuple
import os
from torch.utils.data import IterableDataset
import cv2
import torch
from tqdm import tqdm
from utils import downsample, bicubic_upsample

class VSRDataset(IterableDataset):
    def __init__(self, data_dir, buffer_size=10, patch=False, downsample_factor=2):
        self.data_dir = data_dir
        self.videos = [f for f in os.listdir(data_dir) if f.endswith(".webm")]
        self.buffer_size = buffer_size
        self.patch = patch
        self.downsample_factor = downsample_factor

    def process_image(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)/255.0
        original_size = (y.shape[2], y.shape[3])
        size = (y.shape[2]//self.downsample_factor, y.shape[3]//self.downsample_factor)
        x = bicubic_upsample(downsample(y, size), original_size)
        return x, y

    def extract_patches(self, x, y):
        ex = lambda im, px, py: im[:, :, 80*(py):80*(py+1), 80*(px):80*(px+1)]
        for i in range(16):
            for j in range(9):
                yield ex(x, i, j), ex(y, i, j)

    def __iter__(self):
        for video in self.videos:
            buffer = torch.empty((0, 3, 720, 1280))
            vidcap = cv2.VideoCapture(os.path.join(self.data_dir, video))
            if not vidcap.isOpened():
                continue
            success = True
            while success:
                success, image = vidcap.read()
                if not success:
                    break
                out_x, out_y = self.process_image(image)
                buffer = torch.cat((buffer, out_x), dim=0)
                if buffer.shape[0] != self.buffer_size:
                    continue

                if not self.patch:
                    yield buffer, out_y
                else:
                    yield from self.extract_patches(buffer, out_y)
                buffer = buffer[1:]
            vidcap.release()


if __name__ == "__main__":
    dataset = VSRDataset("./data/raw/val", patch=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024)
    for X, Y in tqdm(dataloader):
        pass
