from typing import Tuple
import os
from torch.utils.data import IterableDataset
import cv2
import torch
from tqdm import tqdm
from utils import downsample, bicubic_upsample

class VSRDataset(IterableDataset):
    def __init__(self, data_dir, buffer_size=10, patch=False, downsample=2, upsample=1):
        self.data_dir = data_dir
        self.videos = [f for f in os.listdir(data_dir) if f.endswith(".webm")]
        self.buffer_size = buffer_size
        self.patch = patch
        self.downsample = downsample
        self.upsample = upsample

    def process_image(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)/255.0
        x = bicubic_upsample(downsample(y, self.downsample), self.upsample)
        return x, y

    def extract_patches(self, x, y):
        mult = 80 * self.upsample
        ex = lambda im, px, py: im[:, :, mult*(py):mult*(py+1), mult*(px):mult*(px+1)]
        for i in range(16):
            for j in range(9):
                yield ex(x, i, j), ex(y, i, j)

    def __iter__(self):
        for video in self.videos:
            buffer = torch.empty((0, 3, 720*self.upsample, 1280*self.upsample))
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
    def __len__(self):
        ans=0
        for video in self.videos:
            vidcap = cv2.VideoCapture(os.path.join(self.data_dir, video))
            ans+=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            vidcap.release()
        return ans


if __name__ == "__main__":
    dataset = VSRDataset("./data/raw/val", patch=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024)
    for X, Y in tqdm(dataloader):
        pass
