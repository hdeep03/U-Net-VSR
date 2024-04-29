from typing import Tuple
import os
from torch.utils.data import IterableDataset
import cv2
import torch
from tqdm import tqdm
from utils import downsample, bicubic_upsample

class VSRDataset(IterableDataset):
    def __init__(self, data_dir, buffer_size=10):
        self.data_dir = data_dir
        self.videos = [f for f in os.listdir(data_dir) if f.endswith(".webm")]
        self.buffer_size = buffer_size

    def process_image(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)/255.0
        x = bicubic_upsample(downsample(y))
        return x, y

    def __iter__(self):
        for video in self.videos:
            buffer = torch.empty((0, 3, 720, 1280))
            vidcap = cv2.VideoCapture(os.path.join(self.data_dir, video))
            success = True
            while success:
                success, image = vidcap.read()
                out_x, out_y = self.process_image(image)
                buffer = torch.cat((buffer, out_x), dim=0)
                if buffer.shape[0] < self.buffer_size:
                    continue
                yield buffer, out_y
                buffer = buffer[1:]


if __name__ == "__main__":
    dataset = VSRDataset("./data/raw/val")
    for x, y in tqdm(dataset):
        pass
        # first_frame = x[0].permute(1, 2, 0).numpy()
        # cv2.imshow("First frame", first_frame)
        # cv2.imshow("Ground truth", y[0].permute(1, 2, 0).numpy())
        # cv2.waitKey(0)
        # exit()