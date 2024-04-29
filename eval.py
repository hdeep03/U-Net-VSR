import torch
from src.model.unet_model import VUNet, UNet
from src.data.dataset import VSRDataset
import argparse
import os
import logging
import numpy as np
from src.metrics.ssim import ssim
import cv2


def reconstruct_image(patches: torch.Tensor) -> torch.Tensor:
    positions = []
    for i in range(16):
        for j in range(9):
            positions.append((j*80, i*80))

    y_img = torch.zeros((3, 720, 1280))
    for idx, (patch, pos) in enumerate(zip(patches, positions)):
        x, y = pos
        y_img[:, x:x+80, y:y+80] = patch.squeeze().cpu()
    return y_img



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./run")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--test_path", type=str, default="./data/raw/test")
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--baseline", action="store_true", default=False)

    args = parser.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    logging.basicConfig(level = logging.INFO, filename = os.path.join(args.run_dir ,"eval.log"), filemode = "a", format = '%(name)s - %(levelname)s - %(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.baseline:
        model = UNet(3)
        testset = VSRDataset(args.test_path, buffer_size=1, patch=args.patch)
    else:
        model = UNet(3 * args.num_frames)
        testset = VSRDataset(args.test_path, buffer_size=args.num_frames, patch=args.patch)
    
    model = model.to(device)

    ckpts = sorted([int(f.split("_")[1].split(".")[0]) for f in os.listdir(args.run_dir) if f.endswith(".pth")])
    if ckpts:
        model.load_state_dict(torch.load(f"{args.run_dir}/model_{ckpts[-1]}.pth", map_location=device))
        logging.info(f"model loaded from {args.run_dir}/model_{ckpts[-1]}.pth")
    model.eval()    
    with torch.inference_mode():
        dataloader = torch.utils.data.DataLoader(testset, batch_size=144)
        i = 0
        for X, y in dataloader:
            i+=1
            if i < 500:
                continue
            X = X.to(device)
            X = X.reshape(X.shape[0], -1, X.shape[-2], X.shape[-1])
            y = y.to(device)
            y = y.squeeze(1)
            y_pred = model(X)
            print(torch.max(y_pred))

            y_img = reconstruct_image(y_pred)
            y_show = (y_img.numpy().transpose(1, 2, 0) * 255).astype('uint8')
            y_gt = reconstruct_image(y)
            y_gt_show = (y_gt.numpy().transpose(1, 2, 0) * 255).astype('uint8')
            cv2.imshow('image', y_show)
            cv2.imwrite('reconstruct.jpg', y_show)
            cv2.waitKey(0)
            cv2.imshow('image', y_gt_show)
            cv2.imwrite('original.jpg', y_gt_show)
            cv2.waitKey(0)
            exit()

    