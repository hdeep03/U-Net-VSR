import torch
from src.model.unet_model import VUNet, UNet
from src.data.dataset import VSRDataset
import argparse
import os
import logging
import numpy as np
from src.metrics.ssim import ssim
import cv2
from src.data.utils import downsample, bicubic_upsample


def reconstruct_image(patches: torch.Tensor, args) -> torch.Tensor:
    if not args.patch:
        return patches.squeeze(0).cpu()

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
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--test_path", type=str, default="./data/raw/test")
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument("--down_sample", type=int, default=2)
    parser.add_argument("--up_sample", type=int, default=1)

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
    ckpt = f"{args.run_dir}/model_{ckpts[-1]}.pth" if ckpts and not args.ckpt else args.ckpt
    model.load_state_dict(torch.load(ckpt, map_location=device))
    logging.info(f"model loaded from {ckpt}")

    model.eval() 

    for video in os.listdir(args.test_path):
        vidcap = cv2.VideoCapture(os.path.join(args.test_path, video))
        i = 0
        while i < args.num_frames - 1: 
            success,image = vidcap.read()
            if success:
                y = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)/255.0
                x = bicubic_upsample(downsample(y, args.downsample), args.upsample)

                i += 1
            else:
                continue
        while i < vidcap.get(cv2.CAP_PROP_FRAME_COUNT):
            if i%100 == 0:
                print(i)
            success,image = vidcap.read()
            if success:
                y = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)/255.0
                x = bicubic_upsample(downsample(y, args.downsample), args.upsample)
                x = x.to(device)
                x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
                y_pred = model(x)
                y = downsample(y, args.downsample)
                
                


        

    with torch.inference_mode():
        dataloader = torch.utils.data.DataLoader(testset, batch_size=144 if args.patch else 1) 
        i = 0
        for X, y in dataloader:
            i+=1

            if i % 1000 ==0:
                X = X.to(device)
                X = X.reshape(X.shape[0], -1, X.shape[-2], X.shape[-1])
                y = y.to(device)
                y = y.squeeze(1)
                y_pred = model(X)
                print(torch.max(y_pred))
                
                y_img = reconstruct_image(y_pred, args)
                y_show = (y_img.numpy().transpose(1, 2, 0).clip(0, 1) * 255).astype('uint8')
                y_gt = reconstruct_image(y, args)
                y_gt_show = (y_gt.numpy().transpose(1, 2, 0).clip(0,1) * 255).astype('uint8')
                # cv2.imshow('image', y_show)
                cv2.imwrite(os.path.join(args.run_dir, f'reconstruct_{ckpt.split("/")[-1]}_{i}.jpg'), y_show)
                # cv2.waitKey(0)
                # cv2.imshow('image', y_gt_show)
                cv2.imwrite(os.path.join(args.run_dir, f'original_{ckpt.split("/")[-1]}_{i}.jpg'), y_gt_show)
            if i == 5000:
                break
    