import torch
from src.model.unet_model import VUNet, UNet
from src.data.dataset import VSRDataset
import argparse
import os
import logging
import numpy as np
from src.metrics.ssim import ssim
import cv2
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM



def reconstruct_image(patches: torch.Tensor, args) -> torch.Tensor:
    if not args.patch:
        return patches.squeeze(0).cpu()
    mult = args.up_sample * 80
    positions = []
    for i in range(16):
        for j in range(9):
            positions.append((j*mult, i*mult))

    y_img = torch.zeros((3, 720, 1280))
    for idx, (patch, pos) in enumerate(zip(patches, positions)):
        x, y = pos
        y_img[:, x:x+mult, y:y+mult] = patch.squeeze().cpu()
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
        testset = VSRDataset(args.test_path, buffer_size=args.num_frames, patch=args.patch, downsample=args.down_sample, upsample=args.up_sample)
    
    model = model.to(device)

    ckpts = sorted([int(f.split("_")[1].split(".")[0]) for f in os.listdir(args.run_dir) if f.endswith(".pth")])
    ckpt = f"{args.run_dir}/model_{ckpts[-1]}.pth" if ckpts and not args.ckpt else args.ckpt
    model.load_state_dict(torch.load(ckpt, map_location=device))
    logging.info(f"model loaded from {ckpt}")

    mse_loss_fn = torch.nn.functional.mse_loss
    ssim_loss = SSIM( data_range=1.0, size_average=True)
    ssim_loss_fn = lambda ypred, y: 1 - ssim_loss(ypred, y)

    model.eval() 
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
                y_upsample = reconstruct_image(X[:,-3:], args)
                y_upsample_show = (y_upsample.numpy().transpose(1, 2, 0).clip(0,1) * 255).astype('uint8')
                logging.info(f"model mse: {mse_loss_fn(y_pred, y).item()}, ssim: {ssim_loss_fn(y_pred, y).item()}") 
                logging.info(f"naive mse: {mse_loss_fn(X[:,-3:], y).item()}, ssim: {ssim_loss_fn(X[:,-3:], y).item()}")
                # cv2.imshow('image', y_show)
                side_by_side = np.concatenate((y_show, y_gt_show, y_upsample_show), axis=1)
                cv2.imwrite(os.path.join(args.run_dir, f'comparison_{ckpt.split("/")[-1]}_{i}.jpg'), side_by_side)

            if i == 5000:
                break
    