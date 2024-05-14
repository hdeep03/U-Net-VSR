import torch
from src.model.unet_model import UNet, RUNet, RUNetSmall
from src.data.dataset import VSRDataset
import argparse
import os
import logging
import numpy as np
from src.metrics.ssim import ssim
import cv2
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import time
from torch.profiler import profile, record_function, ProfilerActivity


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

def pnsr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    max_pixel_val = 255.0
    return 10 * np.log10(max_pixel_val**2 / mse)


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
    parser.add_argument("--use_tanh", action="store_true")
    parser.add_argument("--use_runet", action="store_true")
    parser.add_argument("--use_subp", action="store_true")
    parser.add_argument("--use_runetsmall", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    logging.basicConfig(level = logging.INFO, filename = os.path.join(args.run_dir ,"eval.log"), filemode = "a", format = '%(name)s - %(levelname)s - %(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.baseline:
        model = UNet(3, use_tanh=args.use_tanh)
        testset = VSRDataset(args.test_path, buffer_size=1, patch=args.patch)
    elif args.use_runet:
        model = RUNet(3 * args.num_frames, use_tanh=args.use_tanh)
        testset = VSRDataset(args.test_path, buffer_size=args.num_frames, patch=args.patch,downsample=args.down_sample, upsample=args.up_sample)
    elif args.use_runetsmall:
        model = RUNetSmall(3 * args.num_frames, use_tanh=args.use_tanh)
        testset = VSRDataset(args.test_path, buffer_size=args.num_frames, patch=args.patch,downsample=args.down_sample, upsample=args.up_sample)
    else:
        model = UNet(3 * args.num_frames, use_tanh=args.use_tanh)
        testset = VSRDataset(args.test_path, buffer_size=args.num_frames, patch=args.patch,downsample=args.down_sample, upsample=args.up_sample)

    
    model = model.to(device)

    ckpts = sorted([int(f.split("_")[1].split(".")[0]) for f in os.listdir(args.run_dir) if f.endswith(".pth")])
    ckpt = f"{args.run_dir}/model_{ckpts[-1]}.pth" if ckpts and not args.ckpt else args.ckpt
    model.load_state_dict(torch.load(ckpt, map_location=device))
    logging.info(f"model loaded from {ckpt}")

    mse_loss_fn = torch.nn.functional.mse_loss
    ssim_loss = SSIM( data_range=1.0, size_average=True)
    ssim_loss_fn = lambda ypred, y: 1 - ssim_loss(ypred, y)

    model.eval() 
    psnr_diff = 0.0
    ssim_diff = 0.0
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
                start_time = time.time()
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                    # with record_function("model_inference"):
                y_pred = model(X)
                
                logging.info(f"elapsed time: {time.time() - start_time}")
                print(torch.max(y_pred))
                logging.info(f"max gpu memory used: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")
                # logging.info(f"max gpu memory used: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")
                
                y_img = reconstruct_image(y_pred, args)
                y_show = (y_img.numpy().transpose(1, 2, 0).clip(0, 1) * 255).astype('uint8')
                y_gt = reconstruct_image(y, args)
                y_gt_show = (y_gt.numpy().transpose(1, 2, 0).clip(0,1) * 255).astype('uint8')
                y_upsample = reconstruct_image(X[:,-3:], args)
                y_upsample_show = (y_upsample.numpy().transpose(1, 2, 0).clip(0,1) * 255).astype('uint8')
                psnr_model = pnsr(y_show, y_gt_show).item()
                psnr_upsample = pnsr(y_upsample_show, y_gt_show).item()
                ssim_model = ssim_loss_fn(y_pred, y).item()
                ssim_upsample = ssim_loss_fn(X[:,-3:], y).item()

                psnr_diff += psnr_model - psnr_upsample
                ssim_diff += ssim_upsample - ssim_model
                logging.info(f"model mse: {mse_loss_fn(y_pred, y).item()}, ssim: {ssim_model}, psnr: {psnr_model}") 
                logging.info(f"naive mse: {mse_loss_fn(X[:,-3:], y).item()}, ssim: {ssim_upsample}, psnr: {psnr_upsample}")
                # cv2.imshow('image', y_show)
                # print(y_show - y_upsample_show)
                side_by_side = np.concatenate((y_show, y_gt_show, y_upsample_show), axis=1)
                cv2.imwrite(os.path.join(args.run_dir, f'comparison_{ckpt.split("/")[-1]}_{i}.jpg'), side_by_side)
                cv2.imwrite(os.path.join(args.run_dir, f'orig_{ckpt.split("/")[-1]}_{i}.jpg'), y_gt_show)
                cv2.imwrite(os.path.join(args.run_dir, f'upsample_{ckpt.split("/")[-1]}_{i}.jpg'), y_upsample_show)
                cv2.imwrite(os.path.join(args.run_dir, f'model_{ckpt.split("/")[-1]}_{i}.jpg'), y_show)
                
                
            if i == 5000:
                break
        logging.info(f"avg psnr diff: {psnr_diff / 5}, avg ssim diff: {ssim_diff / 5}")

    