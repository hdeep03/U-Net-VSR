# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import time
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
import config
import model
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset, TrainValidImageFromVideoDataset
# from image_quality_assessment import PSNR, SSIM
from utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter
from src.data.utils import downsample, bicubic_upsample
import imgproc
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


def downsample(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    return F.interpolate(x, size=(x.size()[-2]//factor, x.size()[-1]//factor), mode='bilinear', align_corners=False)
def bicubic_upsample(x: torch.Tensor, factor: int=1) -> torch.Tensor:
    return F.interpolate(x, size=(720*factor, 1280*factor), mode='bicubic', align_corners=False)

def reconstruct_image(patches: torch.Tensor) -> torch.Tensor:
    mult = 80
    positions = []
    for i in range(16):
        for j in range(9):
            positions.append((j*mult, i*mult))

    y_img = torch.zeros((3, 720, 1280))
    for idx, (patch, pos) in enumerate(zip(patches, positions)):
        x, y = pos
        y_img[:, x:x+mult, y:y+mult] = patch.squeeze().cpu()
    return y_img

def extract_patches(self, x):
    patches = []
    mult = 80
    ex = lambda im, px, py: im[:, :, mult*(py):mult*(py+1), mult*(px):mult*(px+1)]
    for i in range(16):
        for j in range(9):
            patches.append(ex(x, i, j)) 
    return patches

def pnsr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    max_pixel_val = 255.0
    return 10 * np.log10(max_pixel_val**2 / mse)
def main():
    # Initialize the number of training epochs
    start_epoch = 0
    espcn_model = build_model().eval()
    print(f"Build `{config.model_arch_name}` model successfully.")

    test_path = "/home/alex/U-Net-VSR/data/raw/test_images"
    psnr_diff = 0.0
    ssim_diff = 0.0
    ssim_loss = SSIM( data_range=1.0, size_average=True)
    ssim_loss_fn = lambda ypred, y: 1 - ssim_loss(ypred, y)

    with torch.inference_mode():
        for image_file in os.listdir(test_path):
            image_path = os.path.join(test_path, image_file)
            img = cv2.imread(image_path) / 255.0
            sr_patches = np.array(img.shape)

            naive_img_tensor = torch.tensor(img, dtype=torch.float32).permute(2,0,1)
            naive_img_tensor = bicubic_upsample(downsample(naive_img_tensor.unsqueeze(0),4)).squeeze(0)

            for i in range(16):
                for j in range(9):
            
                    img_tensor = torch.tensor(img[i*80:(i+1)*80, j*80:(j+1)*80], dtype=torch.float32).permute(2,0,1)/255.0
                    gt_image_path = os.path.join(test_path, image_file.replace(".png", "_gt_crop.png"))
                    
                    cv2.imwrite(gt_image_path, (img_tensor.permute(1,2,0).numpy() *255.0).clip(0,255).astype(np.uint8))
                    img_tensor = downsample(img_tensor.unsqueeze(0),4).squeeze(0)
                    lr_path = os.path.join(test_path, image_file.replace(".png", "_lrx4.png"))                    
                    cv2.imwrite(lr_path, (img_tensor.permute(1,2,0).numpy() *255.0).clip(0,255).astype(np.uint8))
                    
                    gt_y_tensor, gt_cb_image, gt_cr_image = imgproc.preprocess_one_image(gt_image_path, config.device)
                    lr_y_tensor, lr_cb_image, lr_cr_image = imgproc.preprocess_one_image(lr_path, "cuda")
                    sr_y_tensor = espcn_model(lr_y_tensor)
                    print(sr_y_tensor.shape)
                    print(lr_y_tensor.shape)
                    sr_y_image = imgproc.tensor_to_image(sr_y_tensor, range_norm=False, half=True)
                    sr_y_image = sr_y_image.astype(np.float32) / 255.0
                    sr_ycbcr_image = cv2.merge([sr_y_image, gt_cb_image, gt_cr_image])
                    sr_image = imgproc.ycbcr_to_bgr(sr_ycbcr_image)
                    sr_patches[i*80:(i+1)*80, j*80:(j+1)*80] = sr_image
                
            naive_img = (naive_img_tensor.numpy().clip(0,1)*255).astype('uint8')
            sr_patches = (sr_patches.clip(0,1)*255).astype('uint8')
            img = cv2.imread(image_path)
            img_tensor = torch.tensor(img, dtype=torch.float32).permute(2,0,1)/255.0
            sr_patches_tensor = torch.tensor(sr_patches, dtype=torch.float32).permute(2,0,1)/255.0

            psnr_model = pnsr(sr_patches, img).item()
            psnr_upsample = pnsr(naive_img, img).item()
            ssim_model = ssim_loss_fn( sr_patches_tensor , img_tensor).item()
            ssim_upsample = ssim_loss_fn( naive_img_tensor, img_tensor).item()
            psnr_diff += psnr_model - psnr_upsample
            ssim_diff += ssim_upsample - ssim_model

    print(f"PSNR_diff: {psnr_diff/5.0}, SSIM_diff: {ssim_diff/5.0}")













def build_model() -> nn.Module:
    espcn_model = model.__dict__[config.model_arch_name](in_channels=config.in_channels,
                                                         out_channels=config.out_channels,
                                                         channels=config.channels)
    espcn_model = espcn_model.to(device=config.device)

    return espcn_model


if __name__ == "__main__":
    main()
