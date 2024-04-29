import torch
from src.model.unet_model import VUNet, UNet
from src.data.dataset import VSRDataset
import argparse
import os
import logging
from tqdm import tqdm
import time
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./run")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--test_path", type=int, default=10)
    parser.add_argument("--patch", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    logging.basicConfig(level = logging.INFO, filename = os.path.join(args.run_dir ,"eval.log"), filemode = "a", format = '%(name)s - %(levelname)s - %(message)s')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.baseline:
        model = UNet(3)
        dataset = VSRDataset(args.data_path, buffer_size=1, patch=args.patch)
        testset = VSRDataset(args.test_path, buffer_size=1, patch=args.patch)
    else:
        model = UNet(3 * args.num_frames)
        dataset = VSRDataset(args.data_path, buffer_size=args.num_frames, patch=args.patch)
        testset = VSRDataset(args.test_path, buffer_size=args.num_frames, patch=args.patch)
    
    model = model.to(device)

    ckpts = sorted([int(f.split("_")[1].split(".")[0]) for f in os.listdir(args.run_dir) if f.endswith(".pth")])
    if ckpts:
        model.load_state_dict(torch.load(f"{args.run_dir}/model_{ckpts[-1]}.pth"))
        logging.info(f"model loaded from {args.run_dir}/model_{ckpts[-1]}.pth")
    model.eval()    
    with torch.inference_mode():
        for el in testset:
            el = el.to(device)

            print(el.shape)
            y = model(el)
            print(y.shape)
            pass




    