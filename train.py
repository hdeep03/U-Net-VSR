import torch
from src.model.unet_model import VUNet, UNet
from src.data.dataset import VSRDataset
import argparse
import os
import logging
from tqdm import tqdm
import time
import wandb
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./run")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_path", type=str, default="./data/raw/train")
    parser.add_argument("--val_path", type=str, default="./data/raw/val")
    parser.add_argument("--loss_fn", type=str, default="mse")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1000)

    args = parser.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    logging.basicConfig(level = logging.INFO, filename = os.path.join(args.run_dir ,"train.log"), filemode = "a", format = '%(name)s - %(levelname)s - %(message)s')
    logging.info(f"args: {args}")

    wandb.init(project="VSR", name=args.run_dir, config=args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.baseline:
        model = UNet(3)
        dataset = VSRDataset(args.data_path, buffer_size=1, patch=args.patch)
        valset = VSRDataset(args.val_path, buffer_size=1, patch=args.patch)
    else:
        model = UNet(3 * args.num_frames)
        dataset = VSRDataset(args.data_path, buffer_size=args.num_frames, patch=args.patch)
        valset = VSRDataset(args.val_path, buffer_size=args.num_frames, patch=args.patch)
    
    model = model.to(device)

    ckpts = sorted([int(f.split("_")[1].split(".")[0]) for f in os.listdir(args.run_dir) if f.endswith(".pth")])
    if ckpts:
        model.load_state_dict(torch.load(f"{args.run_dir}/model_{ckpts[-1]}.pth"))
        logging.info(f"model loaded from {args.run_dir}/model_{ckpts[-1]}.pth")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size)

    logging.info(f"dataset size: {len(dataset)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    loss_fn = torch.nn.functional.mse_loss
    if args.loss_fn == "ssim":
        ssim_loss = SSIM( data_range=1.0, size_average=True)

        loss_fn = lambda ypred, y: 1 - ssim_loss(ypred, y)
    elif args.loss_fn == "ms_ssim":
        ssim_loss = MS_SSIM( data_range=1.0, size_average=True)
        loss_fn = lambda ypred, y: 1 - ssim_loss(ypred, y)
    j = 0
    for i in range(args.epochs):
        logging.info(f"epoch: {i}")

        time1 = time.time()
        for x, y in tqdm(dataloader):        
            model.train()
            # x : [batch_size, nframes, 3, 64, 64]
            # y: [batch_size, 3, 64, 64]
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
            y = y.reshape(y.shape[0], -1, y.shape[-2], y.shape[-1])
            optimizer.zero_grad()
            ypred = model(x)
            loss = loss_fn(ypred, y)
            loss.backward()
            optimizer.step()
    
            j += 1
            if j % args.log_every == 0:
                logging.info(f"loss: {loss}, sample/s = {args.log_every * x.shape[0] / (time.time() - time1)}")
                wandb.log({"train_loss": loss, "samples_per_second": args.log_every * x.shape[0] / (time.time() - time1)})
                time1 = time.time()
            if j % args.save_every == 0:
                torch.save(model.state_dict(), os.path.join(args.run_dir, f"model_{j}.pth"))
                logging.info(f"model saved at {args.run_dir}/model_{j}.pth")
        
                model.eval()
                num_val_samples = 100
                with torch.inference_mode():
                    total_loss = 0.0
                    i = 0
                    for x, y in tqdm(valloader):
                        x = x.to(device)
                        y = y.to(device)
                        x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
                        y = y.reshape(y.shape[0], -1, y.shape[-2], y.shape[-1])
                        ypred = model(x)
                        total_loss += loss_fn(ypred, y).item()
                        i += 1
                        if i == num_val_samples:
                            break
                    logging.info(f"avg val_loss: {total_loss / num_val_samples}")
                    wandb.log({"val_loss": total_loss / num_val_samples})
    

    wandb.finish()