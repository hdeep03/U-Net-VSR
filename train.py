import torch
from src.model.unet_model import VUNet, UNet
from src.data.dataset import VSRDataset
import argparse
import os
import logging
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./run")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_path", type=str, default="./data/raw/train")
    parser.add_argument("--val_path", type=str, default="./data/raw/val")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--log_every", type=int, default=100)

    args = parser.parse_args()
    logging.basicConfig(level = logging.INFO, filename = os.path.join(args.run_dir ,"train.log"), filemode = "w", format = '%(name)s - %(levelname)s - %(message)s')
    logging.info(f"args: {args}")

    os.makedirs(args.run_dir, exist_ok=True)

    if args.baseline:
        model = UNet(3)
        dataset = VSRDataset(args.data_path, buffer_size=1, patch=args.patch)
        valset = VSRDataset(args.val_path, buffer_size=1, patch=args.patch)
    else:
        model = UNet(3 * args.num_frames)
        dataset = VSRDataset(args.data_path, buffer_size=args.num_frames, patch=args.patch)
        valset = VSRDataset(args.val_path, buffer_size=args.num_frames, patch=args.patch)
    
    ckpts = sorted([int(f.split("_")[1].split(".")[0]) for f in os.listdir(args.run_dir) if f.endswith(".pth")])
    if ckpts:
        model.load_state_dict(torch.load(f"{args.run_dir}/model_{ckpts[-1]}.pth"))
        logging.info(f"model loaded from {args.run_dir}/model_{ckpts[-1]}.pth")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size)

    logging.info(f"dataset size: {len(dataset)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)



    for i in range(args.epochs):
        logging.info(f"epoch: {i}")
        j = 0
        model.train()
        for x, y in tqdm(dataloader):        
            # x : [batch_size, nframes, 3, 64, 64]
            # y: [batch_size, 3, 64, 64]
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
            optimizer.zero_grad()
            ypred = model(x)
            loss = torch.nn.functional.mse_loss(ypred, y)
            loss.backward()
            optimizer.step()
    
            j += 1
            if j % args.log_every == 0:
                logging.info(f"loss: {loss}")
        
        torch.save(model.state_dict(), os.path.join(args.run_dir, f"model_{i}.pth"))
        logging.info(f"model saved at {args.run_dir}/model_{i}.pth")
        model.eval()
        with torch.inference_mode():
            total_loss = 0.0
            for x, y in tqdm(valloader):
                x = x.to(device)
                y = y.to(device)
                x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
                ypred = model(x)
                total_loss += torch.nn.functional.mse_loss(ypred, y).item()
            logging.info(f"avg val_loss: {total_loss / len(valset)}")
    

