import torch
import numpy as np
from src.model.unet_model import RUNetSmall
from tqdm import trange

mps_device = torch.device("mps")

model = RUNetSmall(30, use_tanh=True).to(mps_device)
compiled_model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))

total_time = []
for x in trange(24):
    frame_data = torch.randn(1, 30, 720, 1280, device=mps_device)
    start = torch.mps.Event(enable_timing=True)
    end = torch.mps.Event(enable_timing=True)
    start.record()
    compiled_model(frame_data)
    end.record()
    torch.mps.synchronize()
    total_time.append(start.elapsed_time(end))

print("AVG: {}; STD: {}".format(np.mean(total_time), np.std(total_time)))
