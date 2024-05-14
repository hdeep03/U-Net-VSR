import torch
import numpy as np
from model import espcn_x4
from tqdm import trange

mps_device = torch.device("mps")

model = espcn_x4(in_channels=1, out_channels=1, channels=64).to(mps_device)
compiled_model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))

# Warm-up phase
for _ in range(10):
    warmup_data = torch.randn(1, 1, 180, 320, device=mps_device)
    compiled_model(warmup_data)

total_time = []
for x in trange(24):
    frame_data = torch.randn(1, 1, 180, 320, device=mps_device)
    start = torch.mps.Event(enable_timing=True)
    end = torch.mps.Event(enable_timing=True)
    
    start.record()
    compiled_model(frame_data)
    end.record()
    
    # end.synchronize()  # Ensure the end event has completed
    total_time.append(start.elapsed_time(end))

print("AVG: {}; STD: {}".format(np.mean(total_time), np.std(total_time)))
