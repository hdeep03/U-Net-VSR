import torch
import time
import numpy as np
from model import espcn_x4
from tqdm import trange

mps_device = torch.device("mps")

model = espcn_x4(in_channels=1, out_channels=1, channels=64).to(mps_device)
# compiled_model = model.eval()
compiled_model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))

# Warm-up phase
for _ in trange(10):
    warmup_data = torch.randn(1, 1, 180, 320, device=mps_device)
    out = compiled_model(warmup_data)
    print(out.shape)

total_time = []
for x in trange(100):
    frame_data = torch.randn(1, 1, 180, 320, device=mps_device)
    # start = torch.mps.Event(enable_timing=True)
    # end = torch.mps.Event(enable_timing=True)
    
    start = time.perf_counter()
    compiled_model(frame_data)
    end = time.perf_counter()
    # torch.mps.synchronize()
    total_time.append((end - start)*1000)

print("AVG: {}; STD: {}".format(np.mean(total_time), np.std(total_time)))
