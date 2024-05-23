import torch
import numpy as np
# from src.model.unet_model import RUNetSmall, RUNet, UNet
from tqdm import trange
import coremltools as ct
import coremltools.optimize.coreml as cto
import time

# mps_device = torch.device("mps")



model = ct.models.MLModel('runet_small.mlpackage')
print(model.get_spec().description.input)

op_config = cto.OpPalettizerConfig(mode="uniform")

# define optimization config by applying the op config globally to all ops 
config = cto.OptimizationConfig(global_config=op_config)

# palettize weights
compressed_mlmodel = cto.palettize_weights(model, config)

total_time = []
error = []
for x in trange(200):
    frame_data = np.random.rand(144, 30, 80, 80)
    start = time.perf_counter()
    compressed_out = compressed_mlmodel.predict({'x': frame_data})
    normal_out = model.predict({'x': frame_data})
    loss = np.sum((compressed_out['var_331'] - normal_out['var_331'])**2)
    error.append(loss)
    # print(output_dict['var_331'].shape)
    total_time.append(time.perf_counter() - start)

print("AVG LOSS: {}; STD LOSS: {}".format(np.mean(error), np.std(error)))
print("AVG: {}; STD: {}".format(np.mean(total_time), np.std(total_time)))
