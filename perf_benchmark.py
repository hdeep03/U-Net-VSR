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

op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=4)

# define optimization config by applying the op config globally to all ops 
config = cto.OptimizationConfig(global_config=op_config)

# palettize weights
compressed_mlmodel = cto.palettize_weights(model, config)

total_time = []
for x in trange(200):
    frame_data = np.random.rand(144, 30, 80, 80)
    start = time.perf_counter()
    output_dict = compressed_mlmodel.predict({'x': frame_data})
    # print(output_dict['var_331'].shape)
    total_time.append(time.perf_counter() - start)

print("AVG: {}; STD: {}".format(np.mean(total_time), np.std(total_time)))
