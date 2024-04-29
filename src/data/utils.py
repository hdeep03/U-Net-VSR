import torch
import torch.nn.functional as F

def downsample(x: torch.Tensor) -> torch.Tensor:
    new_height = 360
    new_width = int(new_height * 16 / 9)
    x_resized = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)
    return x_resized


def bicubic_upsample(x: torch.Tensor) -> torch.Tensor:
    new_height = 720
    new_width = int(new_height * 16 / 9)
    x_resized = F.interpolate(x, size=(new_height, new_width), mode='bicubic', align_corners=False)
    return x_resized