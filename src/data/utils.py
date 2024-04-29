import torch
import torch.nn.functional as F

def downsample(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=(360, 640), mode='bilinear', align_corners=False)


def bicubic_upsample(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=(720, 1280), mode='bicubic', align_corners=False)