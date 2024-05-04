import torch
import torch.nn.functional as F

def downsample(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    return F.interpolate(x, size=(720//factor, 1280//factor), mode='bilinear', align_corners=False)


def bicubic_upsample(x: torch.Tensor, factor: int=1) -> torch.Tensor:
    return F.interpolate(x, size=(720*factor, 1280*factor), mode='bicubic', align_corners=False)

