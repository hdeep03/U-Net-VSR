from typing import Tuple
import torch
import torch.nn.functional as F

def downsample(x: torch.Tensor, target_res: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=target_res, mode='bilinear', align_corners=False)


def bicubic_upsample(x: torch.Tensor, target_res: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=target_res, mode='bicubic', align_corners=False)