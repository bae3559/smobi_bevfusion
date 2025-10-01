from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser"]


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # print(f"ConvFuser DEBUG: Number of input tensors: {len(inputs)}")
        # for i, inp in enumerate(inputs):
        #     print(f"  Input {i}: shape = {inp.shape}")

        # Get target spatial size from the first tensor (usually camera)
        target_h, target_w = inputs[0].shape[2], inputs[0].shape[3]

        # Resize all inputs to match the target spatial size
        resized_inputs = []
        for inp in inputs:
            if inp.shape[2] != target_h or inp.shape[3] != target_w:
                inp = torch.nn.functional.interpolate(
                    inp, size=(target_h, target_w), mode='bilinear', align_corners=False
                )
            resized_inputs.append(inp)

        return super().forward(torch.cat(resized_inputs, dim=1))
