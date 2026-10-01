"""DiGIT-inspired masked multi-dilated gated residual for frozen VAD features.

    Source: Dotori-HJ/DiGIT, models/digit/modules.py::GatedConv (CVPR 2025).
    Adaptation: one identity-initialized residual in each independent VAD head;
    no supervised proposal decoder or custom CUDA operator. See source manifest
    and licenses/DiGIT-Apache-2.0.txt for provenance and licensing.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DilatedContext(nn.Module):
    def __init__(self, width, levels=4, kernel=7):
        super().__init__()
        if levels < 1 or width % levels or kernel < 1 or kernel % 2 != 1:
            raise ValueError("Dilated context needs divisible width and an odd kernel")
        self.norm = nn.LayerNorm(width)
        self.paths = nn.Linear(width, width * 2)
        part = width // levels
        self.convolutions = nn.ModuleList([
            nn.Conv1d(part, part, kernel, padding=dilation * (kernel // 2),
                      dilation=dilation, groups=part)
            for dilation in range(1, levels + 1)
        ])
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values, valid):
        # Mask before each operation that can mix temporal positions.
        clean = torch.where(valid[..., None], values, 0.)
        local, gate = self.paths(self.norm(clean)).chunk(2, dim=-1)
        local = torch.where(valid[..., None], local, 0.).transpose(1, 2)
        groups = local.chunk(len(self.convolutions), dim=1)
        local = torch.cat([conv(group) for conv, group in zip(self.convolutions, groups)], 1)
        local = torch.where(valid[:, None], local, 0.).transpose(1, 2)
        delta = self.output(local * F.silu(gate))
        return torch.where(valid[..., None], clean + delta, 0.)

