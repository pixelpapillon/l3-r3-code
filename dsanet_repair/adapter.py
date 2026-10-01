"""Adapters that expose DSANet logits to the A1/A2 intervention engine.

The adapter deliberately works in raw-logit space.  DSANet's official test
code combines a binary sigmoid and a class softmax; using those probabilities
inside the intervention search would make response magnitudes temperature and
normalization dependent.  We instead expose the binary logit and one
normal-referenced margin for every abnormal class.
"""

from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class ProbeConfig:
    topk_divisor: int = 16

    def __post_init__(self):
        if self.topk_divisor < 1:
            raise ValueError("topk_divisor must be positive")


def raw_dsanet_logits(output):
    """Return binary logits and abnormal-vs-normal class margins.

    Args:
        output: the tuple returned by the official DSANet forward method.

    Returns:
        ``(binary, classes)`` with shapes ``[B,T]`` and ``[B,T,C-1]``.
    """
    if not isinstance(output, (tuple, list)) or len(output) < 3:
        raise ValueError("Expected an official DSANet output tuple")
    binary = output[1]
    class_logits = output[2]
    if binary.ndim != 3 or binary.shape[-1] != 1 or class_logits.ndim != 3:
        raise ValueError("Unexpected DSANet logit shapes")
    if class_logits.shape[:2] != binary.shape[:2] or class_logits.shape[-1] < 2:
        raise ValueError("DSANet binary/class timelines are incompatible")
    # Margin to the explicit normal class preserves category meaning while
    # avoiding a probability-simplex response artifact.
    margins = class_logits[..., 1:] - class_logits[..., :1]
    return binary.squeeze(-1), margins


class DSANetProbeAdapter(nn.Module):
    """Make an official DSANet checkpoint consumable by SelectiveRepair.

    Inputs are fixed-grid CLIP features, either ``[T,D]`` or ``[B,T,D]``.
    Repair evidence is generated with a frozen teacher; no test labels are
    accepted by this interface.
    """

    def __init__(self, backbone, prompt_text, dnp_use=True, topk_divisor=16):
        super().__init__()
        self.backbone = backbone
        self.prompt_text = tuple(prompt_text)
        self.dnp_use = bool(dnp_use)
        self.config = ProbeConfig(topk_divisor)

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def forward(self, features, lengths=None):
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError("DSANet features must have shape [T,D] or [B,T,D]")
        batch, steps, _ = features.shape
        expected = getattr(self.backbone, "visual_length", steps)
        if steps != expected:
            raise ValueError(f"DSANet expects {expected} steps, got {steps}")
        if lengths is None:
            lengths = torch.full((batch,), steps, dtype=torch.long, device=features.device)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device=features.device)
        if lengths.shape != (batch,) or bool((lengths < 1).any()) or bool((lengths > steps).any()):
            raise ValueError("Invalid DSANet sequence lengths")
        output = self.backbone(features, None, list(self.prompt_text), lengths, self.dnp_use)
        binary, classes = raw_dsanet_logits(output)
        return {"binary": binary, "classes": classes}
