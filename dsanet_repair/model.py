"""A3: train-time learning of calibrated intervention-response fields."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .adapter import raw_dsanet_logits


class InterventionResponseHead(nn.Module):
    """Predict class-specific removable evidence from DSANet temporal features.

    A depth-wise temporal mixer preserves local event boundaries, while learned
    class queries give every abnormal category its own response direction.  The
    binary channel is learned jointly as target 0.  Outputs are non-negative,
    matching the semantics "how much evidence disappears after a legal edit".
    """

    def __init__(self, width, targets, dropout=0.1):
        super().__init__()
        if width < 1 or targets < 1:
            raise ValueError("width and targets must be positive")
        self.width = int(width)
        self.targets = int(targets)
        self.norm = nn.LayerNorm(width)
        self.temporal = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1, groups=width),
            nn.GELU(),
            nn.Conv1d(width, width, 1),
            nn.Dropout(dropout),
        )
        self.class_queries = nn.Parameter(torch.empty(targets, width))
        self.logit_gate = nn.Linear(targets, targets, bias=False)
        self.bias = nn.Parameter(torch.zeros(targets))
        nn.init.trunc_normal_(self.class_queries, std=0.02)
        nn.init.eye_(self.logit_gate.weight)

    def forward(self, temporal_features, base_logits):
        if temporal_features.ndim != 3 or base_logits.ndim != 3:
            raise ValueError("Expected [B,T,D] features and [B,T,K] base logits")
        if temporal_features.shape[:2] != base_logits.shape[:2]:
            raise ValueError("Feature and logit timelines do not match")
        if temporal_features.shape[-1] != self.width or base_logits.shape[-1] != self.targets:
            raise ValueError("Response-head channel mismatch")
        h = self.norm(temporal_features)
        h = h + self.temporal(h.transpose(1, 2)).transpose(1, 2)
        semantic = torch.einsum("btd,kd->btk", h, F.normalize(self.class_queries, dim=-1))
        semantic = semantic / math.sqrt(self.width)
        gated = self.logit_gate(base_logits)
        return F.softplus(semantic + 0.25 * gated + self.bias)


class ResponseAugmentedDSANet(nn.Module):
    """Wrap DSANet with a train-only response head without editing upstream.

    The hook taps the output of DSANet's final visual projection, i.e. the same
    representation used by its binary and semantic branches.  At deployment,
    ``export_backbone_state`` drops the auxiliary head and produces an ordinary
    DSANet state dict accepted by the official evaluator.
    """

    def __init__(self, backbone, dropout=0.1):
        super().__init__()
        if not hasattr(backbone, "linear") or not hasattr(backbone, "visual_width"):
            raise ValueError("Backbone does not expose the official DSANet visual projection")
        self.backbone = backbone
        # One binary response plus C-1 abnormal class responses = num_class.
        targets = int(backbone.num_class)
        self.response_head = InterventionResponseHead(backbone.visual_width, targets, dropout)
        self._temporal_features = None
        self._tap = self.backbone.linear.register_forward_hook(self._capture_temporal)

    def _capture_temporal(self, _module, _inputs, output):
        self._temporal_features = output

    def forward(self, visual, padding_mask, text, lengths, dnp_use, scale=10):
        self._temporal_features = None
        output = self.backbone(visual, padding_mask, text, lengths, dnp_use, scale)
        if self._temporal_features is None:
            raise RuntimeError("DSANet temporal feature tap did not fire")
        binary, classes = raw_dsanet_logits(output)
        base = torch.cat([binary.unsqueeze(-1), classes], dim=-1)
        response = self.response_head(self._temporal_features, base)
        return output, response

    def export_backbone_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.backbone.state_dict().items()}

    def close(self):
        self._tap.remove()
