"""Bounded corrections conditioned on DSANet's pretrained normality geometry."""

import torch
from torch import nn
from torch.nn import functional as F

from dsanet_repair.adapter import raw_dsanet_logits
from dsanet_repair.numerics import decisions_from_features


class NativeCorrection(nn.Module):
    def __init__(self, backbone, hidden=128, binary_radius=.5, semantic_radius=2.,
                 prototype_temperature=.1):
        super().__init__()
        if binary_radius < 0 or semantic_radius < 0 or prototype_temperature <= 0:
            raise ValueError("Invalid correction budgets/temperature")
        self.backbone = backbone.requires_grad_(False).eval()
        self.prototype_temperature = float(prototype_temperature)
        width, channels = backbone.visual_width, backbone.num_class
        self.correction = nn.Sequential(
            nn.LayerNorm(width * 2 + 2), nn.Linear(width * 2 + 2, hidden), nn.GELU(),
            nn.Linear(hidden, channels),
        )
        # Identical inference to the anchor before any optimizer update.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.register_buffer("radii", torch.tensor([binary_radius] + [semantic_radius] * (channels - 1)))
        self.last_base_raw = None
        self.last_correction = None

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def normal_context(self, features, binary, lengths):
        module = self.backbone.video_anomaly_refiner
        # Reuse the TRAINED SGNM prototype aggregator, respecting valid lengths.
        groups = {}
        for index, length in enumerate(lengths.tolist()):
            if length == 0:  # upstream evaluator appends an empty block at exact multiples
                continue
            if not 0 < length <= features.shape[1]:
                raise ValueError("Invalid native correction sequence length")
            count = max(1, int(length * module.normal_selection_ratio))
            indices = binary[index, :length, 0].topk(count, largest=False).indices
            groups.setdefault(count, []).append((index, features[index, indices]))
        result = torch.zeros_like(features)
        for rows in groups.values():
            indices = torch.tensor([row[0] for row in rows], device=features.device)
            selected = torch.stack([row[1] for row in rows])
            prototypes = module.video_prototypes[None].expand(len(rows), -1, -1)
            for block in module.dnp_extractor:
                prototypes = block(prototypes, selected)
            affinity = (F.normalize(features[indices], dim=-1) @
                        F.normalize(prototypes, dim=-1).transpose(1, 2)) / self.prototype_temperature
            result[indices] = affinity.softmax(-1) @ prototypes
        return result

    def forward(self, visual, padding_mask, text, lengths, DNP_use=False, scale=10):
        if DNP_use:
            raise ValueError("Native correction exposes detection outputs only; set DNP_use=False")
        # The pinned official evaluator leaves lengths on CPU, even on CUDA.
        lengths = torch.as_tensor(lengths, device=visual.device, dtype=torch.long)
        with torch.no_grad():
            features = self.backbone.encode_video(visual, padding_mask, lengths)
            base = decisions_from_features(self.backbone, features, text, False, scale)
            context = self.normal_context(features, base[1], lengths)
            probability = base[2][..., 1:].softmax(-1)
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(-1, keepdim=True)
            inputs = torch.cat([F.layer_norm(features, (features.shape[-1],)),
                                F.layer_norm(features - context, (features.shape[-1],)),
                                base[1].sigmoid(), entropy], -1)
            binary, margins = raw_dsanet_logits(base)
            self.last_base_raw = torch.cat([binary[..., None], margins], -1)
        delta = self.correction(inputs).tanh() * self.radii
        valid = torch.arange(visual.shape[1], device=visual.device)[None] < lengths[:, None]
        delta = delta * valid[..., None]
        self.last_correction = delta
        binary = base[1] + delta[..., :1]
        # Keep normal's semantic logit fixed; abnormal margins inherit exact bounds.
        semantic = torch.cat([base[2][..., :1], base[2][..., 1:] + delta[..., 1:]], -1)
        return (base[0], binary, semantic, base[3], base[4])
