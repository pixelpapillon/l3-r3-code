"""Six adapters using unchanged DSANet inference and its trained SGNM context."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from dsanet_repair.adapter import raw_dsanet_logits
from dsanet_repair.numerics import decisions_from_features
from dsanet_repair.revision.model import NativeCorrection
from . import VARIANTS, SPLIT_VARIANTS
from .projection import zero_mean_box
from .temporal import DilatedContext


class StudyCorrection(NativeCorrection):
    def __init__(self, backbone, variant, hidden=128, binary_radius=.5, semantic_radius=2.,
                 prototype_temperature=.1, temporal_levels=4, temporal_kernel=7):
        if variant not in VARIANTS or binary_radius <= 0 or semantic_radius <= 0:
            raise ValueError("Unknown configuration or nonpositive correction radius")
        super().__init__(backbone, hidden, binary_radius, semantic_radius, prototype_temperature)
        self.variant = variant
        self.split = variant in SPLIT_VARIANTS
        # Reuse the exact initialized R2 hidden mapping. Only task sharing changes.
        embedding, output = self.correction[:3], self.correction[3]
        # Model construction must not perturb the data/augmentation RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.binary_output = nn.Linear(hidden, 1)
            self.semantic_output = nn.Linear(hidden, backbone.num_class - 1)
        with torch.no_grad():
            self.binary_output.weight.copy_(output.weight[:1])
            self.binary_output.bias.copy_(output.bias[:1])
            self.semantic_output.weight.copy_(output.weight[1:])
            self.semantic_output.bias.copy_(output.bias[1:])
        if self.split:
            self.binary_embedding = copy.deepcopy(embedding)
            self.semantic_embedding = embedding
        else:
            self.shared_embedding = embedding
        del self.correction
        self.binary_temporal = self.semantic_temporal = None
        if variant == "p4-dilated-context":
            with torch.random.fork_rng(devices=[]):
                self.binary_temporal = DilatedContext(hidden, temporal_levels, temporal_kernel)
                self.semantic_temporal = copy.deepcopy(self.binary_temporal)
        self.last_hidden = None
        self.last_features = None
        self.last_base_outputs = None
        self.last_unprojected = None
        self.last_outputs = None

    def encode_hidden(self, inputs, valid, features, context, base, lengths):
        """Shared forward hook; P1-P6 retain their original computations."""
        if self.split:
            binary_hidden, semantic_hidden = self.binary_embedding(inputs), self.semantic_embedding(inputs)
            if self.binary_temporal is not None:
                binary_hidden = self.binary_temporal(binary_hidden, valid)
                semantic_hidden = self.semantic_temporal(semantic_hidden, valid)
            self.last_hidden = None
        else:
            binary_hidden = semantic_hidden = self.shared_embedding(inputs)
            self.last_hidden = binary_hidden
        return binary_hidden, semantic_hidden

    def corrections(self, binary_hidden, semantic_hidden, valid):
        raw_binary = self.binary_output(binary_hidden).squeeze(-1).tanh() * self.radii[0]
        self.last_unprojected = torch.where(valid, raw_binary, 0.)
        binary_delta = zero_mean_box(raw_binary, valid, float(self.radii[0]))
        semantic_delta = self.semantic_output(semantic_hidden).tanh() * self.radii[1:]
        return binary_delta, torch.where(valid[..., None], semantic_delta, 0.)

    def parameter_partitions(self):
        def parameters(*modules):
            return [p for module in modules if module is not None for p in module.parameters() if p.requires_grad]
        if self.split:
            return {
                "binary": parameters(self.binary_embedding, self.binary_temporal, self.binary_output),
                "semantic": parameters(self.semantic_embedding, self.semantic_temporal, self.semantic_output),
            }
        return {"shared": parameters(self.shared_embedding),
                "binary": parameters(self.binary_output), "semantic": parameters(self.semantic_output)}

    def forward(self, visual, padding_mask, text, lengths, DNP_use=False, scale=10):
        if DNP_use:
            raise ValueError("The expansion modifies detection outputs; DNP_use must be False")
        lengths = torch.as_tensor(lengths, device=visual.device, dtype=torch.long)
        if lengths.shape != visual.shape[:1] or bool(((lengths < 0) | (lengths > visual.shape[1])).any()):
            raise ValueError("Invalid sequence lengths")
        valid = torch.arange(visual.shape[1], device=visual.device)[None] < lengths[:, None]
        with torch.no_grad():
            features = self.backbone.encode_video(visual, padding_mask, lengths)
            base = decisions_from_features(self.backbone, features, text, False, scale)
            context = self.normal_context(features, base[1], lengths)
            probability = base[2][..., 1:].softmax(-1)
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(-1, keepdim=True)
            inputs = torch.cat([F.layer_norm(features, (features.shape[-1],)),
                                F.layer_norm(features - context, (features.shape[-1],)),
                                base[1].sigmoid(), entropy], -1)
            inputs = torch.where(valid[..., None], inputs, 0.)
            binary, margins = raw_dsanet_logits(base)
            self.last_base_raw = torch.cat([binary[..., None], margins], -1)
            self.last_features = torch.where(valid[..., None], F.normalize(features - context, dim=-1), 0.)
            self.last_base_outputs = (base[1], base[2])
        binary_hidden, semantic_hidden = self.encode_hidden(inputs, valid, features, context, base, lengths)
        binary_delta, semantic_delta = self.corrections(binary_hidden, semantic_hidden, valid)
        self.last_correction = torch.cat([binary_delta[..., None], semantic_delta], -1)
        binary = base[1] + binary_delta[..., None]
        semantic = torch.cat([base[2][..., :1], base[2][..., 1:] + semantic_delta], -1)
        self.last_outputs = (base[0], binary, semantic, base[3], base[4])
        return self.last_outputs
