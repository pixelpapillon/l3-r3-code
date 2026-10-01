"""Q1-Q6 reuse the anchor, data interface, correction heads and evaluator."""

import torch

from dsanet_repair.expansion.model import StudyCorrection
from dsanet_repair.expansion.projection import zero_mean_box
from . import ALL_CONFIGS
from .attention import EvidenceAttention
from .class_memory import ClassReacquisitionMemory
from .flow import BoundaryFlow
from .state_filter import InnovationFilter, FixedEWMA


class RefinedCorrection(StudyCorrection):
    def __init__(self, backbone, args):
        super().__init__(backbone, "p1-center-full", args.hidden, args.binary_radius,
                         args.semantic_radius, args.prototype_temperature)
        self.variant, self.configuration = args.variant, ALL_CONFIGS[args.variant]
        self.attention = self.flow = self.state_filter = self.class_memory = None
        self.mechanism = None
        # Set only by the official per-video evaluator wrapper. Training batch
        # rows are independent videos/edits and must never share state.
        self.official_video_chunks = False
        # Common adapter initialization and data RNG stay paired across variants.
        with torch.random.fork_rng(devices=[]):
            if self.configuration.attention != "none":
                self.attention = EvidenceAttention(backbone.visual_width, args.hidden,
                    args.attention_heads, self.configuration.attention, args.memory_slots)
            if self.configuration.flow:
                self.flow = BoundaryFlow(args.flow_steps, args.flow_temperature,
                                         phase_shift=self.configuration.edge_phase)
            if self.configuration.state_filter:
                self.state_filter = InnovationFilter()
            if self.configuration.fixed_filter:
                self.state_filter = FixedEWMA(args.fixed_ewma_gain)
            if self.configuration.class_memory:
                self.class_memory = ClassReacquisitionMemory(backbone.visual_width, args.hidden)
            if self.configuration.mechanism != "none":
                from dsanet_repair.frontier.mechanisms import make_mechanism
                self.mechanism = make_mechanism(self.configuration.mechanism,
                                               backbone.visual_width, args.hidden, args)
        self.module_diagnostics = {}

    def parameter_partitions(self):
        groups = super().parameter_partitions()
        for module in (self.attention, self.flow, self.state_filter, self.class_memory, self.mechanism):
            if module is not None:
                groups["shared"].extend(module.parameters())
        return groups

    def encode_hidden(self, inputs, valid, features, context, base, lengths):
        hidden = self.shared_embedding(inputs)
        self.module_diagnostics = {}
        if self.mechanism is not None:
            if self.official_video_chunks:
                rows, steps = valid.shape
                sequence = lambda value: value.reshape(1, rows * steps, *value.shape[2:])
                mechanism_base = (base[0], sequence(base[1]), sequence(base[2]))
                updated, diagnostic = self.mechanism(
                    sequence(hidden), sequence(features), sequence(context),
                    mechanism_base, valid.reshape(1, rows * steps))
                hidden = updated.reshape_as(hidden)
            else:
                hidden, diagnostic = self.mechanism(hidden, features, context, base, valid)
            self.module_diagnostics.update(diagnostic)
        if self.attention is not None:
            hidden, diagnostic = self.attention(hidden, features, context, base, lengths, valid)
            self.module_diagnostics.update(diagnostic)
        if self.class_memory is not None:
            if self.official_video_chunks:
                rows, steps = valid.shape
                sequence = lambda value: value.reshape(1, rows * steps, *value.shape[2:])
                memory_base = (base[0], sequence(base[1]), sequence(base[2]))
                updated, diagnostic = self.class_memory(
                    sequence(hidden), sequence(features), sequence(context),
                    memory_base, valid.reshape(1, rows * steps))
                hidden = updated.reshape_as(hidden)
            else:
                hidden, diagnostic = self.class_memory(hidden, features, context, base, valid)
            self.module_diagnostics.update(diagnostic)
        self.last_hidden = hidden
        return hidden, hidden

    def corrections(self, binary_hidden, semantic_hidden, valid):
        # Same derivative at zero as P1, but no pre-projection tanh saturation.
        raw_binary = self.radii[0] * self.binary_output(binary_hidden).squeeze(-1)
        raw_binary = torch.where(valid, raw_binary, 0.)
        self.last_unprojected = raw_binary
        centered = raw_binary - raw_binary.sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
        centered = torch.where(valid, centered, 0.)
        semantic = self.semantic_output(semantic_hidden).tanh() * self.radii[1:]
        semantic = torch.where(valid[..., None], semantic, 0.)
        combined = torch.cat([centered[..., None], semantic], -1)
        if self.flow is not None:
            combined, diagnostic = self.flow(combined, self.last_features,
                *self.last_base_outputs, valid)
            self.module_diagnostics.update(diagnostic)
        if self.state_filter is not None:
            if self.official_video_chunks:
                rows, steps = valid.shape
                sequence = lambda value: value.reshape(1, rows * steps, *value.shape[2:])
                filtered, diagnostic = self.state_filter(
                    sequence(combined), sequence(self.last_features),
                    *(sequence(value) for value in self.last_base_outputs),
                    valid.reshape(1, rows * steps))
                combined = filtered.reshape_as(combined)
            else:
                combined, diagnostic = self.state_filter(combined, self.last_features,
                    *self.last_base_outputs, valid)
            self.module_diagnostics.update(diagnostic)
        binary = zero_mean_box(combined[..., 0], valid, float(self.radii[0]))
        self.module_diagnostics["gauge_centered_std"] = (
            centered.square().sum() / valid.sum().clamp_min(1)).sqrt()
        return binary, combined[..., 1:]
