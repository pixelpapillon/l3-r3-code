"""R3 remains intact; new evidence enters BEFORE its original bounded outputs."""

import torch
from dsanet_repair.eventnext.model import NextCorrection
from dsanet_repair.expansion.projection import zero_mean_box
from . import CONFIGS, legacy_options
from .modules import SlotEvidence, ResidualDurationReadout


class R3Transfer(NextCorrection):
    def __init__(self, backbone, args):
        super().__init__(backbone, legacy_options(args))
        self.variant, self.transfer_configuration = args.variant, CONFIGS[args.variant]
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(args.seed + 303).get_state())
            self.slot_evidence = SlotEvidence(args.hidden, args.event_rank,
                                             self.transfer_configuration.circulation)
        self.duration = (ResidualDurationReadout(backbone.num_class - 1)
                         if self.transfer_configuration.duration else None)
        self.extra_evidence = None

    def parameter_partitions(self):
        groups = super().parameter_partitions()
        groups["shared"].extend(self.slot_evidence.parameters())
        if self.duration is not None:
            groups["shared"].extend(self.duration.parameters())
        return groups

    def encode_hidden(self, *args, **kwargs):
        binary, semantic = super().encode_hidden(*args, **kwargs)
        hidden, text, _, widths, valid, bounds = self.event_state
        class_scale = 1 + .25 * self.event_core.class_scale(self.event_core.query(text)).tanh()
        evidence, edges, diagnostics = self.slot_evidence(
            hidden, bounds, widths, valid, self.event_core.last["support"], class_scale)
        if self.duration is not None:
            # Condition only on the frozen anchor's semantic odds. These logits
            # are NOT the Evolve hierarchical joint prior. The transformed field
            # is still only an additive input to R3's bounded correction heads.
            raw = self.last_base_raw
            if self.official_video_chunks:
                raw = raw.reshape(1, -1, raw.shape[-1])
            evidence, info = self.duration(raw[..., 1:], evidence, edges)
            diagnostics.update(info)
        self.extra_evidence = evidence.reshape(*semantic[0].shape[:2], -1)
        self.module_diagnostics.update({k: v.detach() for k, v in diagnostics.items()})
        return binary, semantic

    def corrections(self, binary_hidden, semantic_hidden, valid):
        # Same centering order as RefinedCorrection -> EventCorrection -> R3,
        # including its floating-point order when the new evidence is zero.
        hidden, field = semantic_hidden
        extra = self.extra_evidence
        raw_binary = self.radii[0] * (self.binary_output(binary_hidden).squeeze(-1) + extra.mean(-1))
        self.last_unprojected = torch.where(valid, raw_binary, 0.)
        centered = self.last_unprojected - self.last_unprojected.sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
        centered = torch.where(valid, centered, 0.)
        binary = zero_mean_box(centered, valid, float(self.radii[0]))
        raw_semantic = self.semantic_output(hidden) + self.event_semantic_output(field).squeeze(-1) + extra
        semantic = torch.where(valid[..., None], raw_semantic.tanh() * self.radii[1:], 0.)
        self.module_diagnostics["gauge_centered_std"] = (centered.square().sum() / valid.sum().clamp_min(1)).sqrt()
        self.module_diagnostics["transfer_evidence_abs"] = (
            extra.abs() * valid[..., None]).sum().detach() / (valid.sum() * extra.shape[-1]).clamp_min(1)
        return binary, semantic
