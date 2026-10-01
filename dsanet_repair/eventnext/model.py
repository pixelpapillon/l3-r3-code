"""Reuse E3's frozen anchor and timeline; replace only declared R components."""

import copy
import math
import torch
from torch import nn

from dsanet_repair.eventstudy import Configuration as EventConfiguration
from dsanet_repair.eventstudy.model import EventCorrection
from dsanet_repair.expansion.projection import zero_mean_box
from . import CONFIGS
from .modules import CoreEnvelope


class NextCorrection(EventCorrection):
    def __init__(self, backbone, args):
        original = copy.copy(args)
        original.variant = "e3-orderless-events"
        super().__init__(backbone, original)
        self.variant, self.next_configuration = args.variant, CONFIGS[args.variant]
        self.event_configuration = EventConfiguration(ordered=False, locked=self.next_configuration.locked)
        self.existence_temperature = args.existence_temperature
        self.reference_momentum = args.reference_momentum
        classes = backbone.num_class - 1
        if self.next_configuration.morphology:
            with torch.random.fork_rng(devices=[]):
                core = CoreEnvelope(backbone.visual_width, args.hidden, classes, args, self.next_configuration)
            # Exact common E3 weights and identical R2/R3 initialization.
            core.load_state_dict(self.event_core.state_dict(), strict=False)
            self.event_core = core
        self.register_buffer("readout_reference", torch.zeros(classes))
        self.register_buffer("readout_count", torch.zeros(classes, dtype=torch.long))
        self.primary_observation = None
        self.last_field = None

    def encode_hidden(self, *args, **kwargs):
        binary, semantic = super().encode_hidden(*args, **kwargs)
        self.last_field = semantic[1]
        return binary, semantic

    def corrections(self, binary_hidden, semantic_hidden, valid):
        binary, semantic = super().corrections(binary_hidden, semantic_hidden, valid)
        if self.next_configuration.existence:
            hidden, field = semantic_hidden
            contribution = torch.einsum("btch,oh->btc", field, self.binary_output.weight)
            tau = self.existence_temperature
            reference = self.readout_reference.detach().clone()
            readout = tau * (torch.logsumexp((contribution - reference) / tau, -1) - math.log(len(reference)))
            # No-event fields must contribute zero, even after reference updates.
            null = tau * (torch.logsumexp(-reference / tau, -1) - math.log(len(reference)))
            raw = self.radii[0] * (self.binary_output(hidden).squeeze(-1) + readout - null)
            self.last_unprojected = torch.where(valid, raw, 0.)
            binary = zero_mean_box(self.last_unprojected, valid, float(self.radii[0]))
            self.module_diagnostics["existence_minus_mean_abs"] = (
                (readout - null - contribution.mean(-1)).abs() * valid).sum() / valid.sum().clamp_min(1)
        return binary, semantic

    def observe_primary(self, count, labels, lengths):
        """Called from the training loss only; updates commit AFTER backward."""
        if not self.training or not (self.next_configuration.assembly or self.next_configuration.existence):
            return
        affinity = self.event_core.last["affinity"][:count].detach() if isinstance(self.event_core, CoreEnvelope) else None
        score = torch.einsum("btch,oh->btc", self.last_field[:count].detach(),
                             self.binary_output.weight.detach())
        self.primary_observation = (affinity, score, labels.detach(), lengths.detach())

    @torch.no_grad()
    def commit_references(self):
        if self.primary_observation is None:
            return
        affinity, score, labels, lengths = self.primary_observation
        self.primary_observation = None
        valid = torch.arange(score.shape[1], device=score.device)[None] < lengths[:, None]
        for c in range(score.shape[-1]):
            mask = valid & (labels[:, c + 1] == 0)[:, None]
            if not bool(mask.any()):
                continue
            if self.next_configuration.existence:
                value = score[..., c][mask].mean()
                rate = self.reference_momentum if self.readout_count[c] > 0 else 0.
                self.readout_reference[c].mul_(rate).add_(value * (1 - rate))
                self.readout_count[c] += mask.sum()
            if self.next_configuration.assembly:
                value = affinity[:, c][mask].mean()
                rate = self.reference_momentum if self.event_core.negative_count[c] > 0 else 0.
                self.event_core.negative_reference[c].mul_(rate).add_(value * (1 - rate))
                self.event_core.negative_count[c] += mask.sum()
