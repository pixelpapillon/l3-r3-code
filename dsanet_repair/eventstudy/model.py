"""Bounded, zero-init event adapter on the existing frozen DSANet anchor."""

import copy
import torch
from torch import nn

from dsanet_repair.refinement.model import RefinedCorrection
from . import CONFIGS
from .geometry import timeline
from .modules import EventCore, ClassAttentionCore, parameter_count


class EventCorrection(RefinedCorrection):
    def __init__(self, backbone, args):
        legacy = copy.copy(args)
        legacy.variant = "c1-long-q1"
        super().__init__(backbone, legacy)
        self.variant = args.variant
        self.event_configuration = CONFIGS[args.variant]
        self.event_core = self.event_semantic_output = None
        self.timeline_edges = None
        # Set only by the event evaluator, from the dataset's unpadded length.
        # Do not infer padding from zero-valued features (zeros can be real).
        self.official_feature_length = None
        self.event_state = None
        self.training_step, self.steps_per_epoch = 0, 1
        self.last_locked_intervals = None
        if self.event_configuration.events:
            with torch.random.fork_rng(devices=[]):
                initial_rng = torch.get_rng_state()
                full = EventCore(backbone.visual_width, args.hidden, args.event_slots,
                                 args.event_rank, args.event_samples)
                if self.event_configuration.generic:
                    self.event_core = ClassAttentionCore(backbone.visual_width, args.hidden,
                                                        args.event_slots, parameter_count(full))
                elif self.event_configuration.ordered:
                    self.event_core = full
                    self.event_core.category_support = self.event_configuration.category_support
                else:
                    torch.set_rng_state(initial_rng)
                    self.event_core = EventCore(backbone.visual_width, args.hidden, args.event_slots,
                        args.event_rank, args.event_samples, self.event_configuration.category_support,
                        self.event_configuration.ordered)
                self.event_semantic_output = nn.Linear(args.hidden, 1, bias=False)
                nn.init.zeros_(self.event_semantic_output.weight)

    def parameter_partitions(self):
        groups = super().parameter_partitions()
        if self.event_core is not None:
            groups["shared"].extend(self.event_core.parameters())
            groups["semantic"].extend(self.event_semantic_output.parameters())
        return groups

    def encode_hidden(self, inputs, valid, features, context, base, lengths):
        if self.event_core is None:
            return super().encode_hidden(inputs, valid, features, context, base, lengths)
        hidden = self.shared_embedding(inputs)
        rows, steps, width = hidden.shape
        if self.official_video_chunks:
            if self.timeline_edges is not None:
                raise ValueError("Training edges cannot be reused in official per-video evaluation")
            hidden, valid = hidden.reshape(1, rows * steps, width), valid.reshape(1, -1)
            if self.official_feature_length is not None:
                actual = int(self.official_feature_length)
                if not 0 < actual <= rows * steps:
                    raise ValueError("Invalid original feature length for event timeline")
                expected = torch.arange(rows * steps, device=valid.device)[None] < actual
                if bool((expected & ~valid).any()):
                    raise ValueError("Official mask excludes real event features")
                # Upstream's exact-multiple branch marks an extra empty chunk
                # valid. Correct ONLY cross-chunk event support; leave the
                # frozen anchor, its inputs and the official score code intact.
                valid = expected
            # Official test input is the uncompressed feature sequence, padded
            # only at its end (including a possible extra empty chunk).
            if not torch.equal(valid, torch.arange(valid.shape[1], device=valid.device)[None] < valid.sum(1)[:, None]):
                raise ValueError("Non-prefix test chunk padding is unsupported")
            centers, widths, _ = timeline(valid.sum(1), rows * steps, hidden.dtype)
        else:
            centers, widths, _ = timeline(lengths, steps, hidden.dtype, self.timeline_edges)
        text = base[0][1:]
        field, bounds, diagnostics = self.event_core(hidden, text, centers, widths, valid)
        self.module_diagnostics = {name: value.detach() for name, value in diagnostics.items()}
        self.event_state = (hidden, text, centers, widths, valid, bounds)
        self.last_hidden = hidden
        binary = hidden + field.mean(2)
        if self.official_video_chunks:
            binary = binary.reshape(rows, steps, width)
            hidden = hidden.reshape(rows, steps, width)
            field = field.reshape(rows, steps, field.shape[2], width)
        return binary, (hidden, field)

    def corrections(self, binary_hidden, semantic_hidden, valid):
        if self.event_core is None:
            return super().corrections(binary_hidden, semantic_hidden, valid)
        hidden, field = semantic_hidden
        binary, _ = super().corrections(binary_hidden, hidden, valid)
        raw = self.semantic_output(hidden) + self.event_semantic_output(field).squeeze(-1)
        semantic = torch.where(valid[..., None], raw.tanh() * self.radii[1:], 0.)
        return binary, semantic

    def locked_raw(self, originals, queries):
        """Reuse anchor/features. Keep original event geometry WITH gradients.

        Edited event content/confidence is recomputed; proposals cannot relocate.
        The ordinary forward's state/diagnostics remain the training log source.
        """
        if not self.event_configuration.locked or self.event_state is None or self.official_video_chunks:
            raise ValueError("Locked responses require a supported training event configuration")
        hidden, text, centers, widths, valid, bounds = self.event_state
        if hidden.shape[0] != originals * (queries + 1):
            raise ValueError("Original/edit packing mismatch")
        locked = bounds[:originals].repeat_interleave(queries, 0)
        self.last_locked_intervals = locked
        field, _, _ = self.event_core(hidden[originals:], text, centers[originals:],
                                     widths[originals:], valid[originals:], intervals=locked)
        before, diagnostics = self.last_unprojected, dict(self.module_diagnostics)
        try:
            binary, semantic = self.corrections(hidden[originals:] + field.mean(2),
                                               (hidden[originals:], field), valid[originals:])
        finally:
            self.last_unprojected, self.module_diagnostics = before, diagnostics
        return self.last_base_raw[originals:] + torch.cat([binary[..., None], semantic], -1)
