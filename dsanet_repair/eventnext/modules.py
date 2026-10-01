"""Shared core/envelope representation; optional set assembly and relation graph.

No class-name branches, frame labels, test adaptation, T-by-T attention, or
temporal Python recurrence. Assembly unrolls the small fixed slot count only.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from dsanet_repair.eventstudy.modules import EventCore


def interval_support(bounds, centers, widths, valid):
    start, end = bounds.unbind(-1)
    # Same original-coordinate contract as E3. No padded values enter pooling.
    softness = .5 / valid.sum(-1).clamp_min(1).to(centers.dtype)
    time = centers[:, None, None]
    support = (torch.sigmoid((time - start[..., None]) / softness[:, None, None, None]) -
               torch.sigmoid((time - end[..., None]) / softness[:, None, None, None]))
    return support * valid[:, None, None]


def normalize_mass(mass):
    return mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)


def marginal_assembly(support, signed_evidence, widths, cost, temperature, training):
    """Hard forward greedy gains, straight-through soft decisions in training.

    Same-class duplicate coverage earns no marginal benefit. Negative evidence
    makes this a heuristic, NOT a monotone-submodular approximation guarantee.
    A null choice terminates assembly; it is not a dummy trainable parameter.
    """
    batch, classes, slots, steps = support.shape
    coverage = support.new_zeros(batch, classes, steps)
    selected = support.new_zeros(batch, classes, slots)
    available = torch.ones_like(selected, dtype=torch.bool)
    active = torch.ones(batch, classes, dtype=torch.bool, device=support.device)
    total_gain = support.new_zeros(batch, classes)
    for _ in range(slots):
        increment = (support - coverage[:, :, None]).clamp_min(0)
        gain = (increment * signed_evidence[:, :, None] * widths[:, None, None]).sum(-1) - cost
        gain = gain.masked_fill(~available, -1e4)
        options = torch.cat([gain, gain.new_zeros(batch, classes, 1)], -1)
        hard = F.one_hot(options.argmax(-1), slots + 1).to(support.dtype)
        choice = hard
        if training:
            soft = (options / temperature).softmax(-1)
            choice = hard + soft - soft.detach()
        picked = choice[..., :slots] * active[..., None]
        selected = selected + picked
        candidate = (picked[..., None] * support).sum(2)
        coverage = torch.maximum(coverage, candidate)
        total_gain = total_gain + (picked * gain).sum(-1)
        available = available & ~(hard[..., :slots].bool() & active[..., None])
        active = active & ~hard[..., -1].bool()
    return selected, coverage, total_gain


class CoreEnvelope(EventCore):
    def __init__(self, width, hidden, classes, args, configuration):
        super().__init__(width, hidden, args.event_slots, args.event_rank, args.event_samples, ordered=False)
        self.configuration = configuration
        self.assembly_cost = args.assembly_cost
        self.assembly_temperature = args.assembly_temperature
        self.register_buffer("negative_reference", torch.zeros(classes))
        self.register_buffer("negative_count", torch.zeros(classes, dtype=torch.long))
        # Add morphology to E3's active mean/variance path, not a zero-filled
        # signature substitute. Global output heads still initialize to zero.
        self.morphology = nn.Sequential(nn.Linear(4 * self.rank + 3, hidden), nn.GELU(),
                                       nn.Linear(hidden, hidden))
        if configuration.interaction:
            self.relation = nn.Sequential(nn.Linear(2 * hidden + 2, hidden), nn.GELU(), nn.Linear(hidden, 2))
            self.relation_message = nn.Linear(2 * hidden, hidden, bias=False)
        else:
            self.relation = self.relation_message = None
        self.last = None

    def describe(self, hidden, text, centers, widths, valid, bounds):
        hidden = torch.where(valid[..., None], hidden, 0.)
        q = self.query(text)
        affinity = torch.einsum("bth,ch->bct", F.normalize(hidden, dim=-1), F.normalize(q, dim=-1))
        affinity = torch.where(valid[:, None], affinity, 0.)
        support = interval_support(bounds, centers, widths, valid)
        mass = support * widths[:, None, None]
        envelope = normalize_mass(mass)
        density = torch.exp((affinity / .2).clamp(-8, 8))[:, :, None]
        core = normalize_mass(mass * density)
        z = self.project(hidden).tanh()
        z = torch.where(valid[..., None], z, 0.)
        scale = 1 + .25 * self.class_scale(q).tanh()

        def pool(weights):
            return torch.einsum("bckt,btr->bckr", weights, z) * scale[None, :, None]

        mean = pool(envelope)
        second = torch.einsum("bckt,btr->bckr", envelope, z.square()) * scale[None, :, None].square()
        variance = (second - mean.square()).clamp_min(0)
        start, end = bounds.unbind(-1)
        progress = ((centers[:, None, None] - start[..., None]) /
                    (end - start).clamp_min(1e-6)[..., None]).clamp(0, 1)
        early = normalize_mass(mass * (1 - progress))
        late = normalize_mass(mass * progress)
        duration = end - start
        left = torch.stack([(start - duration * .5).clamp_min(0), start], -1)
        right = torch.stack([end, (end + duration * .5).clamp_max(1)], -1)
        surrounding = interval_support(left, centers, widths, valid) + interval_support(right, centers, widths, valid)
        surrounding = surrounding * widths[:, None, None]
        surrounding_mean = pool(normalize_mass(surrounding))
        has_context = surrounding.sum(-1) > 1e-6
        contrast = torch.where(has_context[..., None], mean - surrounding_mean, 0.)
        # Probability masses use native widths, so unequal training bins do not
        # masquerade as temporal concentration. Relative effective width <= 1.
        effective = 1 / (core.square() / widths[:, None, None].clamp_min(1e-8)).sum(-1).clamp_min(1e-8)
        concentration = (effective / mass.sum(-1).clamp_min(1e-8)).clamp(0, 1)
        center_of_evidence = (core * progress).sum(-1)
        extras = torch.cat([pool(core) - mean, pool(early) - mean, pool(late) - mean, contrast,
                            concentration[..., None], center_of_evidence[..., None],
                            has_context.to(mean.dtype)[..., None]], -1)
        nodes = self.trajectory(torch.cat([mean, variance], -1)) + self.morphology(extras)
        nodes = nodes * valid.any(-1)[:, None, None, None]
        return nodes, support, core, affinity, concentration

    def relations(self, nodes, bounds):
        left, right = nodes.unsqueeze(3), nodes.unsqueeze(2)
        start, end = bounds.unbind(-1)
        intersection = (torch.minimum(end.unsqueeze(-1), end.unsqueeze(-2)) -
                        torch.maximum(start.unsqueeze(-1), start.unsqueeze(-2))).clamp_min(0)
        duration = end - start
        union = duration.unsqueeze(-1) + duration.unsqueeze(-2) - intersection
        iou = intersection / union.clamp_min(1e-8)
        center = (start + end) * .5
        gap = (center.unsqueeze(-1) - center.unsqueeze(-2)).abs()
        descriptor = torch.cat([left + right, (left - right).abs(), iou[..., None], gap[..., None]], -1)
        return self.relation(descriptor)

    def pair_prediction(self, hidden, text, centers, widths, valid, bounds):
        if self.relation is None or bounds.shape[-2:] != (2, 2):
            raise ValueError("Interaction supervision needs exactly two query regions")
        bounds = bounds[:, None].expand(-1, len(text), -1, -1)
        nodes, *_ = self.describe(hidden, text, centers, widths, valid, bounds)
        return self.relations(nodes, bounds)[:, :, 0, 1]

    def forward(self, hidden, text, centers, widths, valid, intervals=None):
        hidden = torch.where(valid[..., None], hidden, 0.)
        bounds = self.propose(hidden, text, centers, widths, valid) if intervals is None else intervals
        if bounds.shape != (hidden.shape[0], len(text), self.slots, 2):
            raise ValueError("Event coordinate shape mismatch")
        nodes, support, core, affinity, concentration = self.describe(hidden, text, centers, widths, valid, bounds)
        diag = {}
        if self.relation is not None:
            relation = self.relations(nodes, bounds)
            mask = ~torch.eye(self.slots, device=hidden.device, dtype=torch.bool)
            signed = relation.mean(-1).tanh() * mask
            messages = []
            for weight in (signed.clamp_min(0), (-signed).clamp_min(0)):
                messages.append(torch.einsum("bcij,bcjh->bcih", weight, nodes) /
                                (1 + weight.sum(-1, keepdim=True)))
            nodes = nodes + self.relation_message(torch.cat(messages, -1))
            diag["relation_abs"] = signed.abs().mean()
        selection = torch.ones_like(bounds[..., 0])
        if self.configuration.assembly:
            signed = ((affinity - self.negative_reference[None, :, None]) / .2).tanh()
            selection, _coverage, gain = marginal_assembly(
                support, signed, widths, self.assembly_cost, self.assembly_temperature, self.training)
            diag.update(assembly_selected=selection.sum(-1).mean(), assembly_gain=gain.mean())
        evidence = support * self.confidence(nodes).sigmoid() * selection[..., None]
        field = torch.einsum("bckt,bckh->btch", evidence, nodes) / (1 + evidence.sum(2).permute(0, 2, 1))[..., None]
        field = torch.where(valid[:, :, None, None], field, 0.)
        self.last = dict(nodes=nodes, bounds=bounds, core=core, support=support, affinity=affinity, field=field,
                         selection=selection, concentration=concentration)
        duration = bounds[..., 1] - bounds[..., 0]
        active = valid.any(-1).to(hidden.dtype)
        norm = (active.sum() * len(text) * self.slots).clamp_min(1)
        end, start = bounds[..., 1], bounds[..., 0]
        inter = (torch.minimum(end[..., :, None], end[..., None, :]) -
                 torch.maximum(start[..., :, None], start[..., None, :])).clamp_min(0)
        pair_iou = inter / (duration[..., :, None] + duration[..., None, :] - inter).clamp_min(1e-8)
        off = ~torch.eye(self.slots, device=hidden.device, dtype=torch.bool)
        diag.update(event_duration_mean=(duration * active[:, None, None]).sum() / norm,
                    event_slot_pair_iou=(pair_iou * off * active[:, None, None, None]).sum() /
                    (active.sum() * len(text) * max(1, self.slots * (self.slots - 1))).clamp_min(1),
                    core_effective_fraction=(concentration * active[:, None, None]).sum() / norm,
                    event_field_abs=field.abs().sum() / (valid.sum() * len(text) * hidden.shape[-1]).clamp_min(1))
        return field, bounds, diag
