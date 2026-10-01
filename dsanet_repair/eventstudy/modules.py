"""M1 soft category/instance intervals and M2 event-relative ordered features.

All candidate classes are evaluated. Overlap is allowed within/across classes;
there is no test label, test memory, class-exclusive assignment or frame target.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .geometry import interpolate


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


def soft_attention(logits, valid):
    weights = logits.masked_fill(~valid[:, None, None], -1e4).softmax(-1)
    weights = weights * valid[:, None, None]
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)


def signature(path):
    delta = path[..., 1:, :] - path[..., :-1, :]
    before = delta.cumsum(-2) - delta
    area = .5 * (torch.einsum("...si,...sj->...ij", before, delta) -
                 torch.einsum("...si,...sj->...ij", delta, before))
    indices = torch.triu_indices(path.shape[-1], path.shape[-1], 1, device=path.device)
    return delta.sum(-2), area[..., indices[0], indices[1]]


class EventCore(nn.Module):
    def __init__(self, width, hidden, slots=8, rank=8, samples=8,
                 category_support=True, ordered=True):
        super().__init__()
        self.slots, self.rank, self.samples = slots, rank, samples
        self.category_support, self.ordered = category_support, ordered
        self.query = nn.Linear(width, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.slot = nn.Parameter(torch.randn(slots, hidden) / math.sqrt(hidden))
        self.geometry = nn.Linear(2 * hidden, 2)
        nn.init.zeros_(self.geometry.weight)
        nn.init.constant_(self.geometry.bias, -2.)
        with torch.no_grad():
            self.geometry.bias[0] = 0
        self.project = nn.Linear(hidden, rank)
        self.class_scale = nn.Linear(hidden, rank)
        ordered_dim = 3 * rank + rank * (rank - 1) // 2
        dimension = ordered_dim if ordered else 2 * rank
        # Every weight is used: capacity matching uses a wider active MLP,
        # not disconnected zero-filled signature channels in E3.
        middle = max(1, round(hidden * (ordered_dim + hidden + 1) / (dimension + hidden + 1)))
        self.trajectory = nn.Sequential(nn.Linear(dimension, middle), nn.GELU(), nn.Linear(middle, hidden))
        self.confidence = nn.Linear(hidden, 1)
        self.diagnostic_mode = "normal"

    def propose(self, hidden, text, centers, widths, valid):
        q = self.query(text)
        support_q = q if self.category_support else q.mean(0, keepdim=True).expand_as(q)
        queries = support_q[:, None] + self.slot[None]
        logits = torch.einsum("ckh,bth->bckt", queries, self.key(hidden)) / math.sqrt(hidden.shape[-1])
        anchors = (torch.arange(self.slots, device=hidden.device, dtype=hidden.dtype) + .5) / self.slots
        logits = logits - ((centers[:, None, None] - anchors[None, None, :, None]) / .2).square()
        weights = soft_attention(logits, valid)
        pooled = torch.einsum("bckt,bth->bckh", weights, hidden)
        query = queries[None].expand(hidden.shape[0], -1, -1, -1)
        offset, duration = self.geometry(torch.cat([pooled, query], -1)).unbind(-1)
        center = (weights * centers[:, None, None]).sum(-1)
        center = (center + .15 * offset.tanh()).clamp(0, 1)
        floor = widths.masked_fill(~valid, torch.inf).min(-1).values
        floor = torch.where(valid.any(-1), floor, torch.ones_like(floor)).clamp(1e-6, 1)
        duration = floor[:, None, None] + (1 - floor[:, None, None]) * duration.sigmoid()
        start = center * (1 - duration)
        return torch.stack([start, start + duration], -1)

    def forward(self, hidden, text, centers, widths, valid, intervals=None):
        hidden = torch.where(valid[..., None], hidden, 0.)
        q = self.query(text)
        bounds = self.propose(hidden, text, centers, widths, valid) if intervals is None else intervals
        expected = (hidden.shape[0], text.shape[0], self.slots, 2)
        if tuple(bounds.shape) != expected:
            raise ValueError("Locked event coordinate shape mismatch")
        start, end = bounds.unbind(-1)
        softness = .5 / valid.sum(-1).clamp_min(1).to(hidden.dtype)
        support = (torch.sigmoid((centers[:, None, None] - start[..., None]) / softness[:, None, None, None]) -
                   torch.sigmoid((centers[:, None, None] - end[..., None]) / softness[:, None, None, None]))
        support = support * valid[:, None, None]
        mass = support * widths[:, None, None]
        weights = mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)
        z = torch.where(valid[..., None], self.project(hidden).tanh(), 0.)
        scale = 1 + .25 * self.class_scale(q).tanh()
        mean = torch.einsum("bckt,btr->bckr", weights, z) * scale[None, :, None]
        second = torch.einsum("bckt,btr->bckr", weights, z.square()) * scale[None, :, None].square()
        variance = (second - mean.square()).clamp_min(0)
        area_energy = hidden.sum() * 0
        if self.ordered:
            phase = torch.linspace(0, 1, self.samples, device=z.device, dtype=z.dtype)
            positions = start[..., None] + (end - start)[..., None] * phase
            path = interpolate(z, centers, positions, valid.sum(-1)) * scale[None, :, None, None]
            if self.diagnostic_mode == "shuffle-interior":
                # Deterministic, endpoint-preserving permutation: no training RNG consumed.
                order = torch.cat([torch.tensor([0], device=z.device),
                                   torch.arange(self.samples - 2, 0, -1, device=z.device),
                                   torch.tensor([self.samples - 1], device=z.device)])
                path = path.index_select(-2, order)
            first, area = signature(path)
            if self.diagnostic_mode == "first-order":
                area = area * 0
            area_energy = area.square().mean()
            descriptor = torch.cat([mean, variance, first, area], -1)
        else:
            descriptor = torch.cat([mean, variance], -1)
        event = self.trajectory(descriptor)
        evidence = support * self.confidence(event).sigmoid()
        # Independent class fields. The +1 keeps vanishing support vanishing;
        # dividing only by support would broadcast an event into the background.
        field = torch.einsum("bckt,bckh->btch", evidence, event) / (1 + evidence.sum(2).permute(0, 2, 1))[..., None]
        field = torch.where(valid[:, :, None, None], field, 0.)
        active = valid.any(-1).to(hidden.dtype)
        norm = active.sum().clamp_min(1) * text.shape[0] * self.slots
        duration = end - start
        intersection = (torch.minimum(end[..., :, None], end[..., None, :]) -
                        torch.maximum(start[..., :, None], start[..., None, :])).clamp_min(0)
        overlap = intersection / (duration[..., :, None] + duration[..., None, :] - intersection).clamp_min(1e-8)
        off_diagonal = ~torch.eye(self.slots, device=hidden.device, dtype=torch.bool)
        overlap = overlap * off_diagonal * active[:, None, None, None]
        diagnostics = {
            "event_duration_mean": (duration * active[:, None, None]).sum() / norm,
            "event_duration_min": duration[valid.any(-1)].min() if bool(valid.any()) else duration.sum() * 0,
            "event_duration_max": duration[valid.any(-1)].max() if bool(valid.any()) else duration.sum() * 0,
            "event_class_duration_spread": (duration.std(1, unbiased=False) * active[:, None]).sum() / (active.sum().clamp_min(1) * self.slots),
            "event_area_energy": area_energy,
            "event_support_mass": mass.sum() / norm,
            "event_field_abs": field.abs().sum() / (valid.sum().clamp_min(1) * text.shape[0] * hidden.shape[-1]),
            "event_slot_pair_iou": overlap.sum() / (active.sum().clamp_min(1) * text.shape[0] * max(1, self.slots * (self.slots - 1))),
        }
        return field, bounds, diagnostics


class ClassAttentionCore(nn.Module):
    """E5: ordinary category attention, without intervals/signatures/locking."""
    def __init__(self, width, hidden, slots, target_parameters):
        super().__init__()
        self.query, self.key = nn.Linear(width, hidden), nn.Linear(hidden, hidden)
        self.slot = nn.Parameter(torch.randn(slots, hidden) / math.sqrt(hidden))
        self.position = nn.Linear(2, hidden)
        used = parameter_count(self)
        middle = max(1, round((target_parameters - used - hidden) / (2 * hidden + 1)))
        self.content = nn.Sequential(nn.Linear(hidden, middle), nn.GELU(), nn.Linear(middle, hidden))

    def forward(self, hidden, text, centers, widths, valid, intervals=None):
        if intervals is not None:
            raise ValueError("Generic attention has no event coordinates to lock")
        position = torch.stack([centers, widths], -1)
        keys = self.key(hidden) + self.position(position)
        query = self.query(text)[:, None] + self.slot[None]
        weights = soft_attention(torch.einsum("ckh,bth->bckt", query, keys) / math.sqrt(hidden.shape[-1]), valid)
        value = self.content(torch.einsum("bckt,bth->bckh", weights, hidden))
        support = weights * valid.sum(-1)[:, None, None, None].clamp_min(1)
        field = torch.einsum("bckt,bckh->btch", support, value) / (1 + support.sum(2).permute(0, 2, 1))[..., None]
        field = torch.where(valid[:, :, None, None], field, 0.)
        return field, None, {"event_field_abs": field.abs().mean()}
