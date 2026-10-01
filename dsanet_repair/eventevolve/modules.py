"""Independent implementations, not copies of external paper repositories.

Haar contrasts and second-order path areas are established mathematics. The
research hypothesis is their event-evidence/readout interface, not their invention.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from dsanet_repair.eventrewrite.modules import MorphologyEvidence, SignedComposition


def subinterval_weights(bounds, edges, pieces=8):
    """Exact bin-overlap integrals; valid even with nonuniform bins and padding."""
    nodes = torch.linspace(0, 1, pieces + 1, device=edges.device, dtype=edges.dtype)
    ends = bounds[:, :1] + (bounds[:, 1:] - bounds[:, :1]) * nodes
    overlap = (torch.minimum(edges[:, None, None, 1:], ends[None, :, 1:, None]) -
               torch.maximum(edges[:, None, None, :-1], ends[None, :, :-1, None])).clamp_min(0)
    return overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-12)


def subinterval_path(bounds, edges, h, pieces=8):
    """Shared O(TD+KSD) prefix-integration path, not two dense KST-by-TD products."""
    nodes = torch.linspace(0, 1, pieces + 1, device=edges.device, dtype=edges.dtype)
    ends = bounds[:, :1] + (bounds[:, 1:] - bounds[:, :1]) * nodes
    ends = ends[None].expand(len(h), -1, -1)
    integral = prefix_at(h, edges, ends)
    return integral.diff(dim=2) / ends.diff(dim=2)[..., None].clamp_min(1e-12)


def haar_basis(pieces=8):
    if pieces < 2 or pieces & (pieces - 1):
        raise ValueError("Haar piece count must be a power of two")
    rows = []
    size = pieces
    while size >= 2:
        for start in range(0, pieces, size):
            row = torch.zeros(pieces)
            row[start:start + size // 2] = 1 / math.sqrt(size)
            row[start + size // 2:start + size] = -1 / math.sqrt(size)
            rows.append(row)
        size //= 2
    return torch.stack(rows)


class DetailMorphology(MorphologyEvidence):
    """M1 upgrade: keep R3-derived contrasts, add zero-DC local detail synthesis.

The base envelope head can change category mass. The separate signed Haar field
resolves within-interval structure and has zero time integral before softmax.
That does NOT imply conservation of the final probability marginal.
"""

    def __init__(self, width, hidden, classes, pieces=8):
        super().__init__(width, hidden, classes)
        self.register_buffer("haar", haar_basis(pieces))
        self.detail_project = nn.Linear(width, hidden, bias=False)
        self.detail_head = nn.Linear(hidden, classes, bias=False)
        nn.init.zeros_(self.detail_head.weight)

    def with_details(self, h, weights, support, subweights, widths, path):
        tokens, coarse, unary = super().forward(h, weights, support)
        coefficients = torch.einsum("hs,bksd->bkhd", self.haar, path)
        scores = self.detail_head(self.detail_project(coefficients).tanh())
        # Evaluate piecewise-constant Haar functions on native bins. Convert
        # integration weights into coverage using each equal subinterval's mass.
        interval_mass = (support * widths[:, None]).sum(-1)
        coverage = subweights * (interval_mass / self.haar.shape[1])[:, :, None, None]
        coverage = coverage / widths[:, None, None].clamp_min(1e-12)
        wave = torch.einsum("hs,bkst->bkht", self.haar, coverage)
        detail = torch.einsum("bkht,bkhc->btc", wave, scores)
        # Fixed dictionary normalization, NOT a data-dependent attention gate.
        detail = detail / support.sum(1).amax(-1).clamp_min(1)[:, None, None]
        w = widths / widths.sum(-1, keepdim=True)
        detail = detail - (w[..., None] * detail).sum(1, keepdim=True)
        detail = detail * (widths > 0)[..., None]
        return tokens, coarse + detail, unary, {
            "detail_rms": ((detail.square().mean(-1) * w).sum(-1).mean()).sqrt(),
            "detail_dc_error": (w[..., None] * detail).sum(1).abs().amax(),
        }


def circulation(path):
    r"""Antisymmetric level-two signature in paired coordinates [B,K,S,R,2].

For increments (dx,dy), A = 1/2 sum_i (sum_{j<i}dx_j dy_i - dy_j dx_i).
Divide by 1 + TV_x*TV_y: bounded, no singular scale normalization at static paths.
Translation invariant, zero on straight paths, sign reversal under time reversal.
"""
    delta = path.diff(dim=-3)
    before = delta.cumsum(-3) - delta
    area = .5 * (before[..., 0] * delta[..., 1] - before[..., 1] * delta[..., 0]).sum(-2)
    variation = delta.abs().sum(-3)
    return area / (1 + variation[..., 0] * variation[..., 1])


class OrderedComposition(SignedComposition):
    """M2 upgrade: symmetric cross-interval pairs plus signed within-path area.

No intervention teacher or claim of causal effect. E14 retains |area| with the
same parameters; only orientation in THIS new branch is removed.
"""

    def __init__(self, width, classes, rank=8, order_blind=False):
        super().__init__(width, classes, rank)
        self.order_blind = order_blind
        self.path_project = nn.Linear(width, 2 * rank, bias=False)
        self.area_head = nn.Linear(rank, classes, bias=False)
        nn.init.zeros_(self.area_head.weight)

    def with_circulation(self, tokens, support, pairs, path):
        field, potentials, render = super().forward(tokens, support, pairs)
        projected = self.path_project(path).reshape(*path.shape[:-1], self.rank, 2)
        area = circulation(projected)
        used = area.abs() if self.order_blind else area
        local = self.area_head(used)
        local_render = support.transpose(1, 2)
        local_render = local_render / local_render.sum(-1, keepdim=True).clamp_min(1e-12)
        ordered = local_render @ local
        return field + ordered, potentials, render, {
            "circulation_rms": area.square().mean().sqrt(),
            "ordered_field_rms": ordered.square().mean().sqrt(),
        }


def prefix_at(values, edges, locations):
    """Integral of a piecewise-constant native-bin field up to arbitrary x.

No T-by-T attention or per-frame Python loop. Padded edges at 1 are supported.
Gradients flow to values, not to fixed geometry/searchsorted indices.
"""
    mass = values * edges.diff(dim=-1)[..., None]
    prefix = F.pad(mass.cumsum(1), (0, 0, 1, 0))
    flat = locations.reshape(values.shape[0], -1).contiguous()
    index = torch.searchsorted(edges.contiguous(), flat, right=True).sub(1).clamp(0, values.shape[1] - 1)
    gather = index[..., None].expand(-1, -1, values.shape[-1])
    result = prefix.gather(1, gather) + values.gather(1, gather) * (flat - edges.gather(1, index))[..., None]
    return result.reshape(*locations.shape, values.shape[-1])


class DurationMarginalReadout(nn.Module):
    r"""M3: reference-conditioned log-partition difference over duration hypotheses.

For every (t,c), a_d=mean_I_d(t)(log P_c-log P_0), e_d=mean_I_d(t)(E_c-E_0).
Delta_tc = logsumexp_d(log pi_dc+a_d+e_d) - logsumexp_d(log pi_dc+a_d).
Q = softmax(log P + [0,Delta]). Learned pi is class-specific, NOT an inference
ground-truth class selector. Widths are relative time, not seconds.

Identity at E=0; constant relative evidence passes through exactly; Delta is
between min/max e_d. Multiple separated events are allowed (no one-event/video
constraint). This is a finite-hypothesis evidence transform, not a calibrated
generative likelihood or an exact temporal segmentation posterior.
"""

    def __init__(self, abnormal_classes, scales=(1/128, 1/32, 1/8, 1/2, 1.)):
        super().__init__()
        if abnormal_classes < 1 or any(not 0 < s <= 1 for s in scales):
            raise ValueError("Invalid duration dictionary")
        self.register_buffer("scales", torch.tensor(scales))
        self.duration_logits = nn.Parameter(torch.zeros(len(scales), abnormal_classes))

    def transform(self, log_prior, evidence, edges):
        # A per-frame common class offset cancels analytically.
        odds = log_prior[..., 1:] - log_prior[..., :1]
        relative = evidence[..., 1:] - evidence[..., :1]
        center = (edges[:, :-1] + edges[:, 1:]) / 2
        left = (center[..., None] - self.scales / 2).clamp_min(0)
        right = (center[..., None] + self.scales / 2).clamp_max(1)
        values = torch.cat((odds, relative), -1)
        mean = (prefix_at(values, edges, right) - prefix_at(values, edges, left)) / (right - left)[..., None]
        a, e = mean.chunk(2, -1)
        log_pi = F.log_softmax(self.duration_logits, 0)
        # Compute the difference relative to the prior posterior; this avoids
        # subtracting two large log-partitions and keeps the E=0 identity stable.
        log_rho = F.log_softmax(a + log_pi, -2)
        delta = torch.logsumexp(log_rho + e, -2)
        field = torch.cat((torch.zeros_like(delta[..., :1]), delta), -1)
        valid = edges.diff(dim=-1) > 0
        field = field * valid[..., None]
        posterior = F.softmax(log_rho + e, -2)
        w = edges.diff(dim=-1)
        entropy = -(posterior * posterior.clamp_min(1e-12).log()).sum(-2)
        expected = (posterior * self.scales[None, None, :, None]).sum(-2)
        info = {"duration_entropy": (entropy.mean(-1) * w).sum(-1).mean(),
                "duration_mean_fraction": (expected.mean(-1) * w).sum(-1).mean(),
                "duration_update_rms": (delta.square().mean(-1) * w).sum(-1).mean().sqrt()}
        # Per-class time-weighted duration diagnostics, without labels at inference.
        for c in range(delta.shape[-1]):
            info["duration_class_" + str(c + 1)] = (expected[..., c] * w).sum(-1).mean()
        return field, info

    def forward(self, log_prior, evidence, edges):
        field, info = self.transform(log_prior, evidence, edges)
        return F.log_softmax(log_prior + field, -1), info
