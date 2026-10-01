"""New evidence on R3's learned class/slot intervals, not Evolve's fixed basis.

Only scalar evidence is synthesized. No extra encoder, proposal bank, R6 joint
cache, or pairwise temporal attention. All classes are evaluated at inference.
"""

import torch
from torch import nn
from torch.nn import functional as F
from dsanet_repair.eventevolve.modules import haar_basis, prefix_at, circulation, DurationMarginalReadout


class SlotEvidence(nn.Module):
    def __init__(self, hidden, rank, ordered=False, pieces=8):
        super().__init__()
        self.rank, self.pieces = rank, pieces
        self.register_buffer("haar", haar_basis(pieces))
        self.project = nn.Linear(hidden, rank, bias=False)
        self.detail_head = nn.Linear(rank, 1, bias=False)
        nn.init.zeros_(self.detail_head.weight)
        # Isolate construction so L1/L2 share exactly the same detail branch.
        if ordered:
            with torch.random.fork_rng(devices=[]):
                self.area_project = nn.Linear(rank, rank * 2, bias=False)
                self.area_head = nn.Linear(rank, 1, bias=False)
                nn.init.zeros_(self.area_head.weight)
        else:
            self.area_project = self.area_head = None

    def forward(self, hidden, bounds, widths, valid, support, class_scale):
        edges = F.pad(widths.cumsum(-1), (1, 0))
        nodes = torch.linspace(0, 1, self.pieces + 1, device=hidden.device, dtype=hidden.dtype)
        ends = bounds[..., :1] + bounds.diff(dim=-1) * nodes
        projected = self.project(torch.where(valid[..., None], hidden, 0.))
        integrals = prefix_at(projected, edges, ends)
        path = integrals.diff(dim=-2) / ends.diff(dim=-1)[..., None].clamp_min(1e-8)
        path = path * class_scale[None, :, None, None]
        coefficients = torch.einsum("hs,bcksr->bckhr", self.haar, path)
        score = self.detail_head(coefficients.tanh()).squeeze(-1)
        # Exact original-bin overlap, including unequal compressed training bins.
        overlap = (torch.minimum(edges[:, None, None, None, 1:], ends[..., 1:, None]) -
                   torch.maximum(edges[:, None, None, None, :-1], ends[..., :-1, None])).clamp_min(0)
        coverage = overlap / widths[:, None, None, None].clamp_min(1e-8)
        wave = torch.einsum("hs,bckst->bckht", self.haar, coverage)
        detail = torch.einsum("bckht,bckh->btc", wave, score)
        # A time-constant per-video/class normalizer preserves signed zero-DC.
        norm = support.sum(2).amax(-1).clamp_min(1)[:, None]
        detail = detail / norm
        detail = torch.where(valid[..., None], detail, 0.)
        w = widths / widths.sum(-1, keepdim=True).clamp_min(1e-8)
        diagnostics = {
            "transfer_detail_rms": (detail.square().mean(-1) * w).sum(-1).mean().sqrt(),
            "transfer_detail_dc_error": (detail * widths[..., None]).sum(1).abs().amax(),
        }
        area_field = torch.zeros_like(detail)
        if self.area_head is not None:
            coordinates = self.area_project(path).reshape(*path.shape[:-1], self.rank, 2)
            area = circulation(coordinates)
            local = self.area_head(area).squeeze(-1)
            render = support / (1 + support.sum(2, keepdim=True))
            area_field = torch.einsum("bckt,bck->btc", render, local)
            diagnostics.update(transfer_area_rms=area.square().mean().sqrt(),
                               transfer_area_field_rms=(area_field.square().mean(-1) * w).sum(-1).mean().sqrt())
        return detail + area_field, edges, diagnostics


class ResidualDurationReadout(DurationMarginalReadout):
    """One prefix pass, exact null subtraction, in R3 pre-bound evidence units."""

    def forward(self, odds, evidence, edges):
        center = (edges[:, :-1] + edges[:, 1:]) / 2
        left = (center[..., None] - self.scales / 2).clamp_min(0)
        right = (center[..., None] + self.scales / 2).clamp_max(1)
        values = torch.cat((odds, evidence), -1)
        mean = (prefix_at(values, edges, right) - prefix_at(values, edges, left)) / (right - left)[..., None]
        a, e = mean.chunk(2, -1)
        log_rho = F.log_softmax(a + F.log_softmax(self.duration_logits, 0), -2)
        delta = torch.logsumexp(log_rho + e, -2) - torch.logsumexp(log_rho, -2)
        delta = torch.where((edges.diff(dim=-1) > 0)[..., None], delta, 0.)
        posterior = F.softmax(log_rho + e, -2)
        entropy = -(posterior * posterior.clamp_min(1e-8).log()).sum(-2)
        w = edges.diff(dim=-1)
        diagnostic = (entropy.mean(-1) * w).sum() / w.sum().clamp_min(1)
        return delta, {"transfer_duration_entropy": diagnostic}
