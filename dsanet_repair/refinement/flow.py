"""Symmetric, boundary-conditioned transport of learned correction logits.

For a path graph, step < 1/2 and conductance in [0,1] make each update a
convex combination. Paired edge flux preserves the sum of every channel.
These are algebraic properties, not guarantees about AP or true boundaries.
"""

import torch
from torch import nn


class BoundaryFlow(nn.Module):
    def __init__(self, steps=3, temperature=.2, phase_shift=False):
        super().__init__()
        if steps < 1 or temperature <= 0:
            raise ValueError("Invalid boundary flow")
        self.steps, self.temperature, self.phase_shift = steps, temperature, phase_shift
        self.diagnostic_bypass = False
        self.edge_gate = nn.Linear(3, 1)
        nn.init.zeros_(self.edge_gate.weight)
        nn.init.constant_(self.edge_gate.bias, 2.)
        self.step_logit = nn.Parameter(torch.tensor(0.))

    def forward(self, correction, residual, binary, semantic, valid):
        value = torch.where(valid[..., None], correction, 0.)
        if self.diagnostic_bypass:
            zero = value.sum() * 0
            return value, {"flow_conductance": zero, "flow_conductance_std": zero,
                           "flow_change_abs": zero, "flow_sum_error": zero}
        if value.shape[1] < 2:
            return value, {"flow_conductance": value.sum() * 0, "flow_sum_error": value.sum() * 0}
        with torch.no_grad():
            residual = torch.where(valid[..., None], residual, 0.)
            geometry = .5 * (residual[:, 1:] - residual[:, :-1]).square().sum(-1)
            probability = torch.where(valid, binary.squeeze(-1).sigmoid(), 0.)
            jump = (probability[:, 1:] - probability[:, :-1]).abs()
            semantic = torch.where(valid[..., None], semantic, 0.)
            categories = semantic[..., 1:].softmax(-1)
            disagreement = .5 * (categories[:, 1:] - categories[:, :-1]).abs().sum(-1)
            edge_features = torch.stack([geometry, jump, disagreement], -1)
            edge_valid = valid[:, 1:] & valid[:, :-1]
            prior = torch.exp(-edge_features.sum(-1) / self.temperature) * edge_valid
        conductance = prior * self.edge_gate(edge_features).sigmoid().squeeze(-1)
        if self.phase_shift:
            # Rotate only within each video's valid edges. It preserves each
            # video's edge-weight multiset but breaks edge/location alignment.
            shifted = []
            for row, count in zip(conductance, edge_valid.sum(1).tolist()):
                if count > 1:
                    shifted.append(torch.cat((row[count // 2:count], row[:count // 2],
                                              row[count:]), dim=0))
                else:
                    shifted.append(row)
            conductance = torch.stack(shifted)
        step = .49 * self.step_logit.sigmoid()
        before = value.sum(1)
        for _ in range(self.steps):
            flux = step * conductance[..., None] * (value[:, 1:] - value[:, :-1])
            zero = torch.zeros_like(value[:, :1])
            value = value + torch.cat([flux, zero], 1) - torch.cat([zero, flux], 1)
        edge_count = edge_valid.sum().clamp_min(1)
        mean = conductance.sum() / edge_count
        diagnostics = {"flow_conductance": mean,
                       "flow_conductance_std": ((conductance - mean).square() * edge_valid).sum().div(edge_count).sqrt(),
                       "flow_change_abs": (value - torch.where(valid[..., None], correction, 0.)).abs().sum()
                                          / valid.sum().clamp_min(1) / value.shape[-1],
                       "flow_sum_error": (value.sum(1) - before).abs().max()}
        return value, diagnostics
