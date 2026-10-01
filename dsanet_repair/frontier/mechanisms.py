"""Five feature-level hypotheses, implemented independently from paper code.

All receive only current-video frozen features and anchor outputs at inference.
None consumes labels, retrieves test neighbours, or updates persistent test state.
These are adaptations of general primitives, not reproductions of source papers.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def masked(value, valid):
    return torch.where(valid[..., None], value, 0.)


def normalized_residual(features, context, valid):
    return F.layer_norm(masked(features - context, valid), (features.shape[-1],))


def window_mean(value, valid, radius):
    """Centered rectangular window, normalized by actual non-padding support."""
    value = masked(value, valid)
    width = 2 * radius + 1
    total = F.avg_pool1d(value.transpose(1, 2), width, 1, radius).transpose(1, 2) * width
    count = F.avg_pool1d(valid[:, None].to(value.dtype), width, 1, radius).transpose(1, 2) * width
    return masked(total / count.clamp_min(1), valid)


class DurationAssembly(nn.Module):
    """Assemble bounded-length interval hypotheses, then redistribute to clips.

    Unlike a pointwise gate, one interval jointly explains all its member clips.
    Overlapping candidates are allowed: there is no single-action transcript or
    mutually exclusive temporal segmentation assumption (important for XD).
    """
    def __init__(self, width, hidden):
        super().__init__()
        self.project = nn.Linear(width, hidden)
        self.boundaries = nn.Linear(hidden, 2)
        self.duration_bias = nn.Parameter(torch.zeros(4))
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.durations = (1, 3, 7, 15)
        self.diagnostic_mode = "normal"

    def forward(self, hidden, features, context, base, valid):
        z = masked(self.project(normalized_residual(features, context, valid)).tanh(), valid)
        edge = self.boundaries(z)
        steps = z.shape[1]
        prefix = F.pad(z.cumsum(1), (0, 0, 1, 0))
        numerator, denominator = torch.zeros_like(z), z.new_zeros(*valid.shape, 1)
        duration_total = z.new_zeros(())
        for index, duration in enumerate(self.durations):
            if duration > steps:
                continue
            starts = torch.arange(steps - duration + 1, device=z.device)
            ends = starts + duration
            count = F.pad(valid.long().cumsum(1), (1, 0))
            active = (count[:, ends] - count[:, starts]) == duration
            mean = (prefix[:, ends] - prefix[:, starts]) / duration
            # Contrast the interval's content with its two endpoint states.
            content = (mean - .5 * (z[:, starts] + z[:, ends - 1])).square().mean(-1)
            score = (edge[:, starts, 0] + edge[:, ends - 1, 1] + content + self.duration_bias[index])
            if self.diagnostic_mode == "uniform_intervals":
                score = torch.zeros_like(score)
            mass = torch.exp(score.clamp(-8, 8)) * active / duration
            weighted = mean * mass[..., None]
            # Difference arrays distribute each interval to all member clips.
            diff = z.new_zeros(z.shape[0], steps + 1, z.shape[-1])
            norm = z.new_zeros(z.shape[0], steps + 1, 1)
            diff = diff.index_add(1, starts, weighted).index_add(1, ends, -weighted)
            norm = norm.index_add(1, starts, mass[..., None]).index_add(1, ends, -mass[..., None])
            numerator = numerator + diff.cumsum(1)[:, :-1]
            denominator = denominator + norm.cumsum(1)[:, :-1]
            duration_total = duration_total + mass.sum() * duration
        assembled = numerator / denominator.clamp_min(1e-5)
        addition = self.output(assembled - z)
        return masked(hidden + addition, valid), {
            "duration_explanation_mass": denominator.sum() / valid.sum().clamp_min(1),
            "duration_weighted_support": duration_total / valid.sum().clamp_min(1),
            "duration_change_abs": masked(addition.abs(), valid).sum() / (valid.sum().clamp_min(1) * hidden.shape[-1]),
        }


class CapacityTransport(nn.Module):
    """Finite-mass, class-anchored latent allocation rather than pseudo labels.

    Sinkhorn balances clip mass against detached anchor-derived slot capacities.
    The resulting soft assignments explain/reconstruct features; they never
    replace the XD multi-label training targets or force one hard class per clip.
    """
    def __init__(self, width, hidden, rank, iterations, entropy):
        super().__init__()
        self.project = nn.Linear(width, rank, bias=False)
        self.output = nn.Linear(2 * rank, hidden, bias=False)
        self.iterations, self.entropy = iterations, entropy
        self.diagnostic_mode = "normal"

    def forward(self, hidden, features, context, base, valid):
        z = masked(F.normalize(self.project(masked(features, valid)), dim=-1), valid)
        text = F.normalize(self.project(base[0]), dim=-1)
        if text.ndim != 2:
            raise ValueError("Expected a shared [classes,width] frozen text bank")
        affinity = z @ text.T / self.entropy
        length = valid.sum(1, keepdim=True).clamp_min(1)
        with torch.no_grad():
            normal = 1 - base[1].sigmoid()
            event = base[1].sigmoid() * base[2][..., 1:].softmax(-1)
            prior = masked(torch.cat([normal, event], -1), valid).sum(1) / length
            capacity = .95 * prior + .05 / text.shape[0]
            capacity = capacity / capacity.sum(-1, keepdim=True)
        log_kernel = affinity.masked_fill(~valid[..., None], -1e4)
        log_row = -length.to(z.dtype).log()
        dual = torch.zeros_like(capacity)
        for _ in range(self.iterations):
            left = log_row - torch.logsumexp(log_kernel + dual[:, None], -1)
            left = left.masked_fill(~valid, -1e4)
            dual = capacity.log() - torch.logsumexp(log_kernel + left[..., None], 1)
        plan = (log_kernel + left[..., None] + dual[:, None]).exp() * valid[..., None]
        # Numerical row normalization makes padding/zero-length rows harmless.
        assignment = plan / plan.sum(-1, keepdim=True).clamp_min(1e-8)
        assignment = masked(assignment, valid)
        if self.diagnostic_mode == "independent_softmax":
            assignment = masked(affinity.softmax(-1), valid)
        mass = assignment.sum(1)
        slots = assignment.transpose(1, 2) @ z / mass.clamp_min(1e-8)[..., None]
        explained = assignment @ slots
        addition = self.output(torch.cat([explained, z - explained], -1))
        return masked(hidden + addition, valid), {
            "capacity_marginal_error": ((mass / length - capacity).abs() * (valid.any(1)[:, None])).mean(),
            "capacity_entropy": -(assignment * assignment.clamp_min(1e-8).log()).sum() / valid.sum().clamp_min(1),
            "capacity_unexplained_energy": masked((z - explained).square(), valid).sum() / valid.sum().clamp_min(1),
        }


class CrossFitDynamics(nn.Module):
    """Cross-fit local latent dynamics: fit one transition parity, predict the other.

    The held-out transition cannot directly train its own operator. Adjacent
    transitions still share states; this is not independent-sample cross-fitting
    or a causal guarantee. Fit uses each video's own features, never its label.
    """
    def __init__(self, width, hidden, rank, ridge):
        super().__init__()
        self.project = nn.Linear(width, rank, bias=False)
        self.output = nn.Linear(3 * rank, hidden, bias=False)
        self.ridge = ridge
        self.diagnostic_mode = "normal"

    def forward(self, hidden, features, context, base, valid):
        z = masked(self.project(normalized_residual(features, context, valid)).tanh(), valid)
        batch, steps, rank = z.shape
        prediction = torch.zeros_like(z)
        coverage = z.new_zeros(batch, steps, 1)
        if steps > 1:
            x, y = z[:, :-1], z[:, 1:]
            pairs = valid[:, :-1] & valid[:, 1:]
            parity = torch.arange(steps - 1, device=z.device) % 2
            eye = torch.eye(rank, dtype=z.dtype, device=z.device)[None]
            forecasts, masks = [], []
            for fold in (0, 1):
                train_mask = pairs & (parity[None] != fold)
                if self.diagnostic_mode == "self_fit":
                    train_mask = pairs
                test_mask = pairs & (parity[None] == fold) & train_mask.any(1)[:, None]
                count = train_mask.sum(1).clamp_min(1).to(z.dtype)[:, None, None]
                xt = x.transpose(1, 2) * train_mask[:, None]
                covariance = (xt @ x) / count + self.ridge * eye
                operator = torch.linalg.solve(covariance, (xt @ y) / count)
                forecasts.append(masked(x @ operator, test_mask))
                masks.append(test_mask)
            prediction = F.pad(forecasts[0] + forecasts[1], (0, 0, 1, 0))
            coverage = F.pad((masks[0] | masks[1]).to(z.dtype), (1, 0))[..., None]
        innovation = (z - prediction) * coverage
        acceleration = torch.cat([torch.zeros_like(z[:, :1]), innovation[:, 1:] - innovation[:, :-1]], 1)
        addition = self.output(torch.cat([prediction, innovation, acceleration], -1))
        return masked(hidden + addition, valid), {
            "dynamics_heldout_energy": innovation.square().sum() / (coverage.sum().clamp_min(1) * rank),
            "dynamics_coverage": coverage.sum() / valid.sum().clamp_min(1),
        }


def path_signature_two(z, valid, window):
    """Local first level and antisymmetric second level of a linear path.

    Signed area is 1/2 sum(i<j) (dx_i tensor dx_j - dx_j tensor dx_i).
    It distinguishes ordered trajectories with identical endpoint displacement.
    """
    z = masked(z, valid)
    pair = valid[:, 1:] & valid[:, :-1]
    delta = F.pad(masked(z[:, 1:] - z[:, :-1], pair), (0, 0, 1, 0))
    # Consecutive valid prefix masks are supplied by DSANet's length contract.
    padded = F.pad(delta.transpose(1, 2), (window - 1, 0))
    increments = padded.unfold(-1, window, 1).permute(0, 2, 3, 1)
    prior = increments.cumsum(2) - increments
    area = .5 * (torch.einsum("btwi,btwj->btij", prior, increments) -
                 torch.einsum("btwi,btwj->btij", increments, prior))
    indexes = torch.triu_indices(z.shape[-1], z.shape[-1], 1, device=z.device)
    return masked(increments.sum(2), valid), masked(area[..., indexes[0], indexes[1]], valid)


class PathSignature(nn.Module):
    def __init__(self, width, hidden, rank, window):
        super().__init__()
        self.project = nn.Linear(width, rank, bias=False)
        self.output = nn.Linear(rank + rank * (rank - 1) // 2, hidden, bias=False)
        self.window = window
        self.diagnostic_mode = "normal"

    def forward(self, hidden, features, context, base, valid):
        z = masked(self.project(normalized_residual(features, context, valid)).tanh(), valid)
        first, area = path_signature_two(z, valid, self.window)
        if self.diagnostic_mode == "first_order":
            area = torch.zeros_like(area)
        addition = self.output(torch.cat([first / math.sqrt(self.window), area / self.window], -1))
        return masked(hidden + addition, valid), {
            "signature_area_energy": area.square().sum() / valid.sum().clamp_min(1),
            "signature_displacement_energy": first.square().sum() / valid.sum().clamp_min(1),
        }


class WaveletScattering(nn.Module):
    """Undecimated Haar detail plus cross-scale modulus envelopes.

    Keeps signed high-frequency evidence instead of treating every sharp change
    as noise. The second-order envelopes distinguish sustained modulation from
    an isolated spike. No external wavelet or CUDA package is required.
    """
    def __init__(self, width, hidden, rank):
        super().__init__()
        self.project = nn.Linear(width, rank, bias=False)
        self.output = nn.Linear(7 * rank, hidden, bias=False)
        self.diagnostic_mode = "normal"

    def forward(self, hidden, features, context, base, valid):
        z = masked(self.project(normalized_residual(features, context, valid)).tanh(), valid)
        smooth, components, energy = z, [], z.new_zeros(())
        for dilation in (1, 2, 4):
            shifted = F.pad(smooth.transpose(1, 2), (dilation, 0))[:, :, :z.shape[1]].transpose(1, 2)
            available = F.pad(valid, (dilation, 0))[:, :z.shape[1]] & valid
            # Replicate each prefix's first available state rather than inventing
            # a zero-to-feature jump at a video's left boundary.
            shifted = torch.where(available[..., None], shifted, smooth)
            detail = masked((smooth - shifted) / math.sqrt(2), valid)
            smooth = masked((smooth + shifted) / math.sqrt(2), valid)
            modulus = detail.abs()
            coarse = dilation * 2
            delayed = F.pad(modulus.transpose(1, 2), (coarse, 0))[:, :, :z.shape[1]].transpose(1, 2)
            supported = F.pad(valid, (coarse, 0))[:, :z.shape[1]] & valid
            delayed = torch.where(supported[..., None], delayed, modulus)
            second = (modulus - delayed).abs() / math.sqrt(2)
            envelope = window_mean(second, valid, coarse)
            if self.diagnostic_mode == "no_second_order":
                envelope = torch.zeros_like(envelope)
            components.extend((detail, envelope))
            energy = energy + detail.square().sum()
        addition = self.output(torch.cat([smooth, *components], -1))
        return masked(hidden + addition, valid), {
            "wavelet_detail_energy": energy / valid.sum().clamp_min(1),
            "wavelet_change_abs": masked(addition.abs(), valid).sum() / (valid.sum().clamp_min(1) * hidden.shape[-1]),
        }


def make_mechanism(name, width, hidden, args):
    rank = args.mechanism_width
    if name == "duration":
        return DurationAssembly(width, hidden)
    if name == "transport":
        return CapacityTransport(width, hidden, rank, args.capacity_iterations, args.capacity_entropy)
    if name == "dynamics":
        return CrossFitDynamics(width, hidden, rank, args.dynamics_ridge)
    if name == "signature":
        return PathSignature(width, hidden, min(rank, 8), args.signature_window)
    if name == "wavelet":
        return WaveletScattering(width, hidden, rank)
    raise ValueError(f"Unknown feature mechanism: {name}")
