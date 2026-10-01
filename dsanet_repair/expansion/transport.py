"""ASOT-inspired weak-label-compatible structural soft targets (training only).

ASOT (CVPR 2024), mingu6/action_seg_ot/src/asot.py, MIT. This reimplementation
uses log-domain mirror descent, an anchor-derived nonuniform marginal, allowed
video-level classes plus background, and normal-residual-gated temporal edges.
It is NOT a reproduction of ASOT's unsupervised clustering pipeline.
"""

import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def structural_targets(binary, semantic, features, lengths, labels, iterations=25,
                       alpha=.3, radius=.04, entropy=.07, mass_weight=.05,
                       geometry_temperature=.2, step_size=1.):
    """Return [B,T,C] detached hypotheses, confidence and scalar diagnostics.

    No frame annotations, test labels, class transcript, balanced class counts,
    external features or student predictions are used to create these targets.
    """
    if binary.ndim == 3:
        binary = binary.squeeze(-1)
    batch, steps, classes = semantic.shape
    if binary.shape != (batch, steps) or labels.shape != (batch, classes) or features.shape[:2] != (batch, steps):
        raise ValueError("Incompatible structural-target inputs")
    if (iterations < 1 or not 0 <= alpha < 1 or not 0 < radius <= 1 or
            min(entropy, geometry_temperature, step_size) <= 0 or mass_weight < 0):
        raise ValueError("Invalid transport hyperparameters")
    if not all(math.isfinite(float(x)) for x in (alpha, radius, entropy, mass_weight, geometry_temperature, step_size)):
        raise ValueError("Nonfinite transport hyperparameters")
    lengths = lengths.to(device=binary.device, dtype=torch.long)
    valid = torch.arange(steps, device=binary.device)[None] < lengths[:, None]
    if lengths.shape != (batch,) or bool(((lengths <= 0) | (lengths > steps)).any()):
        raise ValueError("Transport training requires nonempty valid sequences")
    if bool((labels.sum(-1) <= 0).any()) or not torch.isfinite(labels).all():
        raise ValueError("Transport requires valid video labels")
    binary, semantic, features = binary.float(), semantic.float(), features.float()
    if not all(torch.isfinite(x[valid]).all() for x in (binary, semantic, features)):
        raise FloatingPointError("Nonfinite valid teacher inputs")
    binary = torch.where(valid, binary, 0.)
    semantic = torch.where(valid[..., None], semantic, 0.)
    features = torch.where(valid[..., None], features, 0.)
    allowed = labels > 0
    allowed[:, 0] = True  # an anomalous bag still contains unknown background
    anomaly = binary.sigmoid()
    joint = torch.cat([(1 - anomaly)[..., None], anomaly[..., None] * semantic[..., 1:].softmax(-1)], -1)
    probabilities = joint.clamp_min(1e-6) * allowed[:, None]
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    marginal = (probabilities * valid[..., None]).sum(1) / lengths[:, None]
    cost = -probabilities.clamp_min(1e-6).log()

    time = torch.arange(steps, device=binary.device)
    distance = (time[:, None] - time[None]).abs()[None]
    neighbourhood = (distance > 0) & (distance <= (lengths * radius).floor().clamp_min(1)[:, None, None])
    square = features.square().sum(-1)
    difference = (square[:, :, None] + square[:, None, :] - 2 * (features @ features.transpose(1, 2))).clamp_min(0) / 2
    adjacency = torch.exp(-difference / geometry_temperature) * neighbourhood
    adjacency = adjacency * valid[:, :, None] * valid[:, None, :] / radius
    # T has row mass 1/N. Only the action marginal is softly constrained.
    log_mass = lengths.float().log()[:, None, None]
    support = valid[..., None] & allowed[:, None]
    log_transport = marginal.clamp_min(1e-12).log()[:, None].expand(-1, steps, -1) - log_mass
    log_transport = log_transport.masked_fill(~support, -torch.inf)
    target_marginal_log = marginal.clamp_min(1e-12).log()[:, None]
    for _ in range(iterations):
        transport = log_transport.exp()
        other_labels = transport.sum(-1, keepdim=True) - transport
        structural = adjacency @ other_labels
        actual_log = transport.sum(1, keepdim=True).clamp_min(1e-12).log()
        # Terms constant over a row vanish under the row-simplex projection.
        derivative = ((1 - alpha) * cost + alpha * structural +
                      mass_weight * (actual_log - target_marginal_log) +
                      entropy * torch.where(support, log_transport, 0.))
        candidate = log_transport - step_size * derivative
        # Padded rows are a dummy one-class simplex during normalization only.
        candidate = torch.where(valid[..., None], candidate, torch.zeros_like(candidate))
        candidate = candidate - torch.logsumexp(candidate, -1, keepdim=True) - log_mass
        log_transport = candidate.masked_fill(~support, -torch.inf)
    targets = log_transport.exp() * lengths[:, None, None]
    # Numerical renormalization, not a hard assignment or an extra prior.
    targets = torch.where(valid[..., None], targets / targets.sum(-1, keepdim=True).clamp_min(1e-12), 0.)
    teacher_entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    max_entropy = allowed.sum(-1).float().log().clamp_min(1e-6)
    confidence = (1 - teacher_entropy / max_entropy[:, None]).clamp(0, 1) * valid
    if not torch.isfinite(targets).all() or not torch.isfinite(confidence).all():
        raise FloatingPointError("Nonfinite structural targets")
    column = targets.sum(1) / lengths[:, None]
    diagnostics = {
        "transport_confidence": confidence[valid].mean(),
        "transport_normal_mass": (targets[..., 0] * valid).sum() / valid.sum(),
        "transport_marginal_change": (column - marginal).abs().sum(-1).mean(),
        "transport_row_error": (targets.sum(-1)[valid] - 1).abs().max(),
    }
    return targets.detach(), confidence.detach(), diagnostics


def transport_losses(binary, semantic, targets, confidence):
    """Factorized joint CE keeps binary/semantic parameter paths disjoint."""
    binary = binary.squeeze(-1)
    anomaly_mass = 1 - targets[..., 0]
    binary_error = F.binary_cross_entropy_with_logits(binary, anomaly_mass, reduction="none")
    class_error = -(targets[..., 1:] * F.log_softmax(semantic[..., 1:], -1)).sum(-1)
    denominator = confidence.sum().clamp_min(1.)
    return ((binary_error * confidence).sum() / denominator,
            (class_error * confidence).sum() / denominator)

