"""OOF-response-supported ranking without declaring edited intervals anomalous.

Only known-normal training bags provide binary negatives. A supported interval
is a positive *bag* whose pooled evidence is ranked, never a set of positive
frame labels. The response envelope remains an empirical measurement.
"""

import torch
from torch.nn import functional as F

from dsanet_repair.expansion.losses import batch_losses as reference_losses
from dsanet_repair.revision.response import masked_topk, scope_masks
from dsanet_repair.revision.train import flatten_decisions


def evidence_ranking(raw, labels, lengths, intervals, lower, upper, weight,
                     divisor=16, margin=.5, temperature=1., binary_temperature=None):
    batch, steps, channels = raw.shape
    queries = intervals.shape[1]
    valid, local, _ = scope_masks(lengths, intervals, steps).unbind(2)
    expanded = raw[:, None].expand(-1, queries, -1, -1).reshape(-1, steps, channels)
    pooled = masked_topk(expanded, local.reshape(-1, steps), divisor).reshape(batch, queries, channels)
    # The lower bound is already null-control corrected in the cache schema.
    confidence = (lower[:, :, 1].clamp_min(0) /
                  (lower[:, :, 1].abs() + (upper[:, :, 1] - lower[:, :, 1]).clamp_min(0) + 1e-6))
    confidence = (confidence * weight[:, :, 1].clamp(0, 1)).detach()
    confidence = confidence * local.any(-1)[..., None]
    abnormal = labels[:, 1:].sum(-1) > 0
    normal = (labels[:, 0] > 0) & ~abnormal
    confidence = confidence * abnormal[:, None, None]
    zero = raw.sum() * 0
    binary_loss, gap = zero, zero.detach()
    binary_weight = confidence[..., 0]
    if bool(normal.any()):
        negative = masked_topk(raw[normal], valid[normal, 0], divisor)[:, 0]
        difference = pooled[..., 0, None] - negative[None, None]
        binary_temperature = temperature if binary_temperature is None else binary_temperature
        error = binary_temperature * F.softplus((margin - difference) / binary_temperature)
        binary_loss = (error.mean(-1) * binary_weight).sum() / binary_weight.sum().clamp_min(1)
        gap = (difference.detach().mean(-1) * binary_weight).sum() / binary_weight.sum().clamp_min(1)
    else:
        binary_weight = binary_weight * 0
    # Missing video classes may be negatives; other PRESENT XD labels may not.
    absent = labels[:, 1:] <= 0
    absent_exists = absent.any(-1)
    negative_semantic = pooled[..., 1:].masked_fill(~absent[:, None], -1e4).max(-1).values
    class_weight = confidence[..., 1:] * (labels[:, None, 1:] > 0) * absent_exists[:, None, None]
    class_difference = pooled[..., 1:] - negative_semantic[..., None]
    class_error = temperature * F.softplus((margin - class_difference) / temperature)
    semantic_loss = (class_error * class_weight).sum() / class_weight.sum().clamp_min(1)
    diagnostics = {
        "rank_binary_mass": binary_weight.sum(), "rank_semantic_mass": class_weight.sum(),
        "rank_query_fraction": (binary_weight > 0).float().mean(),
        "rank_normal_bags": normal.float().sum(), "rank_binary_gap": gap,
    }
    if bool(normal.any()):
        active = (difference.detach() < margin + binary_temperature).float()
        diagnostics["rank_binary_active_fraction"] = (
            active.mean(-1) * (binary_weight > 0)).sum() / (binary_weight > 0).sum().clamp_min(1)
    return binary_loss, semantic_loss, diagnostics


def batch_losses(model, batch, prompt, cfg, device, args):
    losses = reference_losses(model, batch, prompt, cfg, device, args)
    if model.configuration.no_response:
        # Preserve forwards, query evaluation and RNG; remove only these losses.
        losses["binary_task"] = losses["binary_task"] - losses["response_binary"]
        losses["semantic_task"] = losses["semantic_task"] - losses["response_semantic"]
        losses["response_binary"] = losses["response_binary"] * 0
        losses["response_semantic"] = losses["response_semantic"] * 0
        losses["response"] = losses["response"] * 0
    if model.configuration.influence:
        from .influence import gated_response_losses
        response_binary, response_semantic, diagnostic = gated_response_losses(
            model, batch, cfg, device, args)
        losses["binary_task"] = losses["binary_task"] - losses["response_binary"] + response_binary
        losses["semantic_task"] = losses["semantic_task"] - losses["response_semantic"] + response_semantic
        losses["response_binary"] = response_binary
        losses["response_semantic"] = response_semantic
        losses["response"] = response_binary + response_semantic
        losses.update(diagnostic)
    for key, value in model.module_diagnostics.items():
        losses[key] = value.detach()
    zero = losses["total"] * 0
    binary_rank, semantic_rank = zero, zero
    if model.configuration.ranking:
        count = batch["features"].shape[0]
        raw = flatten_decisions(model.last_outputs)[:count]
        values = [batch[key].to(device, non_blocking=True) for key in
                  ("labels", "length", "intervals", "lower", "upper", "weight")]
        binary_rank, semantic_rank, diagnostic = evidence_ranking(
            raw, *values, args.topk_divisor, args.rank_margin, args.rank_temperature,
            model.configuration.binary_rank_temperature or None)
        losses.update({k: v.detach() for k, v in diagnostic.items()})
    losses["rank_binary"] = binary_rank
    losses["rank_semantic"] = semantic_rank
    losses["binary_task"] = losses["binary_task"] + args.rank_weight * binary_rank
    losses["semantic_task"] = losses["semantic_task"] + args.semantic_rank_weight * semantic_rank
    losses["total"] = losses["binary_task"] + losses["semantic_task"]
    return losses
