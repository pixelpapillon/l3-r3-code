"""M3 equal-budget free/locked response learning in exact cached logit units."""

import torch
from torch.nn import functional as F

from dsanet_repair.expansion.losses import batch_losses as c1_losses
from dsanet_repair.revision.response import scoped_responses, feasible_envelopes
from dsanet_repair.revision.train import flatten_decisions


def response_terms(response, lower, upper, weight, null, args):
    violation = F.relu(lower - (response - null)) + F.relu((response - null) - upper)
    error = F.smooth_l1_loss(violation, torch.zeros_like(violation), reduction="none")
    binary, semantic = response.sum() * 0, response.sum() * 0
    for scope, coefficient in enumerate((args.response_weight, args.local_weight, args.outside_weight)):
        terms = error[:, :, scope] * weight[:, :, scope]
        norm = weight[:, :, scope].sum().clamp_min(1)
        binary = binary + coefficient * terms[..., 0].sum() / norm
        semantic = semantic + coefficient * terms[..., 1:].sum() / norm
    return binary, semantic, violation


def batch_losses(model, batch, prompt, cfg, device, args, class_audit=False):
    if "edges" not in batch or "response_holdout" not in batch:
        raise ValueError("Event study requires its timeline/holdout dataset wrapper")
    batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
             for key, value in batch.items()}
    count, queries = batch["edited"].shape[:2]
    holdout = batch["response_holdout"]
    train_batch = dict(batch)
    train_batch["weight"] = batch["weight"] * (~holdout)[:, :, None, None]
    model.timeline_edges = torch.cat([batch["edges"], batch["edges"].repeat_interleave(queries, 0)])
    try:
        losses = c1_losses(model, train_batch, prompt, cfg, device, args)
    finally:
        model.timeline_edges = None
    free = flatten_decisions(model.last_outputs)
    before, after = free[:count], free[count:].reshape(count, queries, *free.shape[1:])
    base = model.last_base_raw
    response_base = scoped_responses(base[:count], base[count:].reshape_as(after), batch["length"],
                                    batch["intervals"], args.topk_divisor)
    lower, upper, weight, _ = feasible_envelopes(response_base, batch["lower"], batch["upper"],
                                               batch["weight"], batch["null"], model.radii)
    ordinary = scoped_responses(before, after, batch["length"], batch["intervals"], args.topk_divisor)
    zero = ordinary.sum() * 0
    alpha = args.locked_alpha * min(1., (model.training_step + 1) / max(1, model.steps_per_epoch))
    if not model.event_configuration.locked:
        alpha = 0.
    locked_binary, locked_semantic, locked = zero, zero, ordinary
    if model.event_configuration.locked:
        edited = model.locked_raw(count, queries).reshape_as(after)
        locked = scoped_responses(before, edited, batch["length"], batch["intervals"], args.topk_divisor)
        locked_binary, locked_semantic, _ = response_terms(
            locked, lower, upper, weight * (~holdout)[:, :, None, None], batch["null"], args)
    free_binary, free_semantic = losses["response_binary"], losses["response_semantic"]
    losses["binary_task"] = losses["binary_task"] + alpha * (locked_binary - free_binary)
    losses["semantic_task"] = losses["semantic_task"] + alpha * (locked_semantic - free_semantic)
    losses["response_binary"] = (1 - alpha) * free_binary + alpha * locked_binary
    losses["response_semantic"] = (1 - alpha) * free_semantic + alpha * locked_semantic
    losses["response"] = losses["response_binary"] + losses["response_semantic"]
    losses["total"] = losses["binary_task"] + losses["semantic_task"]
    losses.update(response_free=(free_binary + free_semantic).detach(),
                  response_locked=(locked_binary + locked_semantic).detach(),
                  locked_alpha=zero.detach() + alpha,
                  locked_supported_mass=(weight * (~holdout)[:, :, None, None]).sum().detach()
                  if model.event_configuration.locked else zero.detach())
    # Held responses NEVER enter total. Record raw sufficient statistics so
    # aggregate diagnostics are weighted by support rather than batch count.
    with torch.no_grad():
        for mode, response in (("free", ordinary), ("locked", locked)):
            if mode == "locked" and not model.event_configuration.locked:
                continue
            _, _, violation = response_terms(response, lower, upper, weight, batch["null"], args)
            held = weight * holdout[:, :, None, None]
            for scope, name in enumerate(("global", "local", "outside")):
                for channel, section in (("binary", slice(0, 1)), ("semantic", slice(1, None))):
                    w = held[:, :, scope, section]
                    prefix = f"held_{mode}_{name}_{channel}"
                    losses[prefix + "_mass"] = w.sum()
                    losses[prefix + "_violation_sum"] = (w * violation[:, :, scope, section]).sum()
                if class_audit:
                    for channel in range(1, held.shape[-1]):
                        w = held[:, :, scope, channel]
                        prefix = f"held_{mode}_{name}_class{channel}"
                        losses[prefix + "_mass"] = w.sum()
                        losses[prefix + "_violation_sum"] = (w * violation[:, :, scope, channel]).sum()
    losses.update(model.module_diagnostics)
    return losses
