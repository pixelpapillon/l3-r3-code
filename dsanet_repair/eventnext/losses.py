"""R response budgets, actual joint interactions, and trained relation readout."""

import torch
from torch.nn import functional as F
from dsanet_repair.eventstudy.losses import batch_losses as event_losses
from dsanet_repair.revision.train import flatten_decisions
from .joint import joint_response, feasible_joint


def batch_losses(model, batch, prompt, cfg, device, args, class_audit=False):
    losses = event_losses(model, batch, prompt, cfg, device, args, class_audit=class_audit)
    count = len(batch["length"])
    labels, lengths = batch["labels"].to(device), batch["length"].to(device)
    model.observe_primary(count, labels, lengths)
    if not model.next_configuration.interaction:
        return losses
    required = ("joint_edited", "joint_intervals", "joint_lower", "joint_upper", "joint_weight", "joint_null", "joint_holdout")
    if any(key not in batch for key in required):
        raise ValueError("R6 requires REAL joint measurements, not sums of cached single responses")
    values = {key: batch[key].to(device) for key in required}
    original = flatten_decisions(model.last_outputs)[:count]
    base_original = model.last_base_raw[:count]
    hidden, text, centers, widths, valid, _ = model.event_state
    # Query coordinates are from the actual cached native grid, not a guessed
    # compressed-index fraction. The same encoder serves learned slots at test.
    edges = batch["edges"].to(device)
    span = (edges.gather(1, lengths[:, None]) - edges[:, :1]).clamp_min(1)
    intervals = values["joint_intervals"]
    bounds = (edges.gather(1, intervals.reshape(count, -1)).reshape(count, 2, 2) - edges[:, :1, None]) / span[:, :, None]
    prediction = model.event_core.pair_prediction(hidden[:count], text, centers[:count], widths[:count], valid[:count], bounds)
    original_diagnostics = dict(model.module_diagnostics)
    model.timeline_edges = edges.repeat_interleave(3, 0)
    try:
        edit = values["joint_edited"]
        output = model(edit.reshape(count * 3, *edit.shape[2:]), None, prompt, lengths.repeat_interleave(3), False)
    finally:
        model.timeline_edges = None
    after = flatten_decisions(output).reshape(count, 3, *original.shape[1:])
    base_after = model.last_base_raw.reshape_as(after)
    response = joint_response(torch.cat([original[:, None], after], 1), lengths, args.topk_divisor)
    base = joint_response(torch.cat([base_original[:, None], base_after], 1), lengths, args.topk_divisor)
    lo, hi, reliability = feasible_joint(base, values["joint_lower"], values["joint_upper"],
                                         values["joint_weight"], values["joint_null"], model.radii)
    corrected = response - values["joint_null"]
    violation = F.relu(lo - corrected) + F.relu(corrected - hi)
    # Each class predicts binary and its own semantic interaction. Binary
    # averaging is for this auxiliary estimate only, not class field readout.
    estimate = torch.cat([prediction[..., 0].mean(1, keepdim=True), prediction[..., 1]], -1)
    graph_violation = F.relu(lo - estimate) + F.relu(estimate - hi)
    held = values["joint_holdout"]
    weight = reliability * (~held)[:, None]
    error = .5 * (F.smooth_l1_loss(violation, torch.zeros_like(violation), reduction="none") +
                  F.smooth_l1_loss(graph_violation, torch.zeros_like(graph_violation), reduction="none"))
    scale = args.response_weight + args.local_weight + args.outside_weight
    terms = error * weight * scale / weight.sum().clamp_min(1)
    joint_binary, joint_semantic = terms[:, 0].sum(), terms[:, 1:].sum()
    alpha = (args.interaction_alpha * min(1., (model.training_step + 1) / max(1, model.steps_per_epoch))
             if bool(weight.sum() > 0) else 0.)
    # Equal total coefficient budget; no hidden doubling by adding a new loss.
    for task, addition in (("binary", joint_binary), ("semantic", joint_semantic)):
        old = losses["response_" + task]
        losses[task + "_task"] = losses[task + "_task"] + alpha * (addition - old)
        losses["response_" + task] = (1 - alpha) * old + alpha * addition
    losses["response"] = losses["response_binary"] + losses["response_semantic"]
    losses["total"] = losses["binary_task"] + losses["semantic_task"]
    losses.update(joint_response_loss=joint_binary + joint_semantic, interaction_alpha=response.new_tensor(alpha),
                  joint_supported_mass=weight.sum(), joint_response_abs=response.detach().abs().mean(),
                  relation_prediction_abs=estimate.detach().abs().mean())
    for sl, name in ((slice(0, 1), "binary"), (slice(1, None), "semantic")):
        w = reliability[:, sl] * held[:, None]
        losses["held_joint_" + name + "_mass"] = w.sum().detach()
        losses["held_joint_" + name + "_violation_sum"] = (w * violation[:, sl]).sum().detach()
        losses["held_relation_" + name + "_mass"] = w.sum().detach()
        losses["held_relation_" + name + "_violation_sum"] = (w * graph_violation[:, sl]).sum().detach()
    model.module_diagnostics = original_diagnostics
    return losses
