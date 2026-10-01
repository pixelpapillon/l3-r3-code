"""First-order cross-video agreement gate for the existing OOF response loss.

This is deliberately NOT an influence-function or test-risk estimate. The
reference objective uses other training videos in the same minibatch only.
"""

import torch
from torch.nn import functional as F

from dsanet_repair.official_losses import mil_binary, mil_class
from dsanet_repair.revision.response import feasible_envelopes, scoped_responses
from dsanet_repair.revision.train import flatten_decisions


def _flat_grad(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
                      for parameter, gradient in zip(parameters, gradients)]).detach()


def _reference_objective(model, batch, labels, lengths, indexes, cfg, args):
    count, queries, steps, _ = batch["edited"].shape
    output = model.last_outputs
    binary = mil_binary(output[1][:count][indexes], labels[indexes], lengths[indexes])
    semantic = mil_class(output[2][:count][indexes], labels[indexes], lengths[indexes])
    edited_binary = output[1][count:].reshape(count, queries, steps)[indexes]
    edited_semantic = output[2][count:].reshape(count, queries, steps, -1)[indexes]
    valid = torch.arange(steps, device=lengths.device)[None, None] < lengths[indexes][:, None, None]
    normal = batch["normal_weight"].to(lengths.device)[indexes, :, None] * valid
    denominator = normal.sum().clamp_min(1)
    normal_loss = ((F.softplus(edited_binary) -
                    F.log_softmax(edited_semantic, -1)[..., 0]) * normal).sum() / denominator
    return binary + cfg.loss2_weight * semantic + args.normal_weight * normal_loss


def gated_response_losses(model, batch, cfg, device, args):
    """Recreate Q1 response terms with stop-gradient per-query agreement weights."""
    count, queries, steps, _ = batch["edited"].shape
    lengths, labels = (batch[name].to(device, non_blocking=True) for name in ("length", "labels"))
    raw = flatten_decisions(model.last_outputs)
    original = raw[:count]
    changed = raw[count:].reshape(count, queries, steps, -1)
    intervals = batch["intervals"].to(device, non_blocking=True)
    response = scoped_responses(original, changed, lengths, intervals, args.topk_divisor)
    base = model.last_base_raw
    base_response = scoped_responses(base[:count], base[count:].reshape_as(changed),
                                     lengths, intervals, args.topk_divisor)
    lower, upper, weight, _ = feasible_envelopes(
        base_response, batch["lower"].to(device), batch["upper"].to(device),
        batch["weight"].to(device), batch["null"].to(device), model.radii)
    corrected = response - batch["null"].to(device).detach()
    violation = F.relu(lower - corrected) + F.relu(corrected - upper)
    error = F.smooth_l1_loss(violation, torch.zeros_like(violation), reduction="none")
    parameters = [p for module in (model.binary_output, model.semantic_output)
                  for p in module.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Agreement gate requires trainable correction heads")
    references = {}
    for parity in (0, 1):
        indexes = torch.arange(count, device=device) % 2 != parity
        if bool(indexes.any()):
            reference = _reference_objective(model, batch, labels, lengths, indexes, cfg, args)
            references[parity] = _flat_grad(reference, parameters)
    factors = torch.ones(count, queries, device=device, dtype=raw.dtype)
    cosine_sum = raw.new_zeros(())
    cosines = []
    probed = 0
    for row in range(count):
        reference = references.get(row % 2)
        if reference is None or float(reference.norm()) <= 1e-12:
            continue
        for query in range(queries):
            probe = (error[row, query] * weight[row, query]).sum()
            if float(probe.detach()) <= 1e-12:
                continue
            candidate = _flat_grad(probe, parameters)
            if float(candidate.norm()) <= 1e-12:
                continue
            cosine = torch.dot(candidate, reference) / (candidate.norm() * reference.norm()).clamp_min(1e-12)
            cosine = cosine.clamp(-1, 1)
            factors[row, query] = (1 + cosine).clamp(0, 2)
            cosine_sum = cosine_sum + cosine
            cosines.append(cosine)
            probed += 1
    effective = weight * factors[:, :, None, None]
    response_binary, response_semantic = raw.sum() * 0, raw.sum() * 0
    diagnostic = {}
    for scope, (name, coefficient) in enumerate(zip(
            ("global", "local", "outside"),
            (args.response_weight, args.local_weight, args.outside_weight))):
        terms = error[:, :, scope] * effective[:, :, scope]
        denominator = effective[:, :, scope].sum().clamp_min(1)
        part_binary = terms[..., 0].sum() / denominator
        part_semantic = terms[..., 1:].sum() / denominator
        response_binary = response_binary + coefficient * part_binary
        response_semantic = response_semantic + coefficient * part_semantic
        diagnostic[name + "_loss"] = (part_binary + part_semantic).detach()
        diagnostic[name + "_binary_loss"] = part_binary.detach()
        diagnostic[name + "_semantic_loss"] = part_semantic.detach()
    diagnostic.update({
        "influence_probe_fraction": raw.new_tensor(probed / max(count * queries, 1)),
        "influence_cosine_mean": cosine_sum / max(probed, 1),
        "influence_factor_mean": factors.mean(),
        "influence_changed_fraction": (factors != 1).float().mean(),
        "influence_factor_std": factors.std(unbiased=False),
    })
    if cosines:
        values = torch.stack(cosines)
        for name, q in (("p10", .1), ("p50", .5), ("p90", .9)):
            diagnostic["influence_cosine_" + name] = torch.quantile(values, q)
    else:
        for name in ("p10", "p50", "p90"):
            diagnostic["influence_cosine_" + name] = raw.new_zeros(())
    return response_binary, response_semantic, {key: value.detach() for key, value in diagnostic.items()}
