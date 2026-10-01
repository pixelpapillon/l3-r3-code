"""Channel-preserving loss decomposition; no changes to cached measurements."""

import torch
from torch.nn import functional as F

from dsanet_repair.official_losses import mil_binary, mil_class
from dsanet_repair.revision.response import scoped_responses, feasible_envelopes
from dsanet_repair.revision.train import flatten_decisions
from .transport import structural_targets, transport_losses


def batch_losses(model, batch, prompt, cfg, device, args):
    values = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    features, edited, lengths = values["features"], values["edited"], values["length"]
    batch_size, queries, steps, width = edited.shape
    combined = torch.cat([features, edited.reshape(-1, steps, width)])
    combined_lengths = torch.cat([lengths, lengths.repeat_interleave(queries)])
    output = model(combined, None, prompt, combined_lengths, False)
    raw = flatten_decisions(output)
    original, changed = raw[:batch_size], raw[batch_size:].reshape(batch_size, queries, steps, -1)
    binary = mil_binary(output[1][:batch_size], values["labels"], lengths)
    semantic = mil_class(output[2][:batch_size], values["labels"], lengths)
    valid = torch.arange(steps, device=device)[None] < lengths[:, None]
    channels = raw.shape[-1]
    excess = F.relu((original - values["anchor"].detach()).abs() - args.anchor_radius).square()
    # The denominator remains all channels, as in the archived R2 objective.
    trust_binary = excess[..., 0][valid].sum() / (valid.sum() * channels)
    trust_semantic = excess[..., 1:][valid].sum() / (valid.sum() * channels)
    response = scoped_responses(original, changed, lengths, values["intervals"], args.topk_divisor)
    base = model.last_base_raw
    base_response = scoped_responses(base[:batch_size], base[batch_size:].reshape_as(changed),
                                     lengths, values["intervals"], args.topk_divisor)
    lower, upper, weight, rejected = feasible_envelopes(
        base_response, values["lower"], values["upper"], values["weight"], values["null"], model.radii)
    corrected = response - values["null"].detach()
    violation = F.relu(lower - corrected) + F.relu(corrected - upper)
    error = F.smooth_l1_loss(violation, torch.zeros_like(violation), reduction="none")
    response_binary, response_semantic = binary * 0, semantic * 0
    diagnostic = {}
    weights = (args.response_weight, args.local_weight, args.outside_weight)
    if args.variant == "p2-center-no-response":
        weights = (0., 0., 0.)
    for scope, name in enumerate(("global", "local", "outside")):
        terms = error[:, :, scope] * weight[:, :, scope]
        denominator = weight[:, :, scope].sum().clamp_min(1)
        part_binary, part_semantic = terms[..., 0].sum() / denominator, terms[..., 1:].sum() / denominator
        response_binary = response_binary + weights[scope] * part_binary
        response_semantic = response_semantic + weights[scope] * part_semantic
        diagnostic.update({f"{name}_loss": part_binary + part_semantic,
                           f"{name}_binary_loss": part_binary, f"{name}_semantic_loss": part_semantic,
                           f"{name}_supported_mass": weight[:, :, scope].sum(),
                           f"{name}_violation_mass": (weight[:, :, scope] * (violation[:, :, scope] > 0)).sum()})
    normal_weight = values["normal_weight"][..., None] * valid[:, None]
    normal_denominator = normal_weight.sum().clamp_min(1)
    edited_binary = output[1][batch_size:].reshape(batch_size, queries, steps)
    edited_semantic = output[2][batch_size:].reshape(batch_size, queries, steps, -1)
    normal_binary = (F.softplus(edited_binary) * normal_weight).sum() / normal_denominator
    normal_semantic = (-F.log_softmax(edited_semantic, -1)[..., 0] * normal_weight).sum() / normal_denominator
    transport_binary, transport_semantic = binary * 0, semantic * 0
    if args.variant == "p5-structured-transport":
        targets, confidence, transport_diagnostic = structural_targets(
            model.last_base_outputs[0][:batch_size], model.last_base_outputs[1][:batch_size],
            model.last_features[:batch_size], lengths, values["labels"],
            iterations=args.transport_iterations, alpha=args.transport_alpha, radius=args.transport_radius,
            entropy=args.transport_entropy, mass_weight=args.transport_mass,
            geometry_temperature=args.transport_geometry_temperature, step_size=args.transport_step)
        transport_binary, transport_semantic = transport_losses(
            output[1][:batch_size], output[2][:batch_size], targets, confidence)
        diagnostic.update(transport_diagnostic)
    binary_task = (binary + args.anchor_weight * trust_binary + response_binary +
                   args.normal_weight * normal_binary + args.transport_weight * transport_binary)
    semantic_task = (cfg.loss2_weight * semantic + args.anchor_weight * trust_semantic + response_semantic +
                     args.normal_weight * normal_semantic + args.transport_weight * transport_semantic)
    combined_valid = torch.arange(steps, device=device)[None] < combined_lengths[:, None]
    delta = model.last_correction.detach()[combined_valid]
    pre = model.last_unprojected.detach()
    means = pre.sum(-1) / combined_lengths.clamp_min(1)
    denominator = pre.square().sum().clamp_min(1e-12)
    post = model.last_correction[..., 0].detach()
    diagnostic.update({
        "binary_delta_abs": delta[:, 0].abs().mean(),
        "semantic_delta_abs": delta[:, 1:].abs().mean(),
        "binary_saturation_fraction": (delta[:, 0].abs() >= .95 * model.radii[0]).float().mean(),
        "semantic_saturation_fraction": (delta[:, 1:].abs() >= .95 * model.radii[1:]).float().mean(),
        "preprojection_constant_energy_fraction": (combined_lengths * means.square()).sum() / denominator,
        "postprojection_mean_abs": (post.sum(-1) / combined_lengths.clamp_min(1)).abs().max(),
        "projection_energy_ratio": post.square().sum() / denominator,
        "projection_interior_fraction": (delta[:, 0].abs() < model.radii[0] - 1e-6).float().mean(),
        "rejected_mass": rejected,
    })
    return {"total": binary_task + semantic_task, "binary_task": binary_task, "semantic_task": semantic_task,
            "binary": binary, "semantic": semantic, "weighted_semantic": cfg.loss2_weight * semantic,
            "video": binary + cfg.loss2_weight * semantic,
            "response": response_binary + response_semantic,
            "response_binary": response_binary, "response_semantic": response_semantic,
            "anchor": trust_binary + trust_semantic, "normal_edit": normal_binary + normal_semantic,
            "transport_binary": transport_binary, "transport_semantic": transport_semantic,
            **diagnostic}
