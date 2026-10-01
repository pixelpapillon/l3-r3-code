"""Task routing and a primary-directed ConsMTL-inspired optimization extension.

P6 combines a one-sided PCGrad-type shared projection with ConsMTL's insight
that head parameters can modify feature-space task gradients. It prioritizes
binary detection instead of reproducing ConsMTL's symmetric bargaining solver.
No claim of monotone AUC/AP or of a descent guarantee under AdamW is made.
"""

import torch


def gradients(loss, parameters, create_graph=False):
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True,
                                 create_graph=create_graph)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, values)]


def flatten(values):
    return torch.cat([value.reshape(-1) for value in values])


def assign(parameters, values):
    if torch.is_tensor(values):
        cursor = 0
        for parameter in parameters:
            parameter.grad = values[cursor:cursor + parameter.numel()].reshape_as(parameter).detach().clone()
            cursor += parameter.numel()
        if cursor != values.numel():
            raise ValueError("Gradient vector has the wrong dimension")
    else:
        for parameter, gradient in zip(parameters, values):
            parameter.grad = gradient.detach().clone()


def protect_primary(primary, auxiliary, ratio=.5):
    """One-sided projection, then a relative auxiliary budget (no parameters)."""
    if primary.shape != auxiliary.shape or ratio < 0:
        raise ValueError("Invalid primary-gradient projection")
    square = primary.square().sum()
    dot = (primary * auxiliary).sum()
    if float(square) <= 1e-20:
        accepted = torch.zeros_like(auxiliary)
    else:
        accepted = auxiliary - dot.clamp_max(0) / square * primary
        scale = (ratio * square.sqrt() / accepted.norm().clamp_min(1e-20)).clamp_max(1.)
        accepted = accepted * scale
    update = primary + accepted
    return update, {"primary_aux_raw_dot": dot,
                    "primary_aux_accepted_dot": (primary * accepted).sum(),
                    "primary_surrogate_descent_dot": (primary * update).sum(),
                    "accepted_auxiliary_norm": accepted.norm()}


def diagnostics(model, losses):
    result = {}
    groups = model.parameter_partitions()
    if "shared" not in groups:
        return {"binary_semantic_cosine": None, "binary_semantic_shared_parameters": 0}
    for left, right, prefix in (("binary", "weighted_semantic", "binary_semantic"),
                                ("binary", "response_binary", "binary_response")):
        a = flatten(gradients(losses[left], groups["shared"]))
        b = flatten(gradients(losses[right], groups["shared"]))
        norm_a, norm_b = float(a.norm()), float(b.norm())
        result.update({f"{prefix}_left_norm": norm_a, f"{prefix}_right_norm": norm_b,
                       f"{prefix}_cosine": float(a @ b) / (norm_a * norm_b) if min(norm_a, norm_b) > 0 else None})
    if "rank_binary" in losses:
        parameters = [p for group in groups.values() for p in group]
        for name in ("rank_binary", "rank_semantic"):
            result[name + "_gradient_norm"] = float(flatten(gradients(losses[name], parameters)).norm())
    return result


def optimizer_step(model, losses, optimizer, args):
    optimizer.zero_grad(set_to_none=True)
    partitions = model.parameter_partitions()
    statistics = {}
    if args.variant != "p6-primary-aligned":
        losses["total"].backward()
        if hasattr(model, "configuration"):
            for name in ("attention", "flow", "state_filter", "class_memory", "mechanism", "event_core"):
                module = getattr(model, name, None)
                if module is not None:
                    gradients_present = [p.grad for p in module.parameters() if p.grad is not None]
                    statistics[name + "_gradient_norm"] = float(flatten(gradients_present).norm()) if gradients_present else 0.
            if getattr(model, "event_core", None) is not None:
                for name in ("geometry", "query", "trajectory"):
                    part = getattr(model.event_core, name, None)
                    if part is not None:
                        values = [p.grad for p in part.parameters() if p.grad is not None]
                        statistics["event_" + name + "_gradient_norm"] = float(flatten(values).norm()) if values else 0.
        clipping_groups = partitions if model.split else {"all": [p for g in partitions.values() for p in g]}
        for name, group in clipping_groups.items():
            statistics[f"gradient_norm_{name}"] = float(torch.nn.utils.clip_grad_norm_(
                group, args.clip_grad, error_if_nonfinite=True))
        optimizer.step()
        return statistics

    shared, binary, semantic = (partitions[k] for k in ("shared", "binary", "semantic"))
    primary = flatten(gradients(losses["binary_task"], shared)).detach()
    auxiliary = flatten(gradients(losses["semantic_task"], shared)).detach()
    binary_grad = gradients(losses["binary_task"], binary)
    semantic_grad = gradients(losses["semantic_task"], semantic)
    hidden = model.last_hidden
    if hidden is None:
        raise ValueError("Primary alignment requires an actual shared hidden representation")
    binary_signal = torch.autograd.grad(losses["binary_task"], hidden, retain_graph=True)[0].detach()
    semantic_signal = torch.autograd.grad(losses["semantic_task"], hidden, retain_graph=True,
                                        create_graph=True)[0]
    # A mixed derivative changes the semantic head's gradient at shared tokens.
    # The reference direction and the extra-gradient budget are both detached.
    alignment = -args.head_alignment_weight * (semantic_signal * binary_signal).sum()
    extra = gradients(alignment, semantic)
    base_norm, extra_norm = flatten(semantic_grad).norm(), flatten(extra).norm()
    multiplier = (args.head_alignment_cap * base_norm.detach() / extra_norm.detach().clamp_min(1e-20)).clamp_max(1.)
    semantic_grad = [g + multiplier * addition for g, addition in zip(semantic_grad, extra)]
    update, projection_statistics = protect_primary(primary, auxiliary, args.auxiliary_gradient_ratio)
    assign(shared, update)
    assign(binary, binary_grad)
    assign(semantic, semantic_grad)
    before = flatten([p.detach() for p in shared]).clone()
    statistics.update({k: float(v.detach()) for k, v in projection_statistics.items()})
    statistics.update({"head_alignment": float(alignment.detach()),
                       "head_alignment_extra_norm": float(extra_norm.detach() * multiplier),
                       "head_alignment_base_norm": float(base_norm.detach())})
    for name, group in partitions.items():
        statistics[f"gradient_norm_{name}"] = float(torch.nn.utils.clip_grad_norm_(
            group, args.clip_grad, error_if_nonfinite=True))
    optimizer.step()
    after = flatten([p.detach() for p in shared])
    # AdamW's preconditioning/momentum/weight decay can change descent geometry.
    statistics["primary_actual_step_dot"] = float(primary @ (before - after))
    return statistics
