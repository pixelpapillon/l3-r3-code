"""Global/local/outside response operators in the same raw-logit units."""

import torch
from torch.nn import functional as F


def masked_topk(logits, mask, divisor=16):
    if logits.ndim != 3 or mask.shape != logits.shape[:2] or divisor < 1:
        raise ValueError("Expected logits [B,T,K], mask [B,T], positive divisor")
    count = mask.sum(1)
    k = torch.minimum(count, count // divisor + 1)
    ordered = logits.masked_fill(~mask[..., None], -torch.inf).sort(dim=1, descending=True).values
    take = torch.arange(logits.shape[1], device=logits.device)[None] < k[:, None]
    return torch.where(take[..., None], ordered, 0.).sum(1) / k.clamp_min(1)[:, None]


def scope_masks(lengths, intervals, steps):
    time = torch.arange(steps, device=lengths.device)[None, None]
    valid = time < lengths[:, None, None]
    valid = valid.expand(-1, intervals.shape[1], -1)
    local = valid & (time >= intervals[..., 0, None]) & (time < intervals[..., 1, None])
    return torch.stack([valid, local, valid & ~local], dim=2)


def scoped_responses(original, edited, lengths, intervals, divisor=16):
    batch, queries, steps, channels = edited.shape
    if original.shape != (batch, steps, channels) or intervals.shape != (batch, queries, 2):
        raise ValueError("Incompatible response inputs")
    masks = scope_masks(lengths, intervals, steps)
    before = original[:, None, None].expand(-1, queries, 3, -1, -1).reshape(-1, steps, channels)
    after = edited[:, :, None].expand(-1, -1, 3, -1, -1).reshape(-1, steps, channels)
    flat_masks = masks.reshape(-1, steps)
    response = masked_topk(before, flat_masks, divisor) - masked_topk(after, flat_masks, divisor)
    return response.reshape(batch, queries, 3, channels)


def envelope_loss(response, lower, upper, weight, null):
    if not all(x.shape == response.shape for x in (lower, upper, weight, null)):
        raise ValueError("Response/envelope shape mismatch")
    corrected = response - null.detach()
    violation = F.relu(lower.detach() - corrected) + F.relu(corrected - upper.detach())
    error = F.smooth_l1_loss(violation, torch.zeros_like(violation), reduction="none")
    loss = (error * weight.detach()).sum() / weight.sum().clamp_min(1)
    return loss, {"supported_mass": weight.sum().detach(),
                  "violation_mass": ((violation > 0) * weight).sum().detach()}


def normal_edit_loss(binary, semantic, lengths, normal_weight):
    """Known-normal bags only; never label an edited abnormal bag as normal."""
    batch, queries, steps = binary.shape
    valid = torch.arange(steps, device=binary.device)[None, None] < lengths[:, None, None]
    weight = normal_weight[..., None] * valid
    binary_error = F.softplus(binary)  # BCE-with-logits target=0
    class_error = -F.log_softmax(semantic, dim=-1)[..., 0]
    return ((binary_error + class_error) * weight).sum() / weight.sum().clamp_min(1)


def feasible_envelopes(base_response, lower, upper, weight, null, radii):
    """Intersect evidence with a necessary outer bound on response changes.

    A masked top-k mean is 1-Lipschitz in the infinity norm. Bounded changes
    to BOTH original and edited logits therefore change a response by at most
    2*radius. Empty intersections abstain; they are not projected into labels.
    This is an output-compatibility property, not an AUC/AP guarantee.
    Nonempty intersections are necessary, not sufficient, for realizability
    by a shared correction network.
    """
    base = base_response.detach() - null
    limit = 2 * radii.to(base)
    intersection_lower = torch.maximum(lower, base - limit)
    intersection_upper = torch.minimum(upper, base + limit)
    valid = intersection_lower <= intersection_upper
    rejected_mass = (weight * ~valid).sum().detach()
    return (torch.where(valid, intersection_lower, 0.),
            torch.where(valid, intersection_upper, 0.), weight * valid, rejected_mass)
