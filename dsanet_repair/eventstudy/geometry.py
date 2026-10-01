"""Normalized original-feature coordinates; not seconds or new preprocessing."""

import torch


def timeline(lengths, steps, dtype, edges=None):
    device = lengths.device
    index = torch.arange(steps, device=device)[None]
    valid = index < lengths[:, None]
    if edges is None:
        raw = torch.arange(steps + 1, device=device, dtype=dtype)[None].expand(len(lengths), -1)
    else:
        raw = edges.to(device=device, dtype=dtype)
        if raw.shape != (len(lengths), steps + 1):
            raise ValueError("Timeline requires padded [batch,steps+1] original edges")
        if not torch.isfinite(raw).all():
            raise ValueError("Nonfinite timeline edges")
    span = raw[:, 1:] - raw[:, :-1]
    if bool((span[valid] <= 0).any()):
        raise ValueError("Valid original-feature edges must strictly increase")
    total = raw.gather(1, lengths[:, None]) - raw[:, :1]
    total = total.clamp_min(1)
    centers = ((raw[:, 1:] + raw[:, :-1]) * .5 - raw[:, :1]) / total
    # Padding must sort strictly after valid coordinates for batched searchsorted.
    centers = torch.where(valid, centers, 2 + index.to(dtype) / max(steps, 1))
    width = torch.where(valid, span / total, 0.)
    return centers, width, valid


def interpolate(values, centers, positions, lengths):
    """Piecewise-linear sampling with gradients to both features and positions."""
    batch, steps, rank = values.shape
    shape = positions.shape
    target = positions.reshape(batch, -1).contiguous()
    upper = torch.searchsorted(centers.contiguous(), target).clamp(0, steps - 1)
    last = (lengths - 1).clamp_min(0)[:, None]
    upper = torch.minimum(upper, last)
    lower = (upper - 1).clamp_min(0)
    t0, t1 = centers.gather(1, lower), centers.gather(1, upper)
    fraction = ((target - t0) / (t1 - t0).clamp_min(1e-8)).clamp(0, 1)
    left = values.gather(1, lower[..., None].expand(-1, -1, rank))
    right = values.gather(1, upper[..., None].expand(-1, -1, rank))
    result = left + fraction[..., None] * (right - left)
    result = torch.where((lengths > 0)[:, None, None], result, 0.)
    return result.reshape(*shape, rank)

