"""Projection onto a masked zero-mean box, with its active-set derivative."""

import math

import torch


class _ZeroMeanBox(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, valid, radius):
        # Bisection is a numerical root solver, not the differentiation rule.
        # The analytic active-set Jacobian below avoids a detached-threshold STE.
        work = values.double() if values.dtype == torch.float64 else values.float()
        count = valid.sum(-1, keepdim=True)
        minimum = work.masked_fill(~valid, torch.inf).amin(-1, keepdim=True)
        maximum = work.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True)
        left, right = minimum - radius, maximum + radius
        left = torch.where(count > 0, left, torch.zeros_like(left))
        right = torch.where(count > 0, right, torch.zeros_like(right))
        for _ in range(64 if work.dtype == torch.float64 else 40):
            middle = (left + right) / 2
            total = torch.where(valid, (work - middle).clamp(-radius, radius), 0.).sum(-1, keepdim=True)
            left = torch.where(total > 0, middle, left)
            right = torch.where(total > 0, right, middle)
        shifted = work - (left + right) / 2
        projected = torch.where(valid, shifted.clamp(-radius, radius), 0.)
        projected = torch.where(maximum == minimum, torch.zeros_like(projected), projected)
        active = valid & (shifted > -radius) & (shifted < radius)
        ctx.save_for_backward(active)
        return projected.to(values.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        active, = ctx.saved_tensors
        gradient = torch.where(active, grad_output, 0.)
        mean = gradient.sum(-1, keepdim=True) / active.sum(-1, keepdim=True).clamp_min(1)
        return torch.where(active, gradient - mean, 0.), None, None


def zero_mean_box(values, valid, radius=.5):
    """Project [B,T] over valid tokens; empty rows and singleton rows become zero.

    Preserves average *logit*, not average probability or global ranking.
    Differentiable within each active set (including higher-order mixed derivatives).
    Padding values are ignored entirely, including nonfinite sentinels.
    """
    if values.ndim != 2 or valid.shape != values.shape or values.shape[1] == 0:
        raise ValueError("Projection expects matching [B,T] arrays, T > 0")
    if not values.is_floating_point() or valid.dtype != torch.bool or values.device != valid.device:
        raise ValueError("Projection requires floating values and a same-device boolean mask")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Projection radius must be finite and positive")
    if not torch.isfinite(values[valid]).all():
        raise FloatingPointError("Nonfinite valid projection input")
    return _ZeroMeanBox.apply(torch.where(valid, values, 0.), valid, float(radius))
