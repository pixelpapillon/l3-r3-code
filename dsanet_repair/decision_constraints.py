"""V2 research kernel: constrain actual decision responses, not an aux head.

Input logits must be binary logits / abnormal-vs-normal margins in the SAME
units as cached teacher responses. Bounds are empirical response envelopes,
not confidence intervals with guaranteed coverage. This is intentionally not
plugged into the legacy v1 cache/runner: v1 discarded edited inputs and signed
response uncertainty, so relabeling that cache as v2 would be incorrect.
"""

import torch
from torch.nn import functional as F


def topk_pool(logits, lengths, divisor=16):
    if logits.ndim != 3 or divisor < 1:
        raise ValueError("Expected [B,T,K] logits and positive divisor")
    lengths = torch.as_tensor(lengths, device=logits.device)
    if lengths.shape != (len(logits),) or not torch.equal(lengths, lengths.long()):
        raise ValueError("Expected one integer length per video")
    if bool(((lengths < 1) | (lengths > logits.shape[1])).any()):
        raise ValueError("Invalid valid lengths")
    return torch.stack([
        row[:int(length)].topk(min(int(length), int(length) // divisor + 1), dim=0).values.mean(0)
        for row, length in zip(logits, lengths)
    ])


def decision_response_loss(original, edited, lengths, lower, upper, reliability,
                           null_response=None, divisor=16):
    """Signed interval constraint on pooled original-minus-edited decisions.

    original [B,T,K], edited [B,Q,T,K], bounds/weights [B,Q,K]. Q are
    actual feature edits, NOT Q alternative hard frame labels. Unsupported
    edits contribute exactly zero gradient. Both paths receive gradients.
    null_response is detached and subtracted in the same raw-logit units.
    """
    if edited.ndim != 4 or original.shape != (edited.shape[0], *edited.shape[2:]):
        raise ValueError("Incompatible original / edited logit shapes")
    batch, queries, steps, channels = edited.shape
    expected = (batch, queries, channels)
    if queries < 1 or any(x.shape != expected for x in (lower, upper, reliability)):
        raise ValueError("Expected [B,Q,K] bounds and weights with Q >= 1")
    if not all(torch.isfinite(x).all() for x in (original, edited, lower, upper, reliability)):
        raise ValueError("Non-finite decision constraints")
    if bool((lower > upper).any()) or bool(((reliability < 0) | (reliability > 1)).any()):
        raise ValueError("Invalid envelope or reliability")
    before = topk_pool(original, lengths, divisor)[:, None]
    repeated_lengths = torch.as_tensor(lengths, device=edited.device).repeat_interleave(queries)
    after = topk_pool(edited.reshape(batch * queries, steps, channels), repeated_lengths, divisor)
    response = before - after.reshape(expected)
    if null_response is not None:
        if null_response.shape != expected or not torch.isfinite(null_response).all():
            raise ValueError("Invalid null response")
        response = response - null_response.detach().to(response)
    # Do not force an arbitrary point estimate inside a supported envelope.
    distance = F.relu(lower.detach().to(response) - response) + F.relu(response - upper.detach().to(response))
    point = F.smooth_l1_loss(distance, torch.zeros_like(distance), reduction="none")
    weights = reliability.detach().to(response)
    loss = (point * weights).sum() / weights.sum().clamp_min(1)
    return loss, {"supported_mass": weights.sum().detach(),
                  "response": response.detach(), "violation": distance.detach()}


def anchor_trust_loss(student, anchor, lengths, radius=0.25):
    """Soft trust region; limits drift, but does NOT guarantee AUC retention."""
    if student.shape != anchor.shape or student.ndim != 3 or radius < 0:
        raise ValueError("Invalid anchor tensors / radius")
    topk_pool(student, lengths)  # shared shape/length validation
    lengths = torch.as_tensor(lengths, device=student.device)
    valid = torch.arange(student.shape[1], device=student.device)[None] < lengths[:, None]
    excess = F.relu((student - anchor.detach().to(student)).abs() - radius)
    return excess.square()[valid].mean()
