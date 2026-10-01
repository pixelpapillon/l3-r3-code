"""Opt-in DSANet numerical repairs; the pinned upstream source is untouched.

These are engineering changes, not a new research module. A matched comparator
must use the same mode. Exported weights retain upstream keys, but evaluating
stable-mode weights with the upstream forward is not an equivalent replay.
"""

from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F


def clear_text_cache(model):
    for module in model.modules():
        if hasattr(module, "_text_features_cache"):
            module._text_features_cache = None


@contextmanager
def fresh_evaluation(model):
    """Never reuse text embeddings from different adapter weights or prompts."""
    was_training = model.training
    clear_text_cache(model)
    try:
        yield
    finally:
        clear_text_cache(model)
        model.train(was_training)


def event_weights(binary_logits, scale=10.0, normal=False):
    """Normalized expm1(scale * sigmoid(z)), evaluated in log space.

The original exp(x)-1 cancels to zero for small x; 1-sigmoid(z) also
rounds to zero for large z. This formula preserves relative weights even
when sigmoid itself underflows. It is equivalent away from degeneracies.
"""
    if not math.isfinite(scale) or not 0 < scale <= 50:
        raise ValueError("scale must be finite and in (0, 50]")
    if binary_logits.ndim != 3 or binary_logits.shape[-1] != 1:
        raise ValueError("Expected [B,T,1] binary logits")
    if not torch.isfinite(binary_logits).all():
        raise FloatingPointError("Non-finite binary logits before event pooling")
    # FP32 for half precision, but preserve double for numerical tests.
    z = binary_logits if binary_logits.dtype == torch.float64 else binary_logits.float()
    z = -z if normal else z
    u = scale * torch.sigmoid(z)
    safe_u = u.clamp_min(1e-3)
    correction = torch.where(
        u < 1e-3,
        torch.log1p(u / 2 + u.square() / 6),
        torch.log(torch.expm1(safe_u)) - torch.log(safe_u),
    )
    log_weights = F.logsigmoid(z) + math.log(scale) + correction
    return torch.softmax(log_weights, dim=1).to(binary_logits.dtype)


def decisions_from_features(model, h, text, DNP_use=False, scale=10):
    """Shared DSANet decision path; also used by the frozen native adapter."""
    binary = model.classifier(h + model.mlp2(h))
    text_ori = model.get_text_features(text)
    if not torch.isfinite(text_ori).all():
        raise FloatingPointError("Non-finite DSANet text embeddings")
    attention = F.normalize(binary.transpose(1, 2) @ h, dim=-1, eps=1e-6)
    enhanced_text = text_ori.unsqueeze(0) + attention
    enhanced_text = enhanced_text + model.mlp1(enhanced_text)
    semantic = (F.normalize(h, dim=-1, eps=1e-6) @
                F.normalize(enhanced_text, dim=-1, eps=1e-6).transpose(1, 2).to(h)) / 0.07
    abnormal = event_weights(binary, scale).transpose(1, 2) @ h
    normal = event_weights(binary, scale, normal=True).transpose(1, 2) @ h
    text_norm = F.normalize(text_ori, dim=-1, eps=1e-6).T.to(h)
    event = F.normalize(abnormal, dim=-1, eps=1e-6) @ text_norm / 0.07
    background = F.normalize(normal, dim=-1, eps=1e-6) @ text_norm / 0.07
    result = (text_ori, binary, semantic, event, background)
    if not DNP_use:
        return result
    reconstructed, gather = model.video_anomaly_refiner(h, binary)
    return (*result, {"reconstructed_features": reconstructed,
                      "g_loss": gather, "original_features": h})


class StableForwardMixin:
    """Same trainable parameters and outputs; safe pooling/normalization."""

    def forward(self, visual, padding_mask, text, lengths, DNP_use, scale=10):
        h = self.encode_video(visual, padding_mask, lengths)
        return decisions_from_features(self, h, text, DNP_use, scale)


def stable_model_class(upstream_class):
    return type("StableDSANet", (StableForwardMixin, upstream_class), {})
