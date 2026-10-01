"""DSANet paper losses, kept numerically aligned with the upstream scripts."""

import torch
from torch import nn
from torch.nn import functional as F


def mil_class(logits, labels, lengths):
    if not torch.isfinite(logits).all():
        raise FloatingPointError("mil_class received non-finite logits")
    rows = []
    labels = labels / labels.sum(1, keepdim=True).clamp_min(1e-6)
    for i in range(len(logits)):
        k = int(int(lengths[i]) / 16 + 1)
        rows.append(torch.topk(logits[i, :int(lengths[i])], k=k, dim=0).values.mean(0))
    return -(labels * F.log_softmax(torch.stack(rows), dim=1)).sum(1).mean()


def mil_binary(logits, labels, lengths):
    if not torch.isfinite(logits).all():
        raise FloatingPointError("mil_binary received non-finite logits")
    target = 1 - labels[:, 0]
    probability = torch.sigmoid(logits.squeeze(-1))
    # Legacy recovery clamp, NOT exact upstream parity. Finite sigmoid inputs
    # produce probabilities in [0, 1]; a CUDA BCE assertion is not evidence
    # that finite sigmoid exceeded that range. Check earlier NaNs / targets.
    probability = probability.clamp(min=1e-6, max=1.0 - 1e-6)
    rows = []
    for i in range(len(logits)):
        k = int(int(lengths[i]) / 16 + 1)
        rows.append(torch.topk(probability[i, :int(lengths[i])], k=k).values.mean())
    return F.binary_cross_entropy(torch.stack(rows), target)


def event_class(logits, labels, lengths, epsilon=0.1):
    if not torch.isfinite(logits).all():
        raise FloatingPointError("event_class received non-finite logits")
    count = logits.shape[-1]
    smooth = (1 - epsilon) * labels / labels.sum(1, keepdim=True).clamp_min(1e-6) + epsilon / count
    rows = [logits[i, :int(lengths[i])].amax(0) for i in range(len(logits))]
    return -(smooth * F.log_softmax(torch.stack(rows), dim=1)).sum(1).mean()


def background_class(logits, labels, lengths, epsilon=0.1):
    if not torch.isfinite(logits).all():
        raise FloatingPointError("background_class received non-finite logits")
    target = torch.full_like(labels, 0.01)
    target[:, 0] = 1
    target = (1 - epsilon) * target / target.sum(1, keepdim=True) + epsilon / logits.shape[-1]
    rows = [logits[i, :int(lengths[i])].amax(0) for i in range(len(logits))]
    return -(target * F.log_softmax(torch.stack(rows), dim=1)).sum(1).mean()


def text_separation(text_features):
    normal = F.normalize(text_features[0], dim=-1)
    if len(text_features) < 2:
        return text_features.sum() * 0
    abnormal = F.normalize(text_features[1:], dim=-1)
    return (abnormal @ normal).abs().mean()


def dnp_consistency(logits, original, reconstructed, lengths):
    reconstruction = (1 - F.cosine_similarity(original, reconstructed, dim=-1)) / 2
    classifier = torch.sigmoid(logits.squeeze(-1))
    mask = torch.arange(logits.shape[1], device=logits.device)[None] < lengths[:, None]
    return F.mse_loss(classifier[mask], reconstruction[mask])


def paper_loss(output, labels, lengths, loss2_weight, dnp_use=True):
    text, logits1, logits2, logits3, logits4 = output[:5]
    parts = {
        "binary": mil_binary(logits1, labels, lengths),
        "semantic": mil_class(logits2, labels, lengths),
        "text": text_separation(text),
        "event": event_class(logits3, labels, lengths),
        "background": background_class(logits4, labels, lengths),
    }
    total = (parts["binary"] + loss2_weight * parts["semantic"] + parts["text"] +
             parts["event"] + parts["background"])
    if dnp_use:
        dnp = output[5]
        parts["consistency"] = dnp_consistency(
            logits1, dnp["original_features"], dnp["reconstructed_features"], lengths
        )
        parts["gather"] = dnp["g_loss"]
        total = total + parts["consistency"] + parts["gather"]
    bad = [name for name, value in parts.items() if not torch.isfinite(value).all()]
    if bad:
        raise FloatingPointError(f"Non-finite paper loss terms: {', '.join(bad)}")
    if not torch.isfinite(total).all():
        raise FloatingPointError("Non-finite paper loss total")
    return total, parts
