"""Transient event/normal evidence retrieval on frozen DSANet representations.

Inspired by attention retrieval in CFR (ICML 2026, official program verified) and
negative attention in Differential Transformer (ICLR 2025). Independent
implementation; no upstream source copied. This is not multimodal CFR.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def evidence_memories(features, context, binary, semantic, text, lengths, slots):
    """Inference uses no video labels, frame labels or cross-video test state."""
    batch, steps, width = features.shape
    valid = torch.arange(steps, device=features.device)[None] < lengths[:, None]
    features = torch.where(valid[..., None], features, 0.)
    context = torch.where(valid[..., None], context, 0.)
    semantic = torch.where(valid[..., None], semantic, 0.)
    probability = torch.where(valid, binary.squeeze(-1).sigmoid(), 0.)
    responsibilities = probability[..., None] * semantic[..., 1:].softmax(-1) * valid[..., None]
    mass = responsibilities.sum(1)
    events = responsibilities.transpose(1, 2) @ features / mass[..., None].clamp_min(1e-8)
    # Preserve category identity even when visually similar categories pool alike.
    texts = text[1:].unsqueeze(0).expand(batch, -1, -1)
    events = .5 * (F.layer_norm(events, (width,)) + F.layer_norm(texts, (width,)))
    event_reliability = mass / lengths.clamp_min(1)[:, None]
    # Normal-context bins respect each sequence's own valid length.
    position = torch.arange(steps, device=features.device)[None]
    bins = (position * slots // lengths.clamp_min(1)[:, None]).clamp_max(slots - 1)
    assignments = F.one_hot(bins, slots).to(features) * valid[..., None]
    normal_weight = assignments * (1 - probability)[..., None]
    normal_mass = normal_weight.sum(1)
    normals = normal_weight.transpose(1, 2) @ context / normal_mass[..., None].clamp_min(1e-8)
    normals = F.layer_norm(normals, (width,))
    normal_reliability = normal_mass / assignments.sum(1).clamp_min(1)
    return events, event_reliability, normals, normal_reliability


class EvidenceAttention(nn.Module):
    """Matched-parameter additive and signed attention, each with a null slot."""

    def __init__(self, width, hidden, heads=4, mode="contrast", slots=8):
        super().__init__()
        if mode not in ("add", "contrast", "event_only") or hidden % heads:
            raise ValueError("Invalid evidence attention")
        self.heads, self.mode, self.slots = heads, mode, slots
        self.diagnostic_disable_event = False
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.output.weight)

    def retrieve(self, queries, memory, reliability):
        batch, steps, hidden = queries.shape
        count, dim = memory.shape[1], hidden // self.heads
        query = self.query(queries).reshape(batch, steps, self.heads, dim).transpose(1, 2)
        key = self.key(memory).reshape(batch, count, self.heads, dim).transpose(1, 2)
        value = self.value(memory).reshape(batch, count, self.heads, dim).transpose(1, 2)
        logits = query @ key.transpose(-1, -2) / math.sqrt(dim)
        prior = reliability.clamp_min(1e-12).log().masked_fill(reliability <= 0, -torch.inf)
        logits = logits + prior[:, None, None]
        # A fixed zero key/value permits abstention, including empty sequences.
        logits = torch.cat([logits, torch.zeros_like(logits[..., :1])], -1)
        weights = logits.softmax(-1)
        retrieved = weights[..., :-1] @ value
        return retrieved.transpose(1, 2).reshape(batch, steps, hidden), weights[..., -1].mean(1)

    def forward(self, hidden, features, context, base, lengths, valid):
        event, er, normal, nr = evidence_memories(
            features, context, base[1], base[2], base[0], lengths, self.slots)
        positive, event_null = self.retrieve(hidden, event, er)
        negative, normal_null = self.retrieve(hidden, normal, nr)
        if self.diagnostic_disable_event:
            positive = torch.zeros_like(positive)
        if self.mode == "event_only":
            mixture = .5 * positive
        else:
            mixture = .5 * (positive + negative if self.mode == "add" else positive - negative)
        addition = self.output(mixture)
        result = torch.where(valid[..., None], hidden + addition, 0.)
        count = valid.sum().clamp_min(1)
        return result, {
            "attention_event_null": (event_null * valid).sum() / count,
            "attention_normal_null": (normal_null * valid).sum() / count,
            "attention_memory_reliability": er.mean(),
            "attention_output_abs": addition[valid].abs().mean() if bool(valid.any()) else addition.sum() * 0,
            "attention_normal_retrieval_abs": negative[valid].abs().mean() if bool(valid.any()) else negative.sum() * 0,
        }
