"""Versioned cache for paired DSANet edit-response training.

Unlike the legacy v1 cache, every supported query keeps an exact replacement
tensor and a signed response envelope.  The cache therefore has enough
information to reconstruct the edited input used by the response constraint.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch


@dataclass
class PairedEdit:
    start: int
    end: int
    replacement: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    reliability: torch.Tensor
    null_response: torch.Tensor
    diagnostics: dict

    def validate(self, feature_dim=None, channels=None, valid_length=None):
        if not 0 <= int(self.start) < int(self.end):
            raise ValueError("Invalid paired-edit interval")
        if valid_length is not None and int(self.end) > int(valid_length):
            raise ValueError("Paired edit exceeds the valid timeline")
        if self.replacement.ndim != 2 or len(self.replacement) != self.end - self.start:
            raise ValueError("Replacement must be [end-start,D]")
        if feature_dim is not None and self.replacement.shape[1] != int(feature_dim):
            raise ValueError("Replacement feature dimension mismatch")
        vectors = (self.lower, self.upper, self.reliability, self.null_response)
        if any(x.ndim != 1 for x in vectors) or len({len(x) for x in vectors}) != 1:
            raise ValueError("Response bounds, weights and nulls must be matching [K] tensors")
        if channels is not None and len(self.lower) != int(channels):
            raise ValueError("Response channel mismatch")
        if not all(torch.isfinite(x).all() for x in (self.replacement, *vectors)):
            raise ValueError("Non-finite paired-edit cache tensor")
        if bool((self.lower > self.upper).any()):
            raise ValueError("Signed response lower bound exceeds upper bound")
        if bool(((self.reliability < 0) | (self.reliability > 1)).any()):
            raise ValueError("Reliability must be in [0,1]")
        return self


@dataclass
class V2ResponseItem:
    source_id: str
    valid_length: int
    feature_dim: int
    anchor: torch.Tensor
    edits: tuple
    diagnostics: dict

    def validate(self, visual_length=None, channels=None):
        if not self.source_id or self.valid_length < 1 or self.feature_dim < 1:
            raise ValueError("Invalid v2 response item identity/shape")
        if visual_length is not None and self.valid_length > int(visual_length):
            raise ValueError("Cached valid length exceeds the DSANet grid")
        if self.anchor.ndim != 2 or self.anchor.shape[0] != self.valid_length:
            raise ValueError("Anchor logits must be [valid_length,K]")
        if channels is not None and self.anchor.shape[1] != int(channels):
            raise ValueError("Anchor channel mismatch")
        if not torch.isfinite(self.anchor).all():
            raise ValueError("Non-finite anchor logits")
        self.edits = tuple(self.edits)
        for edit in self.edits:
            edit.validate(self.feature_dim, self.anchor.shape[1], self.valid_length)
        return self


def _tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def cache_signature(items, metadata):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for item in sorted(items, key=lambda value: value.source_id):
        digest.update(item.source_id.encode())
        digest.update(str((item.valid_length, item.feature_dim)).encode())
        digest.update(_tensor_bytes(item.anchor))
        for edit in item.edits:
            digest.update(str((edit.start, edit.end)).encode())
            for tensor in (edit.replacement, edit.lower, edit.upper,
                           edit.reliability, edit.null_response):
                digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _pack_edit(edit):
    return {
        "start": int(edit.start), "end": int(edit.end),
        "replacement": edit.replacement.detach().cpu().float(),
        "lower": edit.lower.detach().cpu().float(),
        "upper": edit.upper.detach().cpu().float(),
        "reliability": edit.reliability.detach().cpu().float(),
        "null_response": edit.null_response.detach().cpu().float(),
        "diagnostics": dict(edit.diagnostics),
    }


def _unpack_edit(value):
    return PairedEdit(
        int(value["start"]), int(value["end"]), value["replacement"],
        value["lower"], value["upper"], value["reliability"],
        value["null_response"], dict(value.get("diagnostics", {})),
    )


def save_v2_cache(path, items, metadata):
    metadata = dict(metadata)
    visual_length = metadata.get("visual_length")
    channels = metadata.get("channels")
    items = [item.validate(visual_length, channels) for item in items]
    identities = [item.source_id for item in items]
    if len(identities) != len(set(identities)):
        raise ValueError("V2 cache requires one item per DSANet training row")
    payload = {
        "format": "dsanet-intervention-response-v2",
        "metadata": metadata,
        "signature": cache_signature(items, metadata),
        "items": {
            item.source_id: {
                "valid_length": int(item.valid_length),
                "feature_dim": int(item.feature_dim),
                "anchor": item.anchor.detach().cpu().float(),
                "edits": [_pack_edit(edit) for edit in item.edits],
                "diagnostics": dict(item.diagnostics),
            }
            for item in items
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_v2_cache(path, expected_metadata=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "dsanet-intervention-response-v2":
        raise ValueError("Unknown paired response-cache format")
    metadata = dict(payload.get("metadata", {}))
    if expected_metadata is not None and metadata != expected_metadata:
        raise ValueError("V2 response-cache metadata mismatch")
    items = []
    for source_id, value in payload.get("items", {}).items():
        items.append(V2ResponseItem(
            str(source_id), int(value["valid_length"]), int(value["feature_dim"]),
            value["anchor"], tuple(_unpack_edit(edit) for edit in value["edits"]),
            dict(value.get("diagnostics", {})),
        ).validate(metadata.get("visual_length"), metadata.get("channels")))
    if not items:
        raise ValueError("Empty v2 response cache")
    if payload.get("signature") != cache_signature(items, metadata):
        raise ValueError("V2 response-cache signature mismatch")
    return {item.source_id: item for item in items}, metadata
