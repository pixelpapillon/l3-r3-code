"""Convert A1/A2 intervention evidence into A3 response supervision."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import torch


@dataclass
class ResponseSupervision:
    source_id: str
    target: torch.Tensor  # [T,K], calibrated selective coalition response
    weight: torch.Tensor  # [T,K], reliability/coverage
    base: torch.Tensor  # [T,K], OOF teacher raw logits
    diagnostics: dict

    def validate(self):
        if not self.source_id:
            raise ValueError("source_id is required")
        if self.target.shape != self.weight.shape or self.target.shape != self.base.shape:
            raise ValueError("Response supervision shapes do not match")
        if self.target.ndim != 2:
            raise ValueError("Response tensors must be [T,K]")
        if not all(torch.isfinite(x).all() for x in (self.target, self.weight, self.base)):
            raise ValueError("Non-finite response supervision")
        if bool((self.target < 0).any()) or bool((self.weight < 0).any()) or bool((self.weight > 1).any()):
            raise ValueError("Targets must be non-negative and weights in [0,1]")
        return self


def evidence_to_supervision(source_id, evidence):
    """Use the selective coalition value as the response target.

    ``features[...,0]`` is the Shapley allocation of calibrated utility after
    edit cost and collateral-category penalties. ``features[...,1]`` is the
    robust z-response.  Reliability combines independent normal-control
    coverage with response magnitude; unsupported positions remain zero-weight
    rather than being mislabeled as negative evidence.
    """
    if evidence.features.ndim != 3 or evidence.features.shape[-1] < 2:
        raise ValueError("RepairEvidence does not contain response features")
    target = evidence.features[..., 0].clamp_min(0).float()
    z_support = evidence.features[..., 1].clamp_min(0).clamp_max(1).float()
    weight = (evidence.gate.float() * (0.25 + 0.75 * z_support)).clamp(0, 1)
    return ResponseSupervision(
        source_id=str(source_id),
        target=target,
        weight=weight,
        base=evidence.base.float(),
        diagnostics=dict(evidence.diagnostics),
    ).validate()


def cache_signature(items, metadata):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for item in sorted(items, key=lambda x: x.source_id):
        digest.update(item.source_id.encode())
        for value in (item.target, item.weight, item.base):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_response_cache(path, items, metadata):
    items = [x.validate() for x in items]
    ids = [x.source_id for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Response cache requires one item per source video")
    payload = {
        "format": "dsanet-intervention-response-v1",
        "metadata": dict(metadata),
        "signature": cache_signature(items, metadata),
        "items": {x.source_id: {
            "target": x.target.cpu(), "weight": x.weight.cpu(), "base": x.base.cpu(),
            "diagnostics": x.diagnostics,
        } for x in items},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_response_cache(path, expected_metadata=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "dsanet-intervention-response-v1":
        raise ValueError("Unknown response-cache format")
    if expected_metadata is not None and payload.get("metadata") != expected_metadata:
        raise ValueError("Response cache metadata mismatch")
    items = [ResponseSupervision(sid, row["target"], row["weight"], row["base"], row["diagnostics"]).validate()
             for sid, row in payload["items"].items()]
    if payload.get("signature") != cache_signature(items, payload["metadata"]):
        raise ValueError("Response cache signature mismatch")
    return {x.source_id: x for x in items}, payload["metadata"]
