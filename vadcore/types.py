from dataclasses import dataclass, field
from typing import Any, Optional
import torch


@dataclass
class VideoSample:
    """No file IO or temporal ground truth. All edges are in seconds."""
    source_id: str
    features: torch.Tensor
    edges: torch.Tensor
    binary_label: Optional[int] = None
    labels: Optional[torch.Tensor] = None

    def __post_init__(self):
        x, t = self.features, self.edges
        if not self.source_id or x.ndim != 2 or len(x) < 1:
            raise ValueError("Expected source_id and nonempty features [T,D].")
        if not x.is_floating_point() or not torch.isfinite(x).all() or x.requires_grad:
            raise ValueError("CLIP features must be finite, floating-point and detached.")
        if t.shape != (len(x) + 1,) or not torch.isfinite(t).all() or not (t[1:] > t[:-1]).all():
            raise ValueError("edges must be strictly increasing [T+1] seconds.")
        if self.binary_label not in (None, 0, 1):
            raise ValueError("binary_label must be 0, 1 or None.")
        if self.labels is not None:
            if self.labels.ndim != 1 or not ((self.labels == -1) | (self.labels == 0) | (self.labels == 1)).all():
                raise ValueError("labels must be [C], with -1/0/1 values.")
            if self.binary_label == 0 and (self.labels == 1).any():
                raise ValueError("Normal video has a positive anomaly category.")

    @property
    def duration(self):
        return float(self.edges[-1] - self.edges[0])

    def inference_copy(self):
        return VideoSample(self.source_id, self.features, self.edges)


@dataclass
class Prediction:
    binary: torch.Tensor
    classes: torch.Tensor
    edges: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate(self):
        if self.binary.ndim != 1 or self.classes.ndim != 2 or len(self.classes) != len(self.binary):
            raise ValueError("Invalid prediction shapes.")
        for value in (self.binary, self.classes):
            if not torch.isfinite(value).all() or (value < -1e-6).any() or (value > 1 + 1e-6).any():
                raise ValueError("Predictions must be finite probabilities.")
        if len(self.edges) != len(self.binary) + 1:
            raise ValueError("Prediction timeline mismatch.")
        return self


def validate_training(samples, classes):
    if not len(samples):
        raise ValueError("Empty training sequence.")
    for v in samples:
        if v.binary_label is None or v.labels is None or len(v.labels) != classes:
            raise ValueError("Training needs binary_label and C category labels.")
        if v.binary_label == 1 and (v.labels == 0).all():
            raise ValueError("Positive bag declares all categories absent; use -1 for unknown.")


def check_disjoint(train, validation):
    overlap = {v.source_id for v in train} & {v.source_id for v in validation}
    if overlap:
        raise ValueError(f"Original-video leakage: {sorted(overlap)[:3]}")
