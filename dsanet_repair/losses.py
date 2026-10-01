"""Losses for A3 intervention-response learning."""

from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ResponseLossConfig:
    reconstruction_weight: float = 1.0
    ranking_weight: float = 0.25
    absent_class_weight: float = 0.05
    temporal_weight: float = 0.02
    ranking_margin: float = 0.1

    def __post_init__(self):
        values = (self.reconstruction_weight, self.ranking_weight,
                  self.absent_class_weight, self.temporal_weight, self.ranking_margin)
        if any(x < 0 for x in values) or self.reconstruction_weight == 0:
            raise ValueError("Invalid response-loss weights")


def _valid_mask(lengths, steps, device):
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)
    if lengths.ndim != 1 or bool((lengths < 1).any()) or bool((lengths > steps).any()):
        raise ValueError("Invalid lengths")
    return torch.arange(steps, device=device)[None, :] < lengths[:, None]


def intervention_response_loss(prediction, target, weight, lengths, video_labels,
                               config=None):
    """Fit held-out intervention responses and regularize weak-label conflicts.

    ``video_labels`` follows DSANet class order and includes normal at index 0.
    It is used only for safe negative constraints: a class absent from the
    video-level annotation should not acquire a large removable-evidence field.
    """
    cfg = config or ResponseLossConfig()
    if prediction.shape != target.shape or prediction.shape != weight.shape or prediction.ndim != 3:
        raise ValueError("Response loss expects matching [B,T,K] tensors")
    batch, steps, targets = prediction.shape
    if video_labels.shape != (batch, targets):
        raise ValueError("video_labels must follow DSANet class order")
    valid = _valid_mask(lengths, steps, prediction.device)
    reliable = weight.to(prediction).clamp(0, 1) * valid.unsqueeze(-1)
    target = target.to(prediction).clamp_min(0)

    point = F.smooth_l1_loss(prediction, target, reduction="none")
    reconstruction = (point * reliable).sum() / reliable.sum().clamp_min(1.0)

    # Within-video ordering is less sensitive than absolute regression to OOF
    # teacher calibration. Compare the strongest and weakest supported times.
    ranking_terms = []
    for b in range(batch):
        for k in range(targets):
            idx = reliable[b, :, k] > 0
            if int(idx.sum()) < 2:
                continue
            truth = target[b, idx, k]
            hi = truth.argmax()
            lo = truth.argmin()
            if truth[hi] - truth[lo] <= 1e-6:
                continue
            gap = prediction[b, idx, k][hi] - prediction[b, idx, k][lo]
            ranking_terms.append(F.relu(cfg.ranking_margin - gap))
    ranking = torch.stack(ranking_terms).mean() if ranking_terms else prediction.sum() * 0

    present_abnormal = video_labels[:, 1:].to(prediction).clamp(0, 1)
    absent = (1 - present_abnormal)[:, None, :] * valid[:, :, None]
    absent_penalty = (prediction[:, :, 1:] * absent).sum() / absent.sum().clamp_min(1.0)
    normal = video_labels[:, :1].to(prediction).clamp(0, 1)[:, None, :]
    normal_mask = normal * valid[:, :, None]
    normal_penalty = (prediction * normal_mask).sum() / (normal_mask.sum().clamp_min(1.0) * targets)

    pair_valid = valid[:, 1:] & valid[:, :-1]
    temporal_delta = (prediction[:, 1:] - prediction[:, :-1]).abs()
    temporal = (temporal_delta * pair_valid.unsqueeze(-1)).sum() / (
        pair_valid.sum().clamp_min(1) * targets
    )
    total = (cfg.reconstruction_weight * reconstruction + cfg.ranking_weight * ranking +
             cfg.absent_class_weight * (absent_penalty + normal_penalty) +
             cfg.temporal_weight * temporal)
    return total, {
        "response_reconstruction": reconstruction.detach(),
        "response_ranking": ranking.detach(),
        "response_absent": absent_penalty.detach(),
        "response_normal": normal_penalty.detach(),
        "response_temporal": temporal.detach(),
        "response_supported": reliable.sum().detach(),
    }
