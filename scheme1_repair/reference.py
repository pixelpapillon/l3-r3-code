from dataclasses import dataclass, asdict, replace
import math
import torch
from torch.nn import functional as F
from vadcore.types import VideoSample


@dataclass
class ReferenceConfig:
    retrievals: int = 2
    positions_per_video: int = 16
    context_steps: int = 4
    max_context_distance: float = 0.6
    duration_tolerance: float = 0.35
    normal_controls: int = 4
    min_controls: int = 3
    scale_floor: float = 0.1
    transition_quantile: float = 0.995
    transition_margin: float = 0.1
    max_reference_videos: int = 128

    def __post_init__(self):
        if min(self.retrievals, self.positions_per_video, self.context_steps,
               self.normal_controls, self.min_controls, self.max_reference_videos) < 1:
            raise ValueError("Reference counts must be positive.")
        if self.min_controls > self.normal_controls or self.scale_floor <= 0:
            raise ValueError("Invalid calibration controls/scale.")


@dataclass
class Edit:
    uid: int
    group: int
    start: int
    end: int
    replacement: torch.Tensor
    reference_id: str
    context_distance: float
    seconds: float


def cpu_sample(v):
    return VideoSample(v.source_id, v.features.detach().cpu().float().clone(),
                       v.edges.detach().cpu().float().clone(), 0,
                       None if v.labels is None else v.labels.detach().cpu().clone())


def descriptor(features, start, end, width):
    pieces = [features[max(0, start - width):start], features[end:min(len(features), end + width)]]
    pieces = [x for x in pieces if len(x)]
    if not pieces:
        return None
    return F.normalize(torch.cat(pieces).float().mean(0), dim=0, eps=1e-6)


def resize_segment(x, n):
    return x if len(x) == n else F.interpolate(x.T[None], size=n, mode="linear",
                                              align_corners=False)[0].T


def apply_edits(features, edits):
    result = features.clone()
    occupied = set()
    for edit in sorted(edits, key=lambda e: e.start):
        if not 0 <= edit.start < edit.end <= len(features):
            raise ValueError("Edit outside timeline.")
        region = set(range(edit.start, edit.end))
        if occupied & region:
            raise ValueError("Overlapping edits are not a valid finite repair.")
        occupied |= region
        result[edit.start:edit.end] = resize_segment(edit.replacement.to(features), edit.end - edit.start)
    return result


class NormalReference:
    def __init__(self, retrieval, calibration, config=None):
        self.config = config or ReferenceConfig()
        if not retrieval or len(calibration) < self.config.min_controls:
            raise ValueError("Not enough normal retrieval/calibration videos.")
        if any(v.binary_label != 0 for v in [*retrieval, *calibration]):
            raise ValueError("Normal reference must use labeled normal training videos only.")
        a, b = {v.source_id for v in retrieval}, {v.source_id for v in calibration}
        if a & b:
            raise ValueError("Retrieval/calibration source leakage.")
        # Enforce one retained sample per original video, not duplicate crop counts.
        if len(a) != len(retrieval) or len(b) != len(calibration):
            raise ValueError("Deduplicate normal reference by original source_id.")
        self.retrieval = [cpu_sample(v) for v in retrieval[:self.config.max_reference_videos]]
        self.calibration = [cpu_sample(v) for v in calibration[:self.config.max_reference_videos]]
        if len(self.calibration) < self.config.min_controls:
            raise ValueError("Reference cap leaves too few calibration videos.")
        deltas = []
        for v in self.retrieval:
            if len(v.features) > 1:
                deltas.append(1 - F.cosine_similarity(v.features[1:], v.features[:-1], dim=-1))
        self.transition_limit = min(2.0, (float(torch.quantile(torch.cat(deltas),
                                   self.config.transition_quantile)) if deltas else 0.0)
                                    + self.config.transition_margin)

    @property
    def source_ids(self):
        return {v.source_id for v in self.retrieval + self.calibration}

    def _interval(self, video, start, duration):
        goal = float(video.edges[start]) + duration
        end = int(torch.searchsorted(video.edges, video.edges.new_tensor(goal)))
        end = min(len(video.features), max(start + 1, end))
        got = float(video.edges[end] - video.edges[start])
        if abs(got - duration) / max(duration, 1e-6) > self.config.duration_tolerance:
            return None
        return start, end

    def retrieve(self, sample, start, end, group=0):
        if sample.source_id in self.source_ids:
            raise ValueError("Query source appears in its normal reference.")
        x = sample.features.detach().cpu().float()
        q = descriptor(x, start, end, self.config.context_steps)
        if q is None:
            return []
        duration = float(sample.edges[end] - sample.edges[start])
        candidates = []
        for v in self.retrieval:
            slots = torch.linspace(0, len(v.features) - 1, min(len(v.features),
                                   self.config.positions_per_video)).long().unique().tolist()
            for a in slots:
                interval = self._interval(v, a, duration)
                if interval is None:
                    continue
                a, b = interval
                d = descriptor(v.features, a, b, self.config.context_steps)
                if d is None:
                    continue
                distance = float(1 - q @ d)
                if distance <= self.config.max_context_distance:
                    candidates.append((distance, v.source_id, a, b, v.features[a:b]))
        candidates.sort(key=lambda z: (z[0], z[1], z[2]))
        return [Edit(-1, group, start, end, segment, sid, max(0.0, dist), duration)
                for dist, sid, _, _, segment in candidates[:self.config.retrievals]]

    def legal(self, features, edits):
        if not edits:
            return True
        features = features.float()
        changed = apply_edits(features, edits)
        for edit in edits:
            a, b = max(0, edit.start - 1), min(len(features), edit.end + 1)
            if b - a > 1:
                d = 1 - F.cosine_similarity(changed[a + 1:b], changed[a:b - 1], dim=-1, eps=1e-6)
                if float(d.max()) > self.transition_limit:
                    return False
        return True

    def controls(self, sample, edits):
        # Match the context not edited in the query; never use test category labels.
        keep = torch.ones(len(sample.features), dtype=torch.bool)
        for e in edits:
            keep[e.start:e.end] = False
        if not keep.any():
            return []
        q = F.normalize(sample.features.detach().cpu()[keep].mean(0), dim=0, eps=1e-6)
        choices = []
        for v in self.calibration:
            mapped = []
            for e in edits:
                # Same duration, similar relative location; do not silently change budget.
                fraction = float((sample.edges[e.start] - sample.edges[0]) / sample.duration)
                seconds = float(v.edges[0]) + fraction * max(0, v.duration - e.seconds)
                start = min(len(v.features) - 1, int(torch.searchsorted(v.edges, v.edges.new_tensor(seconds))))
                interval = self._interval(v, start, e.seconds)
                if interval is None:
                    mapped = []
                    break
                a, b = interval
                mapped.append(replace(e, start=a, end=b))
            if len(mapped) != len(edits):
                continue
            occupied = []
            for e in mapped:
                occupied.extend(range(e.start, e.end))
            if len(set(occupied)) != len(occupied) or not self.legal(v.features, mapped):
                continue
            mask = torch.ones(len(v.features), dtype=torch.bool)
            mask[occupied] = False
            if not mask.any():
                continue
            d = F.normalize(v.features[mask].mean(0), dim=0, eps=1e-6)
            distance = float(1 - q @ d)
            if distance <= self.config.max_context_distance:
                choices.append((distance, v, mapped))
        choices.sort(key=lambda x: (x[0], x[1].source_id))
        return [(v, es) for _, v, es in choices[:self.config.normal_controls]]

    def package(self):
        def pack(v):
            return {"source_id": v.source_id, "features": v.features, "edges": v.edges, "labels": v.labels}
        return {"config": asdict(self.config), "retrieval": [pack(v) for v in self.retrieval],
                "calibration": [pack(v) for v in self.calibration]}

    @classmethod
    def from_package(cls, item):
        def unpack(v):
            return VideoSample(v["source_id"], v["features"], v["edges"], 0, v["labels"])
        return cls([unpack(v) for v in item["retrieval"]], [unpack(v) for v in item["calibration"]],
                   ReferenceConfig(**item["config"]))
