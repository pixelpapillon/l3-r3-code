"""Independent-source retrieval with explicit native-clip coordinates."""

from dataclasses import dataclass

import torch

from scheme1_repair.reference import Edit, NormalReference, descriptor


@dataclass
class IndexedEdit(Edit):
    reference_start: int = 0
    reference_end: int = 0


class IndependentReference(NormalReference):
    def retrieve(self, sample, start, end, group=0):
        if sample.source_id in self.source_ids:
            raise ValueError("Query source appears in its normal reference")
        features = sample.features.detach().cpu().float()
        query = descriptor(features, start, end, self.config.context_steps)
        if query is None:
            return []
        duration = float(sample.edges[end] - sample.edges[start])
        candidates = []
        # Keep the best interval per independent original source, THEN top-k.
        for video in self.retrieval:
            best = None
            positions = torch.linspace(0, len(video.features) - 1,
                                       min(len(video.features), self.config.positions_per_video))
            for position in positions.long().unique().tolist():
                interval = self._interval(video, position, duration)
                if interval is None:
                    continue
                left, right = interval
                context = descriptor(video.features, left, right, self.config.context_steps)
                if context is None:
                    continue
                distance = float(1 - query @ context)
                if distance > self.config.max_context_distance:
                    continue
                candidate = (distance, video.source_id, left, right)
                if best is None or candidate < best[0]:
                    best = (candidate, video)
            if best is not None:
                candidates.append(best)
        candidates.sort(key=lambda value: value[0])
        return [IndexedEdit(
            -1, group, start, end, video.features[left:right], identity,
            max(0., distance), duration, left, right,
        ) for (distance, identity, left, right), video in candidates[:self.config.retrievals]]
