from dataclasses import dataclass, asdict
import hashlib
import itertools
import math
import random
import torch
from torch.nn import functional as F
from vadcore.backbone import bag_logits
from .reference import apply_edits


@dataclass
class RepairConfig:
    interval_fractions: tuple = (0.0625, 0.125, 0.25)
    interval_seconds: tuple = (1., 2., 4., 8.)
    max_groups: int = 8
    max_edits: int = 2
    max_edit_fraction: float = 0.4
    beam_width: int = 8
    max_evaluations: int = 128
    search: str = "beam"  # exact: exhaust configured finite edit space or fail
    evidence_threshold: float = 1.0
    z_scale: float = 4.0
    cost_weight: float = 0.1
    collateral_weight: float = 0.2
    shapley_samples: int = 32
    seed: int = 42
    use_null_calibration: bool = True
    use_selectivity: bool = True
    attribution: str = "coalitional"

    def __post_init__(self):
        if self.search not in ("beam", "exact") or self.attribution not in ("coalitional", "leave_one_out"):
            raise ValueError("Unknown repair search/attribution.")
        if min(self.max_groups, self.max_edits, self.beam_width, self.max_evaluations,
               self.shapley_samples) < 1 or not 0 < self.max_edit_fraction < 1 or self.z_scale <= 0:
            raise ValueError("Invalid finite repair budget.")


@dataclass
class RepairEvidence:
    base: torch.Tensor  # [T, 1+C] logits
    features: torch.Tensor  # [T, 1+C, 6]
    gate: torch.Tensor  # [T, 1+C], label-independent reference support
    diagnostics: dict


def coalition_attribution(group_count, entries, targets, samples=32, seed=42, mode="coalitional"):
    """Shapley-type attribution EXACTLY for the searched family, not all edits."""
    values = {}
    zero = torch.zeros(targets)
    def value(mask):
        if mask not in values:
            permitted = [u for groups, u in entries if groups & ~mask == 0]
            values[mask] = torch.stack([zero, *permitted]).amax(0)
        return values[mask]
    phi = torch.zeros(group_count, targets)
    full = (1 << group_count) - 1
    if mode == "leave_one_out":
        for group in range(group_count):
            phi[group] = value(full) - value(full ^ (1 << group))
        return phi, value(full)
    rng = random.Random(seed)
    for _ in range(samples):
        order = list(range(group_count))
        rng.shuffle(order)
        mask = 0
        for group in order:
            nxt = mask | (1 << group)
            phi[group] += value(nxt) - value(mask)
            mask = nxt
    return phi / samples, value(full)


class SelectiveRepair:
    def __init__(self, probe, reference, config=None, trained_source_ids=()):
        self.probe = probe.eval().requires_grad_(False)
        self.reference = reference
        self.config = config or RepairConfig()
        self.trained_source_ids = set(trained_source_ids)

    @property
    def device(self):
        return next(self.probe.parameters()).device

    def _output(self, features):
        return {k: v[0].float().cpu() for k, v in self.probe(features.to(self.device)).items()}

    def _bag(self, features):
        return bag_logits(self._output(features), self.probe.config.topk_divisor)

    def _intervals(self, sample, base):
        n = len(sample.features)
        sizes = {max(1, round(n * f)) for f in self.config.interval_fractions}
        dt = sample.duration / n
        sizes |= {max(1, round(s / dt)) for s in self.config.interval_seconds}
        intervals = set()
        for size in sorted(sizes):
            if size >= n:
                continue
            for start in range(0, n - size + 1, max(1, size // 2)):
                end = start + size
                if float(sample.edges[end] - sample.edges[start]) <= sample.duration * self.config.max_edit_fraction:
                    intervals.add((start, end))
        intervals = sorted(intervals)
        if len(intervals) <= self.config.max_groups:
            return intervals
        half = max(1, self.config.max_groups // 2)
        # Uniform candidates give missed events a chance, not only peak refinement.
        chosen = {intervals[i] for i in torch.linspace(0, len(intervals) - 1, half).long().tolist()}
        ranked = sorted(intervals, key=lambda ab: -float(base[ab[0]:ab[1]].amax()))
        for interval in ranked:
            chosen.add(interval)
            if len(chosen) == self.config.max_groups:
                break
        return sorted(chosen)

    @torch.no_grad()
    def extract(self, sample):
        if sample.source_id in self.trained_source_ids or sample.source_id in self.reference.source_ids:
            raise ValueError("Repair evidence requires held-out original-video queries.")
        cfg = self.config
        out = self._output(sample.features)
        base = torch.cat([out["binary"][:, None], out["classes"]], -1)
        base_bag = bag_logits(out, self.probe.config.topk_divisor)
        n, h = base.shape
        intervals = self._intervals(sample, base)
        edits = []
        for group, (a, b) in enumerate(intervals):
            for edit in self.reference.retrieve(sample, a, b, group):
                edit.uid = len(edits)
                edits.append(edit)
        failures = {"illegal": 0, "insufficient_controls": 0, "budget": 0}
        cache, baseline_controls = {}, {}
        budget = sample.duration * cfg.max_edit_fraction
        gate = torch.zeros(n, h)

        def permitted(combo):
            es = [edits[i] for i in combo]
            if sum(e.seconds for e in es) > budget:
                return False
            return all(a.end <= b.start or b.end <= a.start for a, b in itertools.combinations(es, 2))

        def evaluate(combo):
            key = tuple(sorted(combo))
            if key in cache:
                return cache[key]
            es = [edits[i] for i in key]
            if not self.reference.legal(sample.features, es):
                failures["illegal"] += 1
                cache[key] = None
                return None
            controls = self.reference.controls(sample, es)
            if len(controls) < self.reference.config.min_controls:
                failures["insufficient_controls"] += 1
                cache[key] = None
                return None
            changed = apply_edits(sample.features, es)
            response = base_bag - self._bag(changed)
            nulls = []
            for normal, remapped in controls:
                if normal.source_id not in baseline_controls:
                    baseline_controls[normal.source_id] = self._bag(normal.features)
                nulls.append(baseline_controls[normal.source_id] - self._bag(apply_edits(normal.features, remapped)))
            nulls = torch.stack(nulls)
            median = nulls.median(0).values
            scale = (nulls - median).abs().median(0).values * 1.4826
            scale = scale.clamp_min(self.reference.config.scale_floor)
            z = (response - median) / scale if cfg.use_null_calibration else response / self.reference.config.scale_floor
            normalized = (z / cfg.z_scale).clamp(-1, 1)
            keep = torch.ones(n, dtype=torch.bool)
            for edit in es:
                keep[edit.start:edit.end] = False
            collateral = torch.zeros(h)
            if cfg.use_selectivity and keep.any() and (~keep).any():
                # Only protect categories whose predicted support is disjoint.
                protected = (base[keep].sigmoid().amax(0) > .7) & (base[~keep].sigmoid().amax(0) < .3)
                protected[0] = False
                for c in range(1, h):
                    others = protected.clone()
                    others[c] = False
                    if others.any():
                        collateral[c] = normalized[others].clamp_min(0).mean()
            distance = float((1 - F.cosine_similarity(sample.features.float(), changed.float(),
                                                     dim=-1, eps=1e-6))[~keep.to(changed.device)].mean())
            cost = sum(e.seconds for e in es) / budget + max(0, distance) * .1
            utility = (normalized - cfg.cost_weight * cost - cfg.collateral_weight * collateral).clamp(0, 1)
            eligible = z >= cfg.evidence_threshold
            utility = utility * eligible
            quality = min(1., len(controls) / self.reference.config.normal_controls)
            for edit in es:
                gate[edit.start:edit.end] = torch.maximum(gate[edit.start:edit.end], torch.full_like(gate[edit.start:edit.end], quality))
            item = {"groups": sum(1 << g for g in {e.group for e in es}), "utility": utility,
                    "z": z, "cost": cost, "collateral": collateral, "distance": max(0, distance),
                    "controls": len(controls), "eligible": eligible, "edits": key}
            cache[key] = item
            return item

        current = [()]
        truncated = False
        for depth in range(1, cfg.max_edits + 1):
            if cfg.search == "exact":
                candidates = itertools.combinations(range(len(edits)), depth)
            else:
                candidates = sorted({tuple(sorted((*old, i))) for old in current
                                     for i in range(len(edits)) if i not in old})
            expanded = []
            for combo in candidates:
                if not permitted(combo):
                    failures["budget"] += 1
                    continue
                if combo not in cache and len(cache) >= cfg.max_evaluations:
                    if cfg.search == "exact":
                        raise RuntimeError("Exact finite search exceeds max_evaluations; increase budget.")
                    truncated = True
                    break
                item = evaluate(combo)
                if item is not None:
                    # Continue even below threshold to allow complementary edits.
                    rank = float((item["z"] / cfg.z_scale).clamp(-1, 1).max()) - cfg.cost_weight * item["cost"]
                    expanded.append((rank, combo))
            if cfg.search == "beam":
                current = [combo for _, combo in sorted(expanded, reverse=True)[:cfg.beam_width]]
                if not current or truncated:
                    break
        valid = [item for item in cache.values() if item is not None]
        seed = cfg.seed + int(hashlib.sha256(sample.source_id.encode()).hexdigest()[:8], 16)
        phi, full = coalition_attribution(len(intervals), [(x["groups"], x["utility"]) for x in valid],
                                         h, cfg.shapley_samples, seed, cfg.attribution)
        features = torch.zeros(n, h, 6)
        counts = torch.zeros(n, h, 1)
        for group, (a, b) in enumerate(intervals):
            covering = [x for x in valid if x["groups"] & (1 << group)]
            if not covering:
                continue
            z = torch.stack([x["z"] for x in covering]).amax(0).clamp(-cfg.z_scale, cfg.z_scale) / cfg.z_scale
            cost = min(x["cost"] for x in covering)
            distance = min(x["distance"] for x in covering)
            row = torch.stack([phi[group], z, full, torch.full((h,), cost),
                               torch.full((h,), distance), gate[a:b].mean(0)], -1)
            features[a:b] += row[None]
            counts[a:b] += 1
        features /= counts.clamp_min(1)
        minimal = []
        for c in range(h):
            possible = [x for x in valid if x["eligible"][c]]
            chosen = min(possible, key=lambda x: x["cost"]) if possible else None
            minimal.append(None if chosen is None else {
                "cost": chosen["cost"], "z": float(chosen["z"][c]),
                "intervals": [[edits[i].start, edits[i].end] for i in chosen["edits"]]})
        return RepairEvidence(base, features, gate, {
            "coverage": float((gate > 0).float().mean()), "evaluations": len(cache),
            "valid_edits": len(valid), "failures": failures, "search_truncated": truncated,
            "search": cfg.search, "attribution_scope": "searched_finite_family",
            "minimal_repairs": minimal, "group_contributions": phi.tolist(),
            "family_value": full.tolist(), "used_test_labels": False,
        })
