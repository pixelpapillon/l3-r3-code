"""Build paired, signed DSANet response caches from existing OOF teachers.

The output is deliberately incompatible with the legacy response cache.  It
stores valid lengths, exact replacement tensors, matched-null offsets and
empirical signed envelopes so the training loader can reconstruct the actual
edited input rather than fitting a synthetic frame target.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from scheme1_repair.reference import (
    NormalReference, ReferenceConfig, apply_edits, resize_segment,
)
from vadcore.types import VideoSample

from .adapter import raw_dsanet_logits
from .data import UCF_LABEL_MAP, XD_LABEL_MAP, label_vector, process_feature, row_id, source_group
from .decision_constraints import topk_pool
from .numerics import clear_text_cache
from .train_full import load_initial_state, make_backbone, official_runtime, setup_seed
from .v2_cache import PairedEdit, V2ResponseItem, save_v2_cache


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sample(path, label, label_map, visual_length):
    values, length = process_feature(np.load(path), visual_length)
    labels = label_vector(label, label_map)
    normal = bool(labels[0] == 1)
    anomaly_labels = labels.clone()
    anomaly_labels[0] = 0
    features = torch.from_numpy(values[:length]).float()
    return VideoSample(
        row_id(path), features, torch.arange(length + 1, dtype=torch.float32),
        int(not normal), anomaly_labels,
    )


def unique_normal_samples(frame, label_map, visual_length, limit, seed):
    normal_token = next(iter(label_map))
    normal = frame[frame["label"].map(lambda x: str(x).split("-")[0] == normal_token)].copy()
    normal["group"] = normal["path"].map(source_group)
    normal = normal.sort_values("path").drop_duplicates("group")
    indices = list(range(len(normal)))
    random.Random(seed).shuffle(indices)
    normal = normal.iloc[indices[:limit]]
    return [valid_sample(row.path, row.label, label_map, visual_length)
            for row in normal.itertuples(index=False)]


def select_query_rows(frame, label_map, limit, seed):
    """One crop per original source, then deterministic normal/anomaly stratification."""
    frame = frame.copy()
    frame["_group"] = frame["path"].map(source_group)
    frame = frame.sort_values("path").drop_duplicates("_group").drop(columns="_group")
    if limit is None or len(frame) <= int(limit):
        return frame.reset_index(drop=True)
    normal_token = next(iter(label_map))
    normal = frame[frame["label"].map(lambda x: str(x).split("-")[0] == normal_token)]
    anomaly = frame.drop(normal.index)
    normal_indices, anomaly_indices = list(normal.index), list(anomaly.index)
    random.Random(seed).shuffle(normal_indices)
    random.Random(seed + 1009).shuffle(anomaly_indices)
    target_normal = min(len(normal_indices), int(limit) // 2)
    target_anomaly = min(len(anomaly_indices), int(limit) - target_normal)
    selected = normal_indices[:target_normal] + anomaly_indices[:target_anomaly]
    if len(selected) < int(limit):
        remainder = [index for index in normal_indices[target_normal:] + anomaly_indices[target_anomaly:]
                     if index not in selected]
        selected.extend(remainder[:int(limit) - len(selected)])
    return frame.loc[selected].sort_values("path").reset_index(drop=True)


@torch.no_grad()
def decisions(model, prompt, sequences, visual_length, device, divisor=16):
    if not sequences:
        raise ValueError("At least one sequence is required")
    width = sequences[0].shape[1]
    if any(x.ndim != 2 or x.shape[1] != width or not 0 < len(x) <= visual_length
           for x in sequences):
        raise ValueError("Invalid variable-length feature batch")
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long, device=device)
    batch = torch.zeros(len(sequences), visual_length, width, device=device)
    for index, sequence in enumerate(sequences):
        batch[index, :len(sequence)] = sequence.to(device)
    output = model(batch, None, prompt, lengths, False)
    binary, classes = raw_dsanet_logits(output)
    logits = torch.cat([binary.unsqueeze(-1), classes], dim=-1)
    bags = topk_pool(logits, lengths, divisor)
    rows = [logits[index, :int(length)].detach().float().cpu()
            for index, length in enumerate(lengths)]
    return rows, bags.detach().float().cpu()


def candidate_intervals(sample, base, labels, fractions, max_groups):
    length = len(sample.features)
    sizes = sorted({max(1, round(length * value)) for value in fractions})
    intervals = set()
    for size in sizes:
        if size >= length:
            continue
        for start in range(0, length - size + 1, max(1, size // 2)):
            intervals.add((start, start + size))
    intervals = sorted(intervals)
    if len(intervals) <= max_groups:
        return intervals
    channels = torch.cat([torch.ones(1), labels[1:]]).bool()
    if not channels.any():
        channels[0] = True
    ranked = sorted(
        intervals,
        key=lambda interval: (
            -float(base[interval[0]:interval[1], channels].amax()),
            interval[0], interval[1],
        ),
    )
    # Preserve one timeline-spread hypothesis instead of selecting peaks only.
    chosen = {intervals[index] for index in
              torch.linspace(0, len(intervals) - 1, min(2, max_groups)).long().tolist()}
    for interval in ranked:
        chosen.add(interval)
        if len(chosen) >= max_groups:
            break
    return sorted(chosen)


def paired_edit_for_interval(sample, interval, reference, teacher, prompt, base_bag,
                             labels, visual_length, device, args, normal_bag_cache):
    start, end = interval
    donors = reference.retrieve(sample, start, end)
    records = []
    failures = {"illegal": 0, "controls": 0}
    for edit in donors:
        if not reference.legal(sample.features, [edit]):
            failures["illegal"] += 1
            continue
        controls = reference.controls(sample, [edit])
        if len(controls) < args.min_controls:
            failures["controls"] += 1
            continue
        sequences = [apply_edits(sample.features, [edit])]
        control_rows = []
        for normal, remapped in controls:
            if normal.source_id not in normal_bag_cache:
                control_rows.append(normal.features)
            sequences.append(apply_edits(normal.features, remapped))
        # Compute missing normal baselines in one batch before the edited batch.
        if control_rows:
            _, normal_bags = decisions(
                teacher, prompt, control_rows, visual_length, device, args.topk_divisor
            )
            for normal, _ in controls:
                if normal.source_id not in normal_bag_cache:
                    normal_bag_cache[normal.source_id] = normal_bags[0]
                    normal_bags = normal_bags[1:]
        _, changed_bags = decisions(
            teacher, prompt, sequences, visual_length, device, args.topk_divisor
        )
        raw_response = base_bag - changed_bags[0]
        nulls = torch.stack([
            normal_bag_cache[normal.source_id] - changed_bags[index + 1]
            for index, (normal, _) in enumerate(controls)
        ])
        corrected = raw_response.unsqueeze(0) - nulls
        records.append({
            "edit": edit, "raw": raw_response, "nulls": nulls,
            "corrected": corrected, "controls": len(controls),
        })
    if len(records) < args.min_donors:
        return None, {**failures, "donors": len(records)}

    samples = torch.cat([record["corrected"] for record in records], dim=0)
    lower = torch.quantile(samples, args.lower_quantile, dim=0)
    upper = torch.quantile(samples, args.upper_quantile, dim=0)
    center = samples.median(dim=0).values
    representative = min(
        records,
        key=lambda record: float(((record["raw"] - record["nulls"].median(0).values) - center).square().mean()),
    )
    null_response = representative["nulls"].median(0).values
    donor_quality = min(1.0, len(records) / args.retrievals)
    control_quality = min(1.0, min(record["controls"] for record in records) / args.normal_controls)
    stability = (1.0 / (1.0 + (upper - lower).abs())).clamp(0.1, 1.0)
    present = torch.cat([torch.ones(1), labels[1:]]).float()
    reliability = stability * donor_quality * control_quality * present
    edit = representative["edit"]
    result = PairedEdit(
        start, end, resize_segment(edit.replacement, end - start).float().cpu(),
        lower.float(), upper.float(), reliability.float(), null_response.float(),
        {
            "reference_id": edit.reference_id,
            "donor_ids": sorted({record["edit"].reference_id for record in records}),
            "donors": len(records),
            "controls_min": min(record["controls"] for record in records),
            "context_distance": float(edit.context_distance),
            "center": center.tolist(),
            "envelope_width": (upper - lower).tolist(),
        },
    ).validate(sample.features.shape[1], len(base_bag), len(sample.features))
    score = float((center.abs() * reliability).max() + 0.1 * reliability.mean())
    return (score, result), {**failures, "donors": len(records)}


def generate(args):
    setup_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest.get("format") != "dsanet-oof-split-v1" or manifest.get("dataset") != args.dataset:
        raise ValueError("OOF manifest does not match the requested dataset")
    if int(manifest.get("seed", -1)) != args.seed:
        raise ValueError("OOF manifest seed mismatch")

    root, cfg, _test, model_class, _data, tools, _stable = official_runtime(
        args.dsanet_root, args.dataset
    )
    label_map = UCF_LABEL_MAP if args.dataset == "ucf" else XD_LABEL_MAP
    prompt = tools.get_prompt_text(label_map)
    anchor = make_backbone(model_class, cfg, device, args.numerical_mode).to(device)
    anchor.load_state_dict(load_initial_state(args.anchor_checkpoint), strict=True)
    anchor.eval().requires_grad_(False)
    clear_text_cache(anchor)

    all_items = []
    fold_diagnostics = []
    seen = set()
    for record in manifest["records"]:
        fold = int(record["fold"])
        checkpoint = Path(args.teacher_pattern.format(fold=fold))
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing OOF teacher: {checkpoint}")
        teacher = make_backbone(model_class, cfg, device, args.numerical_mode).to(device)
        teacher.load_state_dict(load_initial_state(checkpoint), strict=True)
        teacher.eval().requires_grad_(False)
        clear_text_cache(teacher)

        train_frame = pd.read_csv(record["train_csv"])
        normal = unique_normal_samples(
            train_frame, label_map, cfg.visual_length,
            args.reference_videos + args.calibration_videos, args.seed + fold,
        )
        if len(normal) < args.reference_videos + args.calibration_videos:
            raise ValueError(f"Fold {fold} has too few independent normal videos")
        calibration = normal[:args.calibration_videos]
        retrieval = normal[args.calibration_videos:]
        reference = NormalReference(
            retrieval, calibration,
            ReferenceConfig(
                retrievals=args.retrievals,
                positions_per_video=args.positions_per_video,
                context_steps=args.context_steps,
                max_context_distance=args.max_context_distance,
                duration_tolerance=args.duration_tolerance,
                normal_controls=args.normal_controls,
                min_controls=args.min_controls,
                max_reference_videos=max(args.reference_videos, args.calibration_videos),
            ),
        )
        query = select_query_rows(
            pd.read_csv(record["query_csv"]), label_map, args.max_queries, args.seed + fold,
        )
        normal_bag_cache = {}
        coverages = []
        for index, row in enumerate(query.itertuples(index=False)):
            sample = valid_sample(row.path, row.label, label_map, cfg.visual_length)
            if sample.source_id in seen:
                raise ValueError(f"Duplicate OOF query row: {sample.source_id}")
            seen.add(sample.source_id)
            labels = label_vector(row.label, label_map)
            anchor_rows, _ = decisions(
                anchor, prompt, [sample.features], cfg.visual_length, device, args.topk_divisor
            )
            edits = []
            diagnostics = {
                "fold": fold,
                "normal": bool(labels[0] == 1),
                "source_group": source_group(row.path),
                "candidates": 0,
            }
            if not bool(labels[0] == 1):
                teacher_rows, teacher_bags = decisions(
                    teacher, prompt, [sample.features], cfg.visual_length, device, args.topk_divisor
                )
                intervals = candidate_intervals(
                    sample, teacher_rows[0], labels, args.interval_fractions, args.max_groups
                )
                ranked = []
                failures = []
                for interval in intervals:
                    result, failure = paired_edit_for_interval(
                        sample, interval, reference, teacher, prompt, teacher_bags[0],
                        labels, cfg.visual_length, device, args, normal_bag_cache,
                    )
                    failures.append({"interval": list(interval), **failure})
                    if result is not None:
                        ranked.append(result)
                ranked.sort(key=lambda value: (-value[0], value[1].start, value[1].end))
                edits = [value[1] for value in ranked[:args.queries_per_video]]
                diagnostics.update({
                    "candidates": len(intervals), "supported_candidates": len(ranked),
                    "failures": failures,
                })
            coverage = sum(bool(edit.reliability.sum()) for edit in edits) / args.queries_per_video
            diagnostics["coverage"] = coverage
            coverages.append(coverage)
            all_items.append(V2ResponseItem(
                sample.source_id, len(sample.features), sample.features.shape[1],
                anchor_rows[0], tuple(edits), diagnostics,
            ).validate(cfg.visual_length, len(label_map)))
            if index % args.log_every == 0:
                print(json.dumps({
                    "phase": "v2_cache", "fold": fold, "query": index,
                    "rows": len(query), "edits": len(edits), "coverage": coverage,
                }), flush=True)
        fold_diagnostics.append({
            "fold": fold, "queries": len(query),
            "mean_coverage": sum(coverages) / max(1, len(coverages)),
        })
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "dataset": args.dataset, "seed": args.seed, "folds": manifest["folds"],
        "visual_length": int(cfg.visual_length), "channels": len(label_map),
        "queries_per_video": args.queries_per_video,
        "numerical_mode": args.numerical_mode,
        "anchor_checkpoint": str(Path(args.anchor_checkpoint).resolve()),
        "anchor_sha256": file_hash(args.anchor_checkpoint),
        "teacher_pattern": args.teacher_pattern,
        "manifest": str(Path(args.manifest).resolve()),
        "response_units": "raw binary logit plus abnormal-minus-normal margins",
        "bounds": f"empirical quantiles [{args.lower_quantile},{args.upper_quantile}] over donor/control responses",
        "query_selection": "one deterministic row per original source group, normal/anomaly stratified",
        "max_queries_per_fold": args.max_queries,
        "fold_diagnostics": fold_diagnostics,
    }
    save_v2_cache(args.output, all_items, metadata)
    print(json.dumps({
        "phase": "v2_cache_complete", "output": str(Path(args.output).resolve()),
        "items": len(all_items), "folds": fold_diagnostics,
    }, indent=2), flush=True)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", choices=("ucf", "xd"), required=True)
    value.add_argument("--dsanet-root", default="third_party/DSANet")
    value.add_argument("--manifest", required=True)
    value.add_argument("--teacher-pattern", required=True)
    value.add_argument("--anchor-checkpoint", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--seed", type=int, default=234)
    value.add_argument("--device", default="auto")
    value.add_argument("--numerical-mode", choices=("legacy", "stable"), default="stable")
    value.add_argument("--reference-videos", type=int, default=64)
    value.add_argument("--calibration-videos", type=int, default=32)
    value.add_argument("--retrievals", type=int, default=3)
    value.add_argument("--min-donors", type=int, default=2)
    value.add_argument("--normal-controls", type=int, default=8)
    value.add_argument("--min-controls", type=int, default=6)
    value.add_argument("--positions-per-video", type=int, default=16)
    value.add_argument("--context-steps", type=int, default=4)
    value.add_argument("--max-context-distance", type=float, default=0.8)
    value.add_argument("--duration-tolerance", type=float, default=0.35)
    value.add_argument("--interval-fractions", type=float, nargs="+", default=(0.0625, 0.125, 0.25))
    value.add_argument("--max-groups", type=int, default=4)
    value.add_argument("--queries-per-video", type=int, default=2)
    value.add_argument("--lower-quantile", type=float, default=0.1)
    value.add_argument("--upper-quantile", type=float, default=0.9)
    value.add_argument("--topk-divisor", type=int, default=16)
    value.add_argument("--max-queries", type=int)
    value.add_argument("--log-every", type=int, default=25)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    if not 0 <= arguments.lower_quantile < arguments.upper_quantile <= 1:
        raise ValueError("Invalid response-envelope quantiles")
    if arguments.min_donors > arguments.retrievals:
        raise ValueError("min_donors exceeds retrievals")
    generate(arguments)
