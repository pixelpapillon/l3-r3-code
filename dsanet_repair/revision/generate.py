"""Measure every training crop using existing OOF teachers (no teacher training)."""

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random
import time

import pandas as pd
import torch

from dsanet_repair.adapter import raw_dsanet_logits
from dsanet_repair.data import UCF_LABEL_MAP, XD_LABEL_MAP, source_group
from dsanet_repair.generate_v2_cache import candidate_intervals
from dsanet_repair.numerics import clear_text_cache
from dsanet_repair.train_full import official_runtime, make_backbone, load_initial_state, setup_seed
from dsanet_repair.train_v2 import file_hash
from scheme1_repair.reference import ReferenceConfig, apply_edits
from vadcore.types import VideoSample

from .cache import (FORMAT, PREPROCESS, SCOPES, crop_key, fingerprint, load_feature,
                    bank_fingerprint, checked_labels, measurement_signature, upstream_signature,
                    save_atomic, validate_item, write_rows)
from .reference import IndependentReference
from .response import scoped_responses


def canonical_rows(frame, available):
    result = frame.copy()
    paths = []
    for path in frame.path:
        key = crop_key(path)
        if key not in available:
            raise ValueError(f"OOF row is not in the actual training CSV: {key}")
        paths.append(available[key])
    result["path"] = paths
    return result


def sample_row(row, labels, visual_length):
    features, length, edges = load_feature(row.path, visual_length)
    vector = checked_labels(row.label, labels)
    anomaly = vector.clone()
    anomaly[0] = 0
    sample = VideoSample(source_group(row.path), torch.from_numpy(features[:length]),
                         edges, int(vector[0] != 1), anomaly)
    return sample, vector, fingerprint(features, length)


@torch.no_grad()
def infer(model, sequences, prompt, visual_length, device, batch_size=32):
    rows = []
    for offset in range(0, len(sequences), batch_size):
        chunk = sequences[offset:offset + batch_size]
        lengths = torch.tensor([len(value) for value in chunk], device=device)
        inputs = torch.zeros(len(chunk), visual_length, chunk[0].shape[1], device=device)
        for index, value in enumerate(chunk):
            inputs[index, :len(value)] = value.to(device)
        output = model(inputs, None, prompt, lengths, False)
        binary, classes = raw_dsanet_logits(output)
        raw = torch.cat([binary[..., None], classes], -1).float().cpu()
        if not torch.isfinite(raw).all():
            raise FloatingPointError("Non-finite measured decisions")
        rows.extend(raw[index, :len(value)].clone() for index, value in enumerate(chunk))
    return rows


def response(before, after, interval, divisor):
    return scoped_responses(before[None], after[None, None],
                            torch.tensor([len(before)]), torch.tensor([[interval]]), divisor)[0, 0]


def edit_pointer(edit):
    return {"start": int(edit.start), "end": int(edit.end), "donor": edit.reference_id,
            "donor_start": int(edit.reference_start), "donor_end": int(edit.reference_end)}


def measure_interval(sample, interval, reference, teacher, prompt, base, labels,
                     cfg, device, args, baseline_cache, changed_cache):
    records, sequences, pending = [], [], {}
    for edit in reference.retrieve(sample, *interval):
        if not reference.legal(sample.features, [edit]):
            continue
        controls = reference.controls(sample, [edit])
        if len(controls) < args.min_controls:
            continue
        record = {"edit": edit, "controls": controls, "query_index": len(sequences), "keys": []}
        sequences.append(apply_edits(sample.features, [edit]))
        for normal, mapped in controls:
            mapped_edit = mapped[0]
            key = (normal.source_id, mapped_edit.start, mapped_edit.end,
                   edit.reference_id, edit.reference_start, edit.reference_end)
            record["keys"].append(key)
            if key not in changed_cache and key not in pending:
                pending[key] = len(sequences)
                sequences.append(apply_edits(normal.features, mapped))
        records.append(record)
    if len(records) < args.min_donors:
        return None
    if len({record["edit"].reference_id for record in records}) != len(records):
        raise AssertionError("Donor count must count independent original sources")
    values = infer(teacher, sequences, prompt, cfg.visual_length, device, args.infer_batch_size)
    # Resolve this interval before applying the bounded cross-query memo eviction.
    current = {key: values[position] for key, position in pending.items()}
    for record in records:
        record["raw"] = response(base, values[record["query_index"]], interval, args.topk_divisor)
        nulls = []
        for (normal, mapped), key in zip(record["controls"], record["keys"]):
            changed = current[key] if key in current else changed_cache[key]
            nulls.append(response(baseline_cache[normal.source_id], changed,
                                  (mapped[0].start, mapped[0].end), args.topk_divisor))
        record["nulls"] = torch.stack(nulls)
        record["corrected"] = record["raw"][None] - record["nulls"]
    changed_cache.update(current)
    while len(changed_cache) > args.null_cache_entries:
        changed_cache.popitem(last=False)
    samples = torch.cat([record["corrected"] for record in records])
    lower = torch.quantile(samples, args.lower_quantile, dim=0)
    upper = torch.quantile(samples, args.upper_quantile, dim=0)
    center = samples.median(0).values
    # Preserve v2's representative/query selection on the GLOBAL response.
    representative = min(records, key=lambda record: float(
        (record["raw"][0] - record["nulls"][:, 0].median(0).values - center[0]).square().mean()))
    donor_quality = len(records) / args.retrievals
    control_quality = min(len(record["controls"]) for record in records) / args.normal_controls
    present = torch.cat([torch.ones(1), labels[1:]])
    weight = (1 / (1 + upper - lower)).clamp(.1, 1) * donor_quality * control_quality * present
    result = {**edit_pointer(representative["edit"]),
              "lower": lower, "upper": upper, "weight": weight,
              "null": representative["nulls"].median(0).values, "normal_weight": 0.,
              "donor_ids": [record["edit"].reference_id for record in records],
              "controls_min": min(len(record["controls"]) for record in records),
              "context_distance": float(representative["edit"].context_distance)}
    score = float((center[0].abs() * weight[0]).max() + .1 * weight[0].mean())
    return score, result


def normal_edits(sample, reference, channels, queries):
    """Augment known-normal videos, without fabricating absent-class frame labels."""
    length = len(sample.features)
    if length < 3:
        return []
    size = max(1, length // 8)
    starts = torch.linspace(0, length - size, queries + 2).long().tolist()[1:-1]
    result = []
    for start in starts:
        donors = [edit for edit in reference.retrieve(sample, start, start + size)
                  if reference.legal(sample.features, [edit])]
        if not donors:
            continue
        edit = donors[0]
        zeros = torch.zeros(len(SCOPES), channels)
        result.append({**edit_pointer(edit), "lower": zeros.clone(), "upper": zeros.clone(),
                       "weight": zeros.clone(), "null": zeros.clone(),
                       "normal_weight": min(1., len(donors) / reference.config.retrievals),
                       "donor_ids": [value.reference_id for value in donors], "controls_min": 0})
    return result


def generate(args):
    if not (0 <= args.lower_quantile < args.upper_quantile <= 1 and
            1 <= args.min_donors <= args.retrievals <= args.reference_videos and
            1 <= args.min_controls <= args.normal_controls <= args.calibration_videos and
            min(args.queries_per_video, args.max_groups, args.infer_batch_size, args.null_cache_entries,
                args.commit_every, args.log_every, args.topk_divisor) > 0):
        raise ValueError("Invalid measurement settings")
    setup_seed(args.seed)
    started = time.monotonic()
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    root, cfg, _, cls, _, tools, _ = official_runtime(args.dsanet_root, args.dataset)
    labels = UCF_LABEL_MAP if args.dataset == "ucf" else XD_LABEL_MAP
    prompt = tools.get_prompt_text(labels)
    train_path = Path(args.train_list or cfg.train_list)
    if not train_path.is_absolute():
        train_path = root / train_path
    full_frame = pd.read_csv(train_path)
    keys = full_frame.path.map(crop_key)
    if keys.duplicated().any():
        raise ValueError("Exact crop names are not unique in training CSV")
    available = dict(zip(keys, full_frame.path))
    actual_labels = dict(zip(keys, full_frame.label.map(str)))
    for label in set(actual_labels.values()):
        checked_labels(label, labels)
    manifest = json.loads(Path(args.manifest).read_text())
    if (manifest.get("format") != "dsanet-oof-split-v1" or
            manifest.get("dataset") != args.dataset or int(manifest.get("seed", -1)) != args.seed):
        raise ValueError("Wrong OOF manifest dataset/seed/format")
    if (len({record["fold"] for record in manifest["records"]}) != len(manifest["records"]) or
            len(manifest["records"]) != manifest.get("folds")):
        raise ValueError("OOF manifest has missing or duplicate folds")
    for record in manifest["records"]:
        for part in ("train", "query"):
            if file_hash(record[f"{part}_csv"]) != record.get(f"{part}_sha256"):
                raise ValueError("OOF split CSV changed since its manifest; verify teacher provenance first")
    checkpoints = {str(record["fold"]): str(Path(args.teacher_pattern.format(fold=record["fold"])))
                   for record in manifest["records"]}
    settings = {key: value for key, value in vars(args).items()
                if key not in ("output_dir", "device", "log_every", "commit_every")}
    settings = json.loads(json.dumps(settings))  # tuples become lists, exactly as in index.json
    metadata = {"dataset": args.dataset, "seed": args.seed, "preprocess": PREPROCESS,
                "class_order": list(labels), "visual_length": int(cfg.visual_length),
                "queries_per_video": args.queries_per_video, "scopes": list(SCOPES),
                "numerical_mode": "stable", "anchor_sha256": file_hash(args.anchor_checkpoint),
                "teacher_sha256": {fold: file_hash(path) for fold, path in checkpoints.items()},
                "fold_csv_sha256": {str(record["fold"]): {name: file_hash(record[name])
                                       for name in ("train_csv", "query_csv")}
                                    for record in manifest["records"]},
                "manifest_sha256": file_hash(args.manifest), "train_csv_sha256": file_hash(train_path),
                "generation_config": settings, "time_units": "original cached-CLIP clip indices, NOT seconds",
                "measurement_source": measurement_signature(),
                "upstream_source": upstream_signature(root),
                "bounds": "empirical donor/control quantiles, not guaranteed confidence intervals",
                "normal_edits": "label-preserving augmentation assumption, used only by integrated variant"}
    metadata["identity"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index.get("format") != FORMAT or index.get("metadata") != metadata:
            raise ValueError("Existing cache belongs to a different configuration; use a new directory")
        if index.get("complete"):
            if set(index["rows"]) != set(available):
                raise ValueError("Completed cache does not cover this training CSV exactly")
            print(json.dumps({"phase": "cache_exists", "rows": len(index["rows"])}), flush=True)
            return
    else:
        if any(output.iterdir()):
            raise FileExistsError("Nonempty cache directory has no revision index; use a new directory")
        index = {"format": FORMAT, "complete": False, "metadata": metadata, "rows": {}}
        save_atomic(index, index_path)
    anchor = make_backbone(cls, cfg, device, "stable").to(device)
    anchor.load_state_dict(load_initial_state(args.anchor_checkpoint), strict=True)
    anchor.eval().requires_grad_(False)
    clear_text_cache(anchor)
    seen = set()
    for record in manifest["records"]:
        fold = int(record["fold"])
        train_frame = canonical_rows(pd.read_csv(record["train_csv"]), available)
        query_frame = canonical_rows(pd.read_csv(record["query_csv"]), available).sort_values("path")
        for frame in (train_frame, query_frame):
            if frame.path.map(crop_key).duplicated().any() or any(
                    str(row.label) != actual_labels[crop_key(row.path)] for row in frame.itertuples()):
                raise ValueError("OOF crop labels/duplicates disagree with the actual CSV")
        if set(train_frame.path.map(source_group)) & set(query_frame.path.map(source_group)):
            raise ValueError("OOF source-group overlap")
        if seen & set(query_frame.path.map(crop_key)):
            raise ValueError("Repeated query crop across OOF folds")
        seen.update(query_frame.path.map(crop_key))
        normal = train_frame[train_frame.label.map(lambda value: str(value).split("-")[0] == next(iter(labels)))].copy()
        normal["group"] = normal.path.map(source_group)
        normal = normal.sort_values("path").drop_duplicates("group")
        order = list(range(len(normal)))
        random.Random(args.seed + fold).shuffle(order)
        total = args.reference_videos + args.calibration_videos
        if len(order) < total:
            raise ValueError("Insufficient independent normal reference videos")
        samples = [sample_row(row, labels, cfg.visual_length)[0]
                   for row in normal.iloc[order[:total]].itertuples(index=False)]
        reference_digest = bank_fingerprint({sample.source_id: {"features": sample.features, "edges": sample.edges}
                                             for sample in samples})
        previous_digest = index.setdefault("references", {}).get(str(fold))
        if previous_digest is not None and previous_digest != reference_digest:
            raise ValueError("Retrieval/calibration inputs changed while resuming measurements")
        index["references"][str(fold)] = reference_digest
        calibration, retrieval = samples[:args.calibration_videos], samples[args.calibration_videos:]
        reference = IndependentReference(retrieval, calibration, ReferenceConfig(
            retrievals=args.retrievals, positions_per_video=args.positions_per_video,
            context_steps=args.context_steps, max_context_distance=args.max_context_distance,
            duration_tolerance=args.duration_tolerance, normal_controls=args.normal_controls,
            min_controls=args.min_controls, max_reference_videos=max(total, 128),
        ))
        bank = {sample.source_id: {"features": sample.features, "edges": sample.edges}
                for sample in reference.retrieval}
        bank_name, shard_name = f"fold_{fold}_bank.pt", f"fold_{fold}.sqlite"
        if (output / bank_name).exists():
            previous = torch.load(output / bank_name, map_location="cpu", weights_only=False)
            if bank_fingerprint(previous) != bank_fingerprint(bank):
                raise ValueError("Normal donor features changed while resuming this cache")
        else:
            save_atomic(bank, output / bank_name)
        index.setdefault("banks", {})[bank_name] = bank_fingerprint(bank)
        todo = query_frame[~query_frame.path.map(crop_key).isin(index["rows"])]
        if not len(todo):
            continue
        teacher = make_backbone(cls, cfg, device, "stable").to(device)
        teacher.load_state_dict(load_initial_state(checkpoints[str(fold)]), strict=True)
        teacher.eval().requires_grad_(False)
        clear_text_cache(teacher)
        values = infer(teacher, [sample.features for sample in reference.calibration], prompt,
                       cfg.visual_length, device, args.infer_batch_size)
        baseline_cache = dict(zip([sample.source_id for sample in reference.calibration], values))
        changed_cache, pending = OrderedDict(), {}
        for number, row in enumerate(todo.itertuples(index=False), 1):
            sample, vector, digest = sample_row(row, labels, cfg.visual_length)
            base_anchor = infer(anchor, [sample.features], prompt, cfg.visual_length,
                                device, args.infer_batch_size)[0]
            if sample.binary_label:
                base = infer(teacher, [sample.features], prompt, cfg.visual_length,
                             device, args.infer_batch_size)[0]
                intervals = candidate_intervals(sample, base, vector, args.interval_fractions, args.max_groups)
                measured = [measure_interval(sample, interval, reference, teacher, prompt, base, vector,
                                             cfg, device, args, baseline_cache, changed_cache)
                            for interval in intervals]
                measured = sorted([value for value in measured if value is not None],
                                  key=lambda value: (-value[0], value[1]["start"], value[1]["end"]))
                edits = [value[1] for value in measured[:args.queries_per_video]]
            else:
                edits = normal_edits(sample, reference, len(labels), args.queries_per_video)
            item = {"length": len(sample.features), "fingerprint": digest,
                    "label": str(row.label), "edges": sample.edges, "anchor": base_anchor,
                    "source_group": sample.source_id, "fold": fold, "edits": edits}
            validate_item(item, len(labels), cfg.visual_length, bank)
            pending[crop_key(row.path)] = item
            if number % args.commit_every == 0 or number == len(todo):
                write_rows(output / shard_name, pending)
                for key, value in pending.items():
                    index["rows"][key] = {"shard": shard_name, "bank": bank_name,
                                          "supported_queries": sum(bool(edit["weight"].sum()) for edit in value["edits"]),
                                          "normal_edits": sum(edit["normal_weight"] > 0 for edit in value["edits"])}
                save_atomic(index, index_path)
                pending.clear()
            if number % args.log_every == 0 or number == 1:
                print(json.dumps({"phase": "exact_crop_cache", "dataset": args.dataset, "fold": fold,
                                  "crop": number, "remaining_in_fold": len(todo) - number,
                                  "edits": len(edits), "seconds": time.monotonic() - started}), flush=True)
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if seen != set(available) or set(index["rows"]) != set(available):
        raise ValueError("OOF measurements must cover every actual training crop exactly once")
    index["complete"] = True
    index["audit"] = {"rows": len(index["rows"]), "supported_queries": sum(
        value["supported_queries"] for value in index["rows"].values()), "normal_edits": sum(
        value["normal_edits"] for value in index["rows"].values()),
        "elapsed_this_invocation_seconds": time.monotonic() - started}
    save_atomic(index, index_path)
    print(json.dumps({"phase": "exact_crop_cache_complete", **index["audit"]}), flush=True)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", choices=("ucf", "xd"), required=True)
    value.add_argument("--dsanet-root", required=True)
    value.add_argument("--manifest", required=True)
    value.add_argument("--teacher-pattern", required=True)
    value.add_argument("--anchor-checkpoint", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--train-list")
    value.add_argument("--seed", type=int, default=234)
    value.add_argument("--device", default="auto")
    value.add_argument("--reference-videos", type=int, default=64)
    value.add_argument("--calibration-videos", type=int, default=32)
    value.add_argument("--retrievals", type=int, default=3)
    value.add_argument("--min-donors", type=int, default=2)
    value.add_argument("--normal-controls", type=int, default=8)
    value.add_argument("--min-controls", type=int, default=6)
    value.add_argument("--positions-per-video", type=int, default=16)
    value.add_argument("--context-steps", type=int, default=4)
    value.add_argument("--max-context-distance", type=float, default=.8)
    value.add_argument("--duration-tolerance", type=float, default=.35)
    value.add_argument("--interval-fractions", type=float, nargs="+", default=(.0625, .125, .25))
    value.add_argument("--max-groups", type=int, default=4)
    value.add_argument("--queries-per-video", type=int, default=2)
    value.add_argument("--lower-quantile", type=float, default=.1)
    value.add_argument("--upper-quantile", type=float, default=.9)
    value.add_argument("--topk-divisor", type=int, default=16)
    value.add_argument("--infer-batch-size", type=int, default=32)
    value.add_argument("--null-cache-entries", type=int, default=2048)
    value.add_argument("--commit-every", type=int, default=128)
    value.add_argument("--log-every", type=int, default=25)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    if not (0 <= args.lower_quantile < args.upper_quantile <= 1 and
            1 <= args.min_donors <= args.retrievals and 1 <= args.min_controls <= args.normal_controls):
        raise ValueError("Invalid measurement settings")
    generate(args)
