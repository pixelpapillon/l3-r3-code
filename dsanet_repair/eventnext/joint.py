"""Independent, immutable R6 supplement with REAL A, B and union forwards.

The frozen anchor is a model-response reference, NOT ground-truth causality or
an independent OOF teacher. Normal controls estimate feature-edit nuisance.
Old cache files/signatures are never changed. No builder runs during training.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3

import torch

from dsanet_repair.data import source_group, UCF_LABEL_MAP, XD_LABEL_MAP
from dsanet_repair.eventstudy.data import EventDataset, selected_group
from dsanet_repair.revision.cache import save_atomic, write_rows, fingerprint, measurement_signature, upstream_signature
from dsanet_repair.revision.generate import infer
from dsanet_repair.revision.response import masked_topk
from dsanet_repair.train_v2 import file_hash
from scheme1_repair.reference import resize_segment


FORMAT = "dsanet-real-joint-response-v2"
INFERENCE_PROTOCOL = "singleton-all-X-A-B-AB-and-controls-v1"
ANCHOR_ATOL, ANCHOR_RTOL = 2e-4, 1e-5


def measure_sequences(teacher, sequences, prompt, visual_length, device):
    """Match cached anchor's singleton numerical path for EVERY sequence.

    DSANet thresholds its cosine graph at 0.7. Small batch-dependent CUDA
    differences can flip graph edges, not merely the final rounding bits.
    Never substitute a cached X into batched A/B/AB, or relax the identity gate.
    """
    return torch.stack(infer(teacher, sequences, prompt, visual_length, device, batch_size=1))


def anchor_diagnostic(raw, cached, key, row_index, length):
    difference = (raw - cached).abs()
    bad = ~torch.isclose(raw, cached, atol=ANCHOR_ATOL, rtol=ANCHOR_RTOL)
    return dict(key=key, row_index=row_index, length=length,
                inference_protocol=INFERENCE_PROTOCOL, atol=ANCHOR_ATOL, rtol=ANCHOR_RTOL,
                max_abs_error=float(difference.max()), mismatched_elements=int(bad.sum()),
                binary_max_abs_error=float(difference[:, 0].max()),
                semantic_max_abs_error=float(difference[:, 1:].max()))


def producer_signature():
    return {"eventnext/joint.py": file_hash(__file__), "measurement": measurement_signature()}


def joint_response(raw, lengths, divisor=16):
    """[B,4,T,C] in X,A,B,AB order. All four use the SAME global functional."""
    if raw.ndim != 4 or raw.shape[1] != 4 or lengths.shape != raw.shape[:1]:
        raise ValueError("Joint response needs original/A/B/AB and per-video lengths")
    batch, _, steps, channels = raw.shape
    mask = torch.arange(steps, device=raw.device)[None] < lengths[:, None]
    scores = masked_topk(raw.reshape(batch * 4, steps, channels),
                        mask[:, None].expand(-1, 4, -1).reshape(batch * 4, steps), divisor)
    scores = scores.reshape(batch, 4, channels)
    return scores[:, 1] + scores[:, 2] - scores[:, 3] - scores[:, 0]


def feasible_joint(base, lower, upper, weight, null, radii):
    corrected = base.detach() - null
    limit = 4 * radii.to(base)
    lo, hi = torch.maximum(lower, corrected - limit), torch.minimum(upper, corrected + limit)
    valid = lo <= hi
    return torch.where(valid, lo, 0.), torch.where(valid, hi, 0.), weight * valid


def control_envelope(responses, noise_floor):
    """Midpoint median avoids the zero-MAD artifact of two lower medians.

    This is a nuisance heuristic, not a finite-sample coverage guarantee.
    """
    if responses.ndim != 2 or len(responses) < 2 or noise_floor <= 0:
        raise ValueError("Need at least two finite normal controls and a positive floor")
    if not torch.isfinite(responses).all():
        raise ValueError("Non-finite normal control response")
    null = responses.quantile(.5, dim=0)
    radius = 1.4826 * (responses - null).abs().quantile(.5, dim=0) + noise_floor
    return null, radius


def apply_pair(features, intervals, donor):
    """One donor, two disjoint masks; AB is an actual composed INPUT."""
    intervals = torch.as_tensor(intervals, dtype=torch.long)
    if intervals.shape != (2, 2):
        raise ValueError("Expected two edit intervals")
    a, b = intervals.tolist()
    if not 0 <= a[0] < a[1] <= b[0] < b[1] <= len(features):
        raise ValueError("Joint regions must be valid, ordered and disjoint")
    edited = features[None].repeat(3, 1, 1)
    for index, (start, end) in enumerate((a, b)):
        replacement = resize_segment(donor, end - start).to(features)
        edited[index, start:end] = replacement
        edited[2, start:end] = replacement
    return edited


def checked_index(directory, data):
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    meta = index.get("metadata", {})
    if (index.get("format") != FORMAT or not index.get("complete") or
            meta.get("inference_protocol") != INFERENCE_PROTOCOL or
            meta.get("infer_batch_size") != 1 or
            meta.get("base_index_sha256") != file_hash(data.root / "index.json") or
            meta.get("producer") != producer_signature() or
            meta.get("anchor_sha256") != data.metadata["anchor_sha256"] or
            meta.get("class_order") != list(data.label_map) or
            meta.get("visual_length") != data.visual_length or
            meta.get("topk_divisor") != data.metadata["generation_config"]["topk_divisor"]):
        raise ValueError("Joint supplement/base-cache/producer identity mismatch")
    if set(data.frame.key) - set(index.get("rows", {})):
        raise ValueError("Incomplete joint supplement crop coverage")
    if index.get("shard_sha256") != file_hash(directory / "joint.sqlite"):
        raise ValueError("Joint measurement shard hash mismatch")
    if index.get("identity") != hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest():
        raise ValueError("Joint supplement identity is invalid")
    return index


class JointDataset(EventDataset):
    def __init__(self, *args, joint_cache, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_root = Path(joint_cache)
        self.joint_index = checked_index(self.joint_root, self)
        self._joint_connection, self._joint_pid = None, None

    def __getstate__(self):
        state = super().__getstate__()
        state["_joint_connection"], state["_joint_pid"] = None, None
        return state

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if self._joint_connection is None or self._joint_pid != os.getpid():
            self._joint_connection = sqlite3.connect((self.joint_root / "joint.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
            self._joint_pid = os.getpid()
        key = result["identity"]
        blob = self._joint_connection.execute("SELECT value FROM measurements WHERE key=?", (key,)).fetchone()
        if blob is None:
            raise ValueError("Missing joint measurement row")
        item = torch.load(io.BytesIO(blob[0]), map_location="cpu", weights_only=False)
        length = int(result["length"])
        if (item["fingerprint"] != fingerprint(result["features"], length) or
                item["length"] != length):
            raise ValueError("Joint measurement belongs to a different input")
        edited = result["features"][None].repeat(3, 1, 1)
        if item["active"]:
            _, bank = self._read(key)
            donor = bank[item["donor"]]["features"]
            edited[:, :length] = apply_pair(result["features"][:length], item["intervals"], donor)
            actual = [fingerprint(value, length) for value in edited]
            if actual != item["edited_fingerprints"]:
                raise ValueError("Joint inputs differ from the actual measured A/B/AB")
        for name in ("lower", "upper", "null", "weight", "measured"):
            if item[name].shape != (self.channels,) or not torch.isfinite(item[name]).all():
                raise ValueError("Invalid joint response shape/value")
        if (bool((item["lower"] > item["upper"]).any()) or
                bool(((item["weight"] < 0) | (item["weight"] > 1)).any())):
            raise ValueError("Invalid joint envelope/reliability")
        result.update(joint_edited=edited, joint_intervals=item["intervals"],
                      joint_holdout=torch.tensor(selected_group(key, self.holdout_fraction, self.holdout_salt + ":joint")))
        for name in ("lower", "upper", "null", "weight", "measured"):
            result["joint_" + name] = item[name]
        return result


@torch.no_grad()
def build(data, teacher, prompt, output, device="cpu", min_controls=2, noise_floor=.05, resume=False):
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError("Joint builder requires a NEW directory; never overwrites caches")
    if min_controls < 2 or noise_floor <= 0:
        raise ValueError("At least two independent normal controls and positive noise floor required")
    meta = dict(base_index_sha256=file_hash(data.root / "index.json"),
                base_identity=data.metadata["identity"], anchor_sha256=data.metadata["anchor_sha256"],
                class_order=list(data.label_map), visual_length=data.visual_length,
                topk_divisor=data.metadata["generation_config"]["topk_divisor"],
                producer=producer_signature(), min_controls=min_controls, noise_floor=noise_floor,
                inference_protocol=INFERENCE_PROTOCOL, infer_batch_size=1,
                reference="frozen-anchor model response, NOT independent OOF/ground-truth causal evidence",
                pair_rule="disjoint quarter/three-quarter regions; width=floor(length/8), minimum 1",
                null_rule="median normal-control joint response; radius=1.4826*MAD+floor")
    index = {"format": FORMAT, "complete": False, "metadata": meta, "rows": {},
             "identity": hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()}
    if resume:
        previous = json.loads((output / "index.json").read_text())
        if (previous.get("format") != FORMAT or previous.get("complete") or
                previous.get("metadata") != meta or previous.get("identity") != index["identity"]):
            raise ValueError("Only an unchanged, incomplete joint build can resume")
        index = previous
        if index["rows"] and not (output / "joint.sqlite").is_file():
            raise ValueError("Partial joint index has no committed shard")
    else:
        output.mkdir(parents=True)
        save_atomic(index, output / "index.json")
    teacher.eval()
    buffer = {}
    for i in range(len(data)):
        if data.frame.key.iloc[i] in index["rows"]:
            continue
        sample = data[i]
        key, length = sample["identity"], int(sample["length"])
        features = sample["features"][:length]
        channels = data.channels
        item = {"length": length, "fingerprint": fingerprint(features, length), "active": length >= 2,
                "intervals": torch.zeros(2, 2, dtype=torch.long),
                **{name: torch.zeros(channels) for name in ("lower", "upper", "null", "weight", "measured")}}
        if length >= 2:
            _, bank = data._read(key)
            # Deterministic independent source selection. Never use the query
            # source as donor/control, or one donor source as its own control.
            unique = {}
            for name in sorted(bank):
                group = source_group(name)
                if group != source_group(key):
                    unique.setdefault(group, name)
            names = list(unique.values())
            if len(names) < min_controls + 1:
                raise ValueError("Insufficient independent normal donor/control sources for " + key)
            offset = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % len(names)
            names = names[offset:] + names[:offset]
            donor = bank[names[0]]["features"]
            size = max(1, length // 8)
            a = max(0, length // 4 - size // 2)
            b = min(length - size, max(a + size, 3 * length // 4 - size // 2))
            intervals = torch.tensor([[a, a + size], [b, b + size]])
            edits = apply_pair(features, intervals, donor)
            controls = [resize_segment(bank[name]["features"], length) for name in names[1:min_controls + 1]]
            sequences = [features, *edits]
            for normal in controls:
                sequences.extend([normal, *apply_pair(normal, intervals, donor)])
            raw = measure_sequences(teacher, sequences, prompt, data.visual_length, device)
            if not torch.allclose(raw[0], sample["anchor"][:length], atol=ANCHOR_ATOL, rtol=ANCHOR_RTOL):
                diagnostic = anchor_diagnostic(raw[0], sample["anchor"][:length], key, i, length)
                save_atomic(dict(identity=index["identity"], **diagnostic), output / "failure.json")
                raise ValueError("Joint builder anchor does not match the original cache: " +
                                 json.dumps(diagnostic, sort_keys=True))
            measured = joint_response(raw.reshape(-1, 4, length, channels),
                                      torch.full((1 + len(controls),), length), meta["topk_divisor"])
            null, radius = control_envelope(measured[1:], noise_floor)
            corrected = measured[0] - null
            present = torch.cat([torch.ones(1), sample["labels"][1:]])
            item.update(intervals=intervals, donor=names[0], controls=names[1:min_controls + 1],
                        edited_fingerprints=[fingerprint(value, length) for value in edits],
                        measured=measured[0], null=null, lower=corrected - radius, upper=corrected + radius,
                        weight=present / (1 + radius))
        buffer[key] = item
        index["rows"][key] = {"active": item["active"], "fingerprint": item["fingerprint"]}
        if len(buffer) >= 64:
            write_rows(output / "joint.sqlite", buffer)
            buffer.clear()
            save_atomic(index, output / "index.json")
            print(json.dumps({"joint_rows_committed": len(index["rows"]), "total": len(data)}), flush=True)
    if buffer:
        write_rows(output / "joint.sqlite", buffer)
    index.update(complete=True, shard_sha256=file_hash(output / "joint.sqlite"))
    save_atomic(index, output / "index.json")
    return index


def main():
    from dsanet_repair.train_full import official_runtime, make_backbone, load_initial_state, setup_seed
    from dsanet_repair.revision.train import resolve_cfg
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dsanet-root", "response-cache", "anchor-checkpoint", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dataset", choices=("ucf", "xd"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="Continue only an unchanged INCOMPLETE supplement")
    args = parser.parse_args()
    setup_seed(234)
    root, cfg, _, cls, _, utils, _ = official_runtime(args.dsanet_root, args.dataset)
    resolve_cfg(cfg, root)
    labels = UCF_LABEL_MAP if args.dataset == "ucf" else XD_LABEL_MAP
    data = EventDataset(cfg.train_list, args.response_cache, labels, cfg.visual_length)
    meta = data.metadata
    if (meta["anchor_sha256"] != file_hash(args.anchor_checkpoint) or meta["dataset"] != args.dataset or
            meta["seed"] != 234 or meta["numerical_mode"] != "stable" or
            meta["measurement_source"] != measurement_signature() or meta["upstream_source"] != upstream_signature(root)):
        raise ValueError("Joint builder requires the exact existing cache, producer and anchor")
    model = make_backbone(cls, cfg, torch.device(args.device), "stable").to(args.device)
    model.load_state_dict(load_initial_state(args.anchor_checkpoint), strict=True)
    result = build(data, model, utils.get_prompt_text(labels), args.output_dir, args.device, resume=args.resume)
    print(json.dumps({"identity": result["identity"], "rows": len(result["rows"]), "complete": result["complete"]}))


if __name__ == "__main__":
    main()
