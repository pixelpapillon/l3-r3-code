"""Sharded exact-crop measurements. Historical v2 caches are not accepted."""

import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dsanet_repair.data import label_vector, process_feature
from scheme1_repair.reference import resize_segment


FORMAT = "dsanet-crop-exact-measurements-v3"
PREPROCESS = "dsanet-deterministic-mean-grid-fp32-v1"
SCOPES = ("global", "local", "outside")


def upstream_signature(root):
    root = Path(root)
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "src").rglob("*.py"))}


def measurement_signature():
    root = Path(__file__).resolve().parents[2]
    names = ["dsanet_repair/revision/" + name + ".py"
             for name in ("cache", "generate", "reference", "response")]
    names += ["dsanet_repair/" + name + ".py" for name in
              ("data", "numerics", "adapter", "generate_v2_cache", "train_full")]
    names += ["scheme1_repair/reference.py", "vadcore/types.py"]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def checked_labels(label, label_map):
    tokens = str(label).split("-")
    known = [token for token in tokens if token in label_map]
    unknown = [token for token in tokens if token not in label_map and token != "0"]
    # XD-Violence's official weak labels use ``0`` as a positional
    # placeholder, e.g. ``B1-B2-0`` and ``G-0-0``.  It is metadata, not a
    # category.  Keep rejecting every other unknown token and reject mixed
    # Normal/abnormal labels as before.
    if not known or unknown or (next(iter(label_map)) in known and len(set(known)) != 1):
        raise ValueError(f"Invalid weak category label: {label}")
    return label_vector(label, label_map)


def crop_key(path):
    # Preserve __0/__1/...; only machine-specific parent directories disappear.
    return Path(path).name


def fingerprint(features, length):
    array = np.ascontiguousarray(np.asarray(features, dtype=np.float32)[:length])
    digest = hashlib.sha256(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def bank_fingerprint(bank):
    digest = hashlib.sha256()
    for name, value in sorted(bank.items()):
        digest.update(name.encode())
        digest.update(fingerprint(value["features"], len(value["features"])).encode())
        digest.update(np.asarray(value["edges"], dtype=np.float32).tobytes())
    return digest.hexdigest()


def load_feature(path, visual_length):
    raw = np.asarray(np.load(path), dtype=np.float32)
    if raw.ndim != 2 or len(raw) < 1 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid CLIP feature: {path}")
    features, length = process_feature(raw, visual_length)
    edges = (np.linspace(0, len(raw), visual_length + 1, dtype=np.int32)
             if len(raw) > visual_length else np.arange(len(raw) + 1))
    return features, length, torch.as_tensor(edges, dtype=torch.float32)


def save_atomic(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".json":
        temporary.write_text(json.dumps(value, indent=2))
    else:
        torch.save(value, temporary)
    temporary.replace(path)


def write_rows(path, rows):
    """One indexed database per OOF fold; random reads don't load whole shards."""
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS measurements (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
        values = []
        for key, item in rows.items():
            buffer = io.BytesIO()
            torch.save(item, buffer)
            values.append((key, buffer.getvalue()))
        with connection:
            connection.executemany("INSERT OR REPLACE INTO measurements VALUES (?, ?)", values)
    finally:
        connection.close()


def validate_item(item, channels, visual_length, bank):
    length = int(item["length"])
    if not 0 < length <= visual_length or item["anchor"].shape != (length, channels):
        raise ValueError("Bad exact-crop anchor shape")
    if not torch.isfinite(item["anchor"]).all() or not item.get("fingerprint"):
        raise ValueError("Invalid anchor or missing input fingerprint")
    if len(item["edges"]) != length + 1 or not bool((item["edges"].diff() > 0).all()):
        raise ValueError("Invalid native-clip grid mapping")
    for edit in item["edits"]:
        if not 0 <= edit["start"] < edit["end"] <= length:
            raise ValueError("Invalid query interval")
        donor = bank[edit["donor"]]
        if not 0 <= edit["donor_start"] < edit["donor_end"] <= len(donor["features"]):
            raise ValueError("Invalid donor interval")
        for name in ("lower", "upper", "weight", "null"):
            value = edit[name]
            if value.shape != (len(SCOPES), channels) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid {name} measurement")
        if bool((edit["lower"] > edit["upper"]).any()):
            raise ValueError("Reversed response envelope")
        if bool(((edit["weight"] < 0) | (edit["weight"] > 1)).any()):
            raise ValueError("Response reliability outside [0,1]")
        if not 0 <= float(edit.get("normal_weight", 0)) <= 1:
            raise ValueError("Invalid normal-edit label weight")


class ExactCropDataset(Dataset):
    def __init__(self, csv_path, cache_dir, label_map, visual_length=256, subset=None):
        self.root = Path(cache_dir)
        self.index = json.loads((self.root / "index.json").read_text())
        if self.index.get("format") != FORMAT or not self.index.get("complete"):
            raise ValueError("A complete crop-exact v3 cache is required")
        self.metadata = self.index["metadata"]
        if self.metadata["preprocess"] != PREPROCESS:
            raise ValueError("Cache preprocessing mismatch")
        if self.metadata["class_order"] != list(label_map):
            raise ValueError("Cache category order mismatch")
        if int(self.metadata["visual_length"]) != int(visual_length):
            raise ValueError("Cache grid mismatch")
        frame = pd.read_csv(csv_path)
        frame["key"] = frame.path.map(crop_key)
        if frame.key.duplicated().any():
            raise ValueError("Non-unique exact crop names; use an explicit dataset-relative identity")
        normal = frame.label.map(lambda value: str(value).split("-")[0] == next(iter(label_map)))
        if subset == "normal":
            frame = frame[normal]
        elif subset == "anomaly":
            frame = frame[~normal]
        elif subset is not None:
            raise ValueError("Unknown subset")
        missing = set(frame.key) - self.index["rows"].keys()
        if missing:
            raise ValueError(f"Missing {len(missing)} exact crop measurements; no group fallback: {sorted(missing)[:3]}")
        self.frame = frame.reset_index(drop=True)
        if not len(self.frame):
            raise ValueError("Empty training subset")
        self.label_map = label_map
        self.visual_length = int(visual_length)
        self.queries = int(self.metadata["queries_per_video"])
        self.channels = len(label_map)
        self._databases = {}
        self._banks = {}
        self._owner_pid = os.getpid()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_databases"], state["_banks"] = {}, {}
        state["_owner_pid"] = None
        return state

    def __len__(self):
        return len(self.frame)

    def _read(self, key):
        if self._owner_pid != os.getpid():
            self._databases = {}
            self._banks = {}
            self._owner_pid = os.getpid()
        location = self.index["rows"][key]
        shard_name = location["shard"]
        if shard_name not in self._databases:
            uri = (self.root / shard_name).resolve().as_uri() + "?mode=ro"
            self._databases[shard_name] = sqlite3.connect(uri, uri=True)
        result = self._databases[shard_name].execute(
            "SELECT value FROM measurements WHERE key = ?", (key,)
        ).fetchone()
        if result is None:
            raise ValueError(f"Incomplete measurement shard: {key}")
        item = torch.load(io.BytesIO(result[0]), map_location="cpu", weights_only=False)
        bank_name = location["bank"]
        if bank_name not in self._banks:
            bank = torch.load(self.root / bank_name, map_location="cpu", weights_only=False)
            if self.index.get("banks", {}).get(bank_name) != bank_fingerprint(bank):
                raise ValueError("Normal donor bank fingerprint mismatch")
            self._banks[bank_name] = bank
        return item, self._banks[bank_name]

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        item, bank = self._read(row.key)
        validate_item(item, self.channels, self.visual_length, bank)
        if len(item["edits"]) > self.queries:
            raise ValueError("More cached queries than declared by the measurement protocol")
        features, length, edges = load_feature(row.path, self.visual_length)
        if (fingerprint(features, length) != item["fingerprint"] or
                length != item["length"] or str(row.label) != item["label"] or
                not torch.equal(edges, item["edges"])):
            raise ValueError(f"Cached measurement does not belong to this actual input: {row.key}")
        original = torch.from_numpy(features)
        edited = original[None].repeat(self.queries, 1, 1)
        dimensions = (self.queries, len(SCOPES), self.channels)
        lower, upper, weight, null = [torch.zeros(dimensions) for _ in range(4)]
        intervals = torch.zeros(self.queries, 2, dtype=torch.long)
        normal_weight = torch.zeros(self.queries)
        labels = checked_labels(row.label, self.label_map)
        for query, edit in enumerate(item["edits"]):
            start, end = edit["start"], edit["end"]
            source = bank[edit["donor"]]["features"][edit["donor_start"]:edit["donor_end"]]
            edited[query, start:end] = resize_segment(source, end - start)
            intervals[query] = torch.tensor([start, end])
            lower[query], upper[query] = edit["lower"], edit["upper"]
            weight[query], null[query] = edit["weight"], edit["null"]
            normal_weight[query] = edit.get("normal_weight", 0.)
        if labels[0] != 1 and bool((normal_weight > 0).any()):
            raise ValueError("Known-normal augmentation cannot supervise an abnormal bag")
        anchor = torch.zeros(self.visual_length, self.channels)
        anchor[:length] = item["anchor"]
        return {"features": original, "edited": edited, "length": torch.tensor(length),
                "labels": labels, "anchor": anchor,
                "lower": lower, "upper": upper, "weight": weight, "null": null,
                "intervals": intervals, "normal_weight": normal_weight, "identity": row.key}
