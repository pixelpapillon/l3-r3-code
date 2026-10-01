"""DSANet CSV loader aligned with an offline intervention-response cache."""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .supervision import load_response_cache


UCF_LABEL_MAP = {
    "Normal": "normal", "Abuse": "abuse", "Arrest": "arrest", "Arson": "arson",
    "Assault": "assault", "Burglary": "burglary", "Explosion": "explosion",
    "Fighting": "fighting", "RoadAccidents": "roadAccidents", "Robbery": "robbery",
    "Shooting": "shooting", "Shoplifting": "shoplifting", "Stealing": "stealing",
    "Vandalism": "vandalism",
}
XD_LABEL_MAP = {
    "A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot",
    "B5": "abuse", "B6": "car accident", "G": "explosion",
}


def row_id(path):
    """Stable cache key for one DSANet training row."""
    return str(Path(path).resolve())


def source_group(path):
    """Original-video group used for leakage-safe OOF assignment."""
    return re.sub(r"__\d+$", "", Path(path).stem)


def label_vector(label, label_map):
    values = list(label_map)
    result = torch.zeros(len(values), dtype=torch.float32)
    for token in str(label).split("-"):
        if token in label_map:
            result[values.index(token)] = 1
    if not result.any():
        raise ValueError(f"Unknown or empty label: {label}")
    return result


def process_feature(feature, length=256):
    """Mirror DSANet's deterministic training-time process_feat."""
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim != 2:
        raise ValueError("CLIP feature must be [T,D]")
    original = len(feature)
    if original > length:
        boundaries = np.linspace(0, original, length + 1, dtype=np.int32)
        output = np.zeros((length, feature.shape[1]), dtype=np.float32)
        for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            output[i] = feature[a:b].mean(0) if a != b else feature[a]
        return output, length
    output = np.zeros((length, feature.shape[1]), dtype=np.float32)
    output[:original] = feature
    return output, original


class CachedResponseDataset(Dataset):
    def __init__(self, csv_path, cache_path, label_map, visual_length=256,
                 subset=None, require_positive_cache=True):
        self.df = pd.read_csv(csv_path)
        if not {"path", "label"}.issubset(self.df.columns):
            raise ValueError("DSANet CSV requires path,label columns")
        if subset not in (None, "normal", "anomaly"):
            raise ValueError("subset must be normal, anomaly, or None")
        normal_name = next(iter(label_map))
        is_normal = self.df["label"].map(lambda x: str(x).split("-")[0] == normal_name)
        if subset == "normal":
            self.df = self.df[is_normal]
        elif subset == "anomaly":
            self.df = self.df[~is_normal]
        self.df = self.df.reset_index(drop=True)
        self.cache, self.cache_metadata = load_response_cache(cache_path)
        self.label_map = dict(label_map)
        self.visual_length = int(visual_length)
        self.targets = len(label_map)
        self.require_positive_cache = bool(require_positive_cache)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        record = self.df.iloc[index]
        path = str(record["path"])
        features, length = process_feature(np.load(path), self.visual_length)
        identity = row_id(path)
        item = self.cache.get(identity)
        labels = label_vector(record["label"], self.label_map)
        normal = bool(labels[0] == 1)
        if item is None:
            if self.require_positive_cache and not normal:
                raise KeyError(f"Missing intervention response for anomalous row: {identity}")
            target = torch.zeros(self.visual_length, self.targets)
            weight = torch.zeros_like(target)
        else:
            target, weight = item.target, item.weight
            if target.shape != (self.visual_length, self.targets):
                raise ValueError(f"Cache shape mismatch for {identity}: {tuple(target.shape)}")
        return (torch.from_numpy(features), str(record["label"]), torch.tensor(length), identity,
                target.float(), weight.float(), labels)
