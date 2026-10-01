"""Paired original/edited DSANet dataset backed by a v2 response cache."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data import label_vector, process_feature, row_id, source_group
from .v2_cache import load_v2_cache


class PairedEditDataset(Dataset):
    def __init__(self, csv_path, cache_path, label_map, visual_length=256, subset=None,
                 max_rows=None, cache_only=True):
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
        self.cache, self.cache_metadata = load_v2_cache(cache_path)
        # Query manifests and training CSVs can refer to the same feature with
        # different absolute roots (for example /data/VAD vs /root/.../VAD).
        # The exact row id remains the primary key; source_group is a
        # deterministic, root-independent fallback because the v2 pilot keeps
        # at most one query row per original video.
        self.cache_groups = {}
        for item in self.cache.values():
            group = item.diagnostics.get("source_group")
            if group is not None:
                if group in self.cache_groups:
                    raise ValueError(f"Duplicate v2 cache source group: {group}")
                self.cache_groups[group] = item
        if cache_only:
            identities = self.df["path"].map(row_id)
            groups = self.df["path"].map(source_group)
            represented = identities.isin(self.cache) | groups.isin(self.cache_groups)
            self.df = self.df[represented].reset_index(drop=True)
        if not len(self.df):
            raise ValueError(f"No {subset or 'training'} rows are represented in the v2 cache")
        if max_rows is not None:
            if int(max_rows) < 1:
                raise ValueError("max_rows must be positive")
            self.df = self.df.iloc[:int(max_rows)].reset_index(drop=True)
        self.label_map = dict(label_map)
        self.visual_length = int(visual_length)
        self.channels = len(label_map)
        self.queries = int(self.cache_metadata.get("queries_per_video", 0))
        if self.queries < 1:
            raise ValueError("V2 cache must declare a positive queries_per_video")
        if int(self.cache_metadata.get("visual_length", -1)) != self.visual_length:
            raise ValueError("V2 cache visual length mismatch")
        if int(self.cache_metadata.get("channels", -1)) != self.channels:
            raise ValueError("V2 cache channel order mismatch")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        record = self.df.iloc[index]
        path = str(record["path"])
        features, valid_length = process_feature(np.load(path), self.visual_length)
        identity = row_id(path)
        item = self.cache.get(identity)
        if item is None:
            item = self.cache_groups.get(source_group(path))
        if item is None:
            raise KeyError(f"Missing v2 cache item: {identity}")
        if item.valid_length != valid_length or item.feature_dim != features.shape[1]:
            raise ValueError(f"V2 cache/source feature mismatch for {identity}")
        if len(item.edits) > self.queries:
            raise ValueError(f"Too many cached edits for {identity}")

        original = torch.from_numpy(features).float()
        edited = original.unsqueeze(0).repeat(self.queries, 1, 1)
        lower = torch.zeros(self.queries, self.channels)
        upper = torch.zeros_like(lower)
        reliability = torch.zeros_like(lower)
        null_response = torch.zeros_like(lower)
        for query, edit in enumerate(item.edits):
            edited[query, edit.start:edit.end] = edit.replacement.to(edited)
            lower[query] = edit.lower
            upper[query] = edit.upper
            reliability[query] = edit.reliability
            null_response[query] = edit.null_response

        anchor = torch.zeros(self.visual_length, self.channels)
        anchor[:valid_length] = item.anchor
        return {
            "features": original,
            "edited": edited,
            "length": torch.tensor(valid_length, dtype=torch.long),
            "identity": identity,
            "lower": lower,
            "upper": upper,
            "reliability": reliability,
            "null_response": null_response,
            "anchor": anchor,
            "labels": label_vector(record["label"], self.label_map),
        }
