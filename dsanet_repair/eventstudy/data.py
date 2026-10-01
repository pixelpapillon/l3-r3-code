"""Read-only exact-cache wrapper; measurement producers and signatures untouched."""

import hashlib
import json
import torch
from torch.utils.data import DataLoader

from dsanet_repair.data import source_group
from dsanet_repair.revision.cache import ExactCropDataset


def selected_group(identity, fraction, salt):
    digest = hashlib.sha256((salt + ":" + source_group(identity)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64 < fraction


class EventDataset(ExactCropDataset):
    def __init__(self, *args, holdout_fraction=.1, holdout_salt="eventstudy-query-v1", **kwargs):
        super().__init__(*args, **kwargs)
        self.holdout_fraction, self.holdout_salt = holdout_fraction, holdout_salt
        self._extra = None

    def _read(self, key):
        item, bank = super()._read(key)
        # Called synchronously once by parent's __getitem__ in each worker.
        self._extra = item
        return item, bank

    def __getitem__(self, index):
        self._extra = None
        result = super().__getitem__(index)
        item, self._extra = self._extra, None
        if item is None:
            raise RuntimeError("Missing exact-cache metadata")
        edges = item["edges"].float()
        padded = edges.new_full((self.visual_length + 1,), float(edges[-1]))
        padded[:len(edges)] = edges
        eligible = [i for i, edit in enumerate(item["edits"]) if bool(edit["weight"].sum() > 0)]
        holdout = torch.zeros(self.queries, dtype=torch.bool)
        if (result["labels"][0] == 0 and len(eligible) >= 2 and
                selected_group(result["identity"], self.holdout_fraction, self.holdout_salt)):
            # Same group selection; choose the last supported query per crop.
            # Its index/time interval need not coincide across different crops.
            holdout[eligible[-1]] = True
        result.update(edges=padded, response_holdout=holdout)
        return result

    def holdout_manifest(self):
        rows = []
        for key in self.frame.key:
            item, _ = super()._read(key)
            eligible = [i for i, edit in enumerate(item["edits"]) if bool(edit["weight"].sum() > 0)]
            abnormal = str(item["label"]).split("-")[0] != next(iter(self.label_map))
            selected = abnormal and len(eligible) >= 2 and selected_group(key, self.holdout_fraction, self.holdout_salt)
            rows.append({"crop": key, "group": source_group(key), "label": item["label"],
                         "eligible": len(eligible), "held_query": eligible[-1] if selected else None})
        rows.sort(key=lambda row: row["crop"])
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        return {"sha256": digest, "rows": rows,
                "held_queries": sum(r["held_query"] is not None for r in rows),
                "held_source_groups": len({r["group"] for r in rows if r["held_query"] is not None}),
                "scope": "unseen edit responses in training videos; not held-out-video validation"}


def make_loaders(cfg, args, label_map, device):
    loaders = []
    for offset, subset in enumerate(("normal", "anomaly") if args.dataset == "ucf" else (None,)):
        data = EventDataset(cfg.train_list, args.response_cache, label_map, cfg.visual_length, subset=subset,
                            holdout_fraction=args.response_holdout_fraction, holdout_salt=args.response_holdout_salt)
        loaders.append(DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=device.type == "cuda", drop_last=args.dataset == "ucf",
                                  generator=torch.Generator().manual_seed(args.seed + offset)))
    if min(map(len, loaders)) < 1:
        raise ValueError("Training subset is smaller than batch size")
    return loaders[0], loaders[1] if len(loaders) > 1 else None, min(map(len, loaders))
