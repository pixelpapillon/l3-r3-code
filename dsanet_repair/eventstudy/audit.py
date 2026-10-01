"""Final-checkpoint, training-video held-query diagnostics; no model selection."""

import hashlib
import json
import torch
from torch.utils.data import DataLoader, Subset

from dsanet_repair.revision import train as reference
from dsanet_repair.numerics import clear_text_cache
from .losses import batch_losses


def manifest(loaders):
    subsets = [loader.dataset.holdout_manifest() for loader in loaders if loader is not None]
    identity = hashlib.sha256(json.dumps([s["sha256"] for s in subsets]).encode()).hexdigest()
    return {"sha256": identity, "subsets": subsets,
            "held_queries": sum(s["held_queries"] for s in subsets),
            "note": "Query holdout is shared by all variants; not an unseen-video validation split."}


@torch.no_grad()
def evaluate_heldout(model, loaders, prompt, cfg, args, device, allocation):
    states, mode = reference.rng_state(loaders), model.training
    sums, examples = {}, 0
    try:
        model.eval()
        clear_text_cache(model)
        for loader, subset in zip([x for x in loaders if x is not None], allocation["subsets"]):
            selected = {row["crop"] for row in subset["rows"] if row["held_query"] is not None}
            indices = [i for i, key in enumerate(loader.dataset.frame.key) if key in selected]
            evaluation = DataLoader(Subset(loader.dataset, indices), batch_size=args.batch_size,
                                    num_workers=0, shuffle=False)
            for batch in evaluation:
                losses = batch_losses(model, batch, prompt, cfg, device, args, class_audit=True)
                examples += len(batch["length"])
                for key, value in losses.items():
                    if key.startswith("held_"):
                        sums[key] = sums.get(key, 0.) + float(value)
        means = {}
        for key, value in sums.items():
            if key.endswith("_mass"):
                prefix = key[:-5]
                means[prefix] = sums[prefix + "_violation_sum"] / value if value > 0 else None
        return {"status": "measured" if examples else "no_eligible_held_queries",
                "class_order": list(loaders[0].dataset.label_map),
                "examples": examples, "allocation_sha256": allocation["sha256"],
                "sufficient_statistics": sums, "weighted_mean_violation": means,
                "scope": "training-video query generalization; not test performance or causal identification"}
    finally:
        model.train(mode)
        clear_text_cache(model)
        reference.restore_rng(states, loaders)
