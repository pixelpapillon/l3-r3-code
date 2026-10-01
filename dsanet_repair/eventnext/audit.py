"""Read-only held-query audits, including the independent R6 query allocation."""

import hashlib
import json
import torch
from torch.utils.data import DataLoader, Subset
from dsanet_repair.eventstudy.audit import manifest as event_manifest
from dsanet_repair.eventstudy.data import selected_group
from dsanet_repair.revision import train as reference
from dsanet_repair.numerics import clear_text_cache
from .losses import batch_losses


def manifest(loaders):
    result = event_manifest(loaders)
    joint = []
    for loader in (x for x in loaders if x is not None):
        data = loader.dataset
        if hasattr(data, "joint_index"):
            joint.append(dict(identity=data.joint_index["identity"], rows=[
                dict(crop=key, held=selected_group(key, data.holdout_fraction, data.holdout_salt + ":joint"))
                for key in data.frame.key]))
    if joint:
        result["single_query_sha256"] = result["sha256"]
        result["joint_subsets"] = joint
        result["sha256"] = hashlib.sha256(json.dumps(
            dict(single=result["single_query_sha256"], joint=joint), sort_keys=True).encode()).hexdigest()
    return result


@torch.no_grad()
def evaluate_heldout(model, loaders, prompt, cfg, args, device, allocation):
    state, mode = reference.rng_state(loaders), model.training
    sums, examples = {}, 0
    try:
        model.eval()
        clear_text_cache(model)
        for loader, subset in zip([x for x in loaders if x is not None], allocation["subsets"]):
            selected = {r["crop"] for r in subset["rows"] if r["held_query"] is not None}
            if model.next_configuration.interaction:
                selected |= {key for key in loader.dataset.frame.key if selected_group(
                    key, args.response_holdout_fraction, args.response_holdout_salt + ":joint")}
            indices = [i for i, key in enumerate(loader.dataset.frame.key) if key in selected]
            for batch in DataLoader(Subset(loader.dataset, indices), batch_size=args.batch_size, shuffle=False):
                values = batch_losses(model, batch, prompt, cfg, device, args, class_audit=True)
                examples += len(batch["length"])
                for key, value in values.items():
                    if key.startswith("held_"):
                        sums[key] = sums.get(key, 0.) + float(value)
        means = {key[:-5]: sums[key[:-5] + "_violation_sum"] / value if value else None
                 for key, value in sums.items() if key.endswith("_mass")}
        return dict(status="measured" if examples else "no_eligible_held_queries", examples=examples,
                    allocation_sha256=allocation["sha256"], sufficient_statistics=sums,
                    weighted_mean_violation=means, class_order=list(loaders[0].dataset.label_map),
                    scope="training-video held responses; NOT unseen-video generalization or causal identification")
    finally:
        model.train(mode)
        clear_text_cache(model)
        reference.restore_rng(state, loaders)
