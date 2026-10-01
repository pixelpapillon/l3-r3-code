"""Immutable milestone snapshots; evaluation cannot perturb training RNG."""

import json
import time
from pathlib import Path

import torch

from dsanet_repair.revision.cache import save_atomic
from dsanet_repair.revision import train as reference
from dsanet_repair.numerics import clear_text_cache


def milestone(output, epoch, args, identity, model, loaders, evaluator):
    if epoch not in args.snapshot_epochs:
        return 0.
    started = time.monotonic()
    directory = Path(output) / "epochs" / f"epoch_{epoch:03d}"
    target = directory / "checkpoint.pt"
    if target.exists():
        payload = torch.load(target, map_location="cpu", weights_only=False)
        if payload.get("identity") != identity or payload.get("epoch") != epoch:
            raise ValueError("Refusing to replace an incompatible milestone")
    else:
        payload = torch.load(Path(output) / "last.pt", map_location="cpu", weights_only=False)
        if payload.get("identity") != identity or payload.get("epoch") != epoch:
            raise ValueError("Milestones can only snapshot the current committed epoch")
        directory.mkdir(parents=True, exist_ok=True)
        save_atomic(payload, target)
    completed = directory / "evaluation.json"
    if completed.exists():
        record = json.loads(completed.read_text())
        if record.get("identity") != identity or record.get("epoch") != epoch:
            raise ValueError("Incompatible milestone evaluation record")
    elif args.evaluate_snapshots and not args.skip_evaluation:
        state, mode = reference.rng_state(loaders), model.training
        try:
            destination = (Path(output) / "final" if epoch == getattr(args, "epochs", None)
                           else directory / "evaluation")
            evaluation_started = time.monotonic()
            metrics = evaluator(destination)
            save_atomic({"identity": identity, "epoch": epoch,
                         "selection_policy": "diagnostic_only_not_test_best",
                         "evaluation_directory": str(destination.relative_to(output)),
                         "evaluation_seconds": time.monotonic() - evaluation_started,
                         "metrics": metrics}, completed)
        finally:
            model.train(mode)
            clear_text_cache(model)
            reference.restore_rng(state, loaders)
    return time.monotonic() - started
