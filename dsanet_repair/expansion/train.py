"""Train one complete expansion configuration from an immutable DSANet anchor.

Uses existing exact-crop measurements; never builds caches, launches teachers,
selects test-best checkpoints, or schedules another configuration implicitly.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dsanet_repair.data import UCF_LABEL_MAP, XD_LABEL_MAP
from dsanet_repair.numerics import clear_text_cache
from dsanet_repair.train_full import official_runtime, make_backbone, load_initial_state, setup_seed
from dsanet_repair.train_v2 import file_hash, epoch_batches
from dsanet_repair.revision.cache import SCOPES, measurement_signature, upstream_signature, save_atomic
from dsanet_repair.revision.evaluation import evaluate
from dsanet_repair.revision import train as reference
from . import VARIANTS
from .model import StudyCorrection
from .losses import batch_losses
from .optimization import diagnostics, optimizer_step
from dsanet_repair.refinement import ALL_CONFIGS as REFINEMENTS, add_arguments, validate_options
from dsanet_repair.eventstudy import CONFIGS as EVENTS


FORMAT = "dsanet-expansion-six-delta-v1"
resolve_cfg = reference.resolve_cfg
make_loaders = reference.make_loaders


def normalize_options(args):
    if not hasattr(args, "snapshot_epochs"):
        args.snapshot_epochs = []
    if not hasattr(args, "evaluate_snapshots"):
        args.evaluate_snapshots = False
    if (len(set(args.snapshot_epochs)) != len(args.snapshot_epochs) or
            any(epoch < 1 or epoch > args.epochs for epoch in args.snapshot_epochs)):
        raise ValueError("Snapshot epochs must be unique and inside the training budget")
    args.snapshot_epochs = sorted(args.snapshot_epochs)
    if args.variant not in VARIANTS and args.variant not in REFINEMENTS and args.variant not in EVENTS:
        raise ValueError("Unknown expansion variant")
    if args.variant in EVENTS:
        from dsanet_repair.eventstudy import validate
        validate(args)
    if args.variant in REFINEMENTS:
        validate_options(args)
    if args.variant == "p2-center-no-response":
        args.response_weight = args.local_weight = args.outside_weight = 0.
    if args.variant in REFINEMENTS and REFINEMENTS[args.variant].no_response:
        args.response_weight = args.local_weight = args.outside_weight = 0.
    positive = (args.epochs, args.batch_size, args.lr, args.clip_grad, args.topk_divisor,
                args.log_every, args.diagnose_every, args.init_probe_batches, args.init_atol,
                args.hidden, args.binary_radius, args.semantic_radius, args.prototype_temperature,
                args.temporal_levels, args.temporal_kernel, args.transport_iterations,
                args.transport_entropy, args.transport_step, args.transport_geometry_temperature)
    nonnegative = (args.workers, args.seed, args.weight_decay, args.anchor_weight, args.anchor_radius,
                   args.response_weight, args.local_weight, args.outside_weight, args.normal_weight,
                   args.transport_weight, args.transport_mass, args.head_alignment_weight,
                   args.head_alignment_cap, args.auxiliary_gradient_ratio)
    if (not all(math.isfinite(float(x)) for x in positive + nonnegative) or min(positive) <= 0 or
            min(nonnegative) < 0 or not 0 <= args.transport_alpha < 1 or not 0 < args.transport_radius <= 1):
        raise ValueError("Invalid expansion options")
    if args.variant == "p4-dilated-context" and (args.hidden % args.temporal_levels or args.temporal_kernel % 2 == 0):
        raise ValueError("P4 needs hidden divisible by temporal levels and an odd kernel")
    return args


def model_for_variant(backbone, args):
    if args.variant in EVENTS:
        from dsanet_repair.eventstudy.model import EventCorrection
        model = EventCorrection(backbone, args)
    elif args.variant in REFINEMENTS:
        from dsanet_repair.refinement.model import RefinedCorrection
        model = RefinedCorrection(backbone, args)
    else:
        model = StudyCorrection(backbone, args.variant, args.hidden, args.binary_radius, args.semantic_radius,
                                args.prototype_temperature, args.temporal_levels, args.temporal_kernel)
    selected = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    group_ids = [id(p) for group in model.parameter_partitions().values() for p in group]
    if len(group_ids) != len(set(group_ids)) or set(group_ids) != {id(p) for _, p in selected}:
        raise ValueError("Trainable parameter partition is incomplete or overlapping")
    return model, selected


def source_signature(root):
    values = reference.source_signature(root)
    repo = Path(__file__).resolve().parents[2]
    for path in sorted(Path(__file__).parent.glob("*.py")):
        values[str(path.relative_to(repo))] = file_hash(path)
    manifest = Path(__file__).with_name("sources.json")
    values[str(manifest.relative_to(repo))] = file_hash(manifest)
    for path in sorted((repo / "dsanet_repair" / "refinement").glob("*")):
        if path.suffix in (".py", ".json"):
            values[str(path.relative_to(repo))] = file_hash(path)
    for path in sorted((repo / "dsanet_repair" / "frontier").glob("*")):
        if path.suffix in (".py", ".json"):
            values[str(path.relative_to(repo))] = file_hash(path)
    for path in sorted((repo / "dsanet_repair" / "eventstudy").glob("*")):
        if path.suffix in (".py", ".json"):
            values[str(path.relative_to(repo))] = file_hash(path)
    return values


def delta_state(model, names):
    state = model.state_dict()
    # Store all non-backbone state, not just gradients: future buffers must replay.
    return {k: v.detach().cpu().clone() for k, v in state.items() if not k.startswith("backbone.")}


def apply_delta(model, payload, names):
    expected = set(delta_state(model, names))
    if set(payload["delta"]) != expected or payload["trainable_names"] != list(names):
        raise ValueError("Delta does not exactly match this expansion architecture")
    if not torch.equal(payload["delta"]["radii"].cpu(), model.radii.cpu()):
        raise ValueError("Checkpoint correction budgets disagree")
    result = model.load_state_dict(payload["delta"], strict=False)
    if result.unexpected_keys or set(result.missing_keys) != set(model.state_dict()) - expected:
        raise ValueError("Incomplete expansion delta")
    clear_text_cache(model)


def save_checkpoint(path, model, names, optimizer, scheduler, epoch, step, args, metadata,
                    identity, loaders, probe, initial, source, training_seconds):
    save_atomic({"format": FORMAT, "dataset": args.dataset, "variant": args.variant,
                 "seed": args.seed, "epoch": epoch, "step": step, "identity": identity,
                 "anchor_sha256": metadata["anchor_sha256"], "cache_identity": metadata["identity"],
                 "config": vars(args), "trainable_names": names, "delta": delta_state(model, names),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "rng": reference.rng_state(loaders), "initialization": probe, "initial": initial,
                 "source": source, "training_seconds": training_seconds}, path)


def train(args):
    normalize_options(args)
    loss_function = batch_losses
    if args.variant in REFINEMENTS:
        from dsanet_repair.refinement.losses import batch_losses as loss_function
    loader_factory = make_loaders
    evaluation_function = evaluate
    event_study = args.variant in EVENTS
    if event_study:
        from dsanet_repair.eventstudy.losses import batch_losses as loss_function
        from dsanet_repair.eventstudy.data import make_loaders as loader_factory
        from dsanet_repair.eventstudy.evaluation import evaluate as evaluation_function
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Use a new output directory or explicitly --resume")
    if args.resume and not (output / "last.pt").is_file():
        raise FileNotFoundError("Resume requires a committed epoch-boundary last.pt")
    setup_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    root, cfg, test_module, model_class, data_module, utils, _ = official_runtime(args.dsanet_root, args.dataset)
    resolve_cfg(cfg, root)
    cfg.seed = args.seed
    labels = UCF_LABEL_MAP if args.dataset == "ucf" else XD_LABEL_MAP
    prompt = utils.get_prompt_text(labels)
    primary, secondary, steps_per_epoch = loader_factory(cfg, args, labels, device)
    loaders = (primary, secondary)
    metadata = primary.dataset.metadata
    if (metadata["dataset"] != args.dataset or int(metadata["seed"]) != args.seed or
            metadata["numerical_mode"] != "stable" or metadata["scopes"] != list(SCOPES) or
            metadata["anchor_sha256"] != file_hash(args.init_checkpoint) or
            metadata.get("measurement_source") != measurement_signature() or
            metadata.get("upstream_source") != upstream_signature(root) or
            metadata["generation_config"]["topk_divisor"] != args.topk_divisor):
        raise ValueError("Cache/anchor/measurement identity mismatch; do not rebuild or bypass silently")
    if secondary is not None and secondary.dataset.metadata != metadata:
        raise ValueError("Balanced loaders have different cache identities")
    if not any(row["supported_queries"] for row in primary.dataset.index["rows"].values()):
        raise ValueError("No supported response queries; this is not the full specified setting")
    allocation = None
    if event_study:
        from dsanet_repair.eventstudy.audit import manifest
        allocation = manifest(loaders)
    source = source_signature(root)
    environment = {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": np.__version__,
                   "cuda": torch.version.cuda, "device": str(device),
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"}
    identity = reference.run_identity(args, metadata, source)
    backbone = make_backbone(model_class, cfg, device, "stable").to(device)
    backbone.load_state_dict(load_initial_state(args.init_checkpoint), strict=True)
    model, selected = model_for_variant(backbone, args)
    model.to(device)
    names = [name for name, _ in selected]
    groups = [{"params": group, "name": name} for name, group in model.parameter_partitions().items()]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * steps_per_epoch,
                                                          eta_min=args.lr * .1)
    start_epoch, step, previous_seconds, initial = 0, 0, 0., None
    if args.resume:
        payload = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT or payload.get("identity") != identity:
            raise ValueError("Cannot resume: configuration/code/cache/anchor identity changed")
        apply_delta(model, payload, names)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch, step, previous_seconds = payload["epoch"], payload["step"], payload["training_seconds"]
        probe, initial = payload["initialization"], payload["initial"]
        reference.restore_rng(payload["rng"], loaders)
    else:
        probe = reference.initialization_probe(model, loaders, prompt, device, args)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        save_atomic({"identity": identity, "config": vars(args), "cache": metadata, "source": source,
                     "initialization": probe, "environment": environment}, output / "run.json")
        if allocation is not None:
            save_atomic(allocation, output / "response_holdout.json")
        if args.evaluate_initial:
            state = reference.rng_state(loaders)
            initial = evaluation_function(model, args.dataset, cfg, labels, test_module, data_module, utils, device, output / "initial")
            reference.restore_rng(state, loaders)
    model.train()
    if event_study:
        allocation_file = output / "response_holdout.json"
        if not allocation_file.exists() or json.loads(allocation_file.read_text()) != allocation:
            raise ValueError("Query-holdout allocation changed; cannot resume this experiment")
    clear_text_cache(model)
    from dsanet_repair.frontier.checkpoints import milestone
    def evaluator(destination):
        return evaluation_function(model, args.dataset, cfg, labels, test_module, data_module, utils, device, destination)
    # A crash after last.pt but before snapshot/evaluation is recoverable without
    # rerunning optimization or changing the committed RNG state.
    if args.resume:
        milestone(output, start_epoch, args, identity, model, loaders, evaluator)
    started = time.monotonic()
    milestone_seconds = 0.
    invocation = time.time_ns()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, args.epochs):
        for batch_index, batch in enumerate(epoch_batches(primary, secondary)):
            if event_study:
                model.training_step, model.steps_per_epoch = step, steps_per_epoch
            losses = loss_function(model, batch, prompt, cfg, device, args)
            if not all(torch.isfinite(value).all() for value in losses.values()):
                raise FloatingPointError(f"Nonfinite loss or diagnostic at {epoch + 1}:{batch_index + 1}")
            diagnostic = diagnostics(model, losses) if not event_study and step % args.diagnose_every == 0 else {}
            updates = optimizer_step(model, losses, optimizer, args)
            if not all(math.isfinite(value) for value in updates.values()):
                raise FloatingPointError("Nonfinite optimization diagnostic")
            scheduler.step()
            step += 1
            record = {"phase": "train", "invocation": invocation, "variant": args.variant,
                      "epoch": epoch + 1, "batch": batch_index + 1, "step": step,
                      **{name: float(value.detach()) for name, value in losses.items()},
                      **diagnostic, **updates, "lr": optimizer.param_groups[0]["lr"]}
            # Only checkpoints commit epochs; retain invocation IDs for interrupted logs.
            with (output / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if batch_index % args.log_every == 0:
                print(json.dumps(record), flush=True)
        save_checkpoint(output / "last.pt", model, names, optimizer, scheduler, epoch + 1, step, args, metadata,
                        identity, loaders, probe, initial, source,
                        previous_seconds + time.monotonic() - started - milestone_seconds)
        milestone_seconds += milestone(output, epoch + 1, args, identity, model, loaders, evaluator)
    training_seconds = previous_seconds + time.monotonic() - started - milestone_seconds
    eval_started = time.monotonic()
    cached_final = output / "epochs" / f"epoch_{args.epochs:03d}" / "evaluation.json"
    if not args.skip_evaluation and args.evaluate_snapshots and cached_final.exists():
        record = json.loads(cached_final.read_text())
        if record.get("identity") != identity or record.get("evaluation_directory") != "final":
            raise ValueError("Final snapshot evaluation identity/path mismatch")
        final, evaluation_seconds = record["metrics"], record["evaluation_seconds"]
    else:
        final = None if args.skip_evaluation else evaluator(output / "final")
        evaluation_seconds = time.monotonic() - eval_started
    summary = {"variant": args.variant, "dataset": args.dataset, "seed": args.seed, "identity": identity,
               "anchor_sha256": metadata["anchor_sha256"], "cache_identity": metadata["identity"],
               "epochs": args.epochs, "steps": step, "checkpoint_policy": "fixed_final_epoch_not_test_best",
               "numerical_mode": "stable", "trainable_parameters": sum(p.numel() for _, p in selected),
               "trainable_names": names, "initialization": probe, "initial": initial, "final": final,
               "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
               "cache_build_audit": primary.dataset.index.get("audit"), "evaluation_skipped": args.skip_evaluation,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "environment": environment,
               "snapshot_epochs": args.snapshot_epochs, "milestone_seconds_this_invocation": milestone_seconds,
               "data_schedule": "balanced min-loader epochs" if args.dataset == "ucf" else "all CSV rows per epoch"}
    if event_study:
        from dsanet_repair.eventstudy.audit import evaluate_heldout
        model.training_step, model.steps_per_epoch = step, steps_per_epoch
        audit_started = time.monotonic()
        held = evaluate_heldout(model, loaders, prompt, cfg, args, device, allocation)
        held.update(identity=identity, epoch=args.epochs)
        save_atomic(held, output / "heldout_response.json")
        summary.update(response_holdout_sha256=allocation["sha256"], heldout_response=held,
                       response_audit_seconds=time.monotonic() - audit_started,
                       event_configuration=vars(model.event_configuration),
                       event_parameters=sum(p.numel() for p in model.event_core.parameters())
                       if model.event_core is not None else 0)
        summary["evaluation_inputs"] = {key: file_hash(getattr(cfg, key)) for key in
            ("test_list", "gt_path", "gt_segment_path", "gt_label_path")} if not args.skip_evaluation else None
        summary["runtime_protocol"] = {key: getattr(cfg, key, None) for key in
                                       ("visual_length", "temp", "loss2_weight")}
    save_atomic(summary, output / "summary.json")
    print(summary, flush=True)
    return summary


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", choices=("ucf", "xd"), required=True)
    value.add_argument("--variant", choices=VARIANTS + tuple(REFINEMENTS) + tuple(EVENTS), required=True)
    for name in ("dsanet-root", "response-cache", "init-checkpoint", "output-dir"):
        value.add_argument("--" + name, required=True)
    integers = {"seed": 234, "epochs": 3, "batch-size": 8, "workers": 4, "hidden": 128,
                "topk-divisor": 16, "init-probe-batches": 2, "diagnose-every": 50, "log-every": 20,
                "temporal-levels": 4, "temporal-kernel": 7, "transport-iterations": 25}
    floats = {"lr": 1e-4, "weight-decay": 1e-4, "clip-grad": 1., "init-atol": 2e-4,
              "response-weight": .5, "local-weight": .25, "outside-weight": .1, "normal-weight": .1,
              "anchor-weight": .1, "anchor-radius": .25, "binary-radius": .5, "semantic-radius": 2.,
              "prototype-temperature": .1, "transport-weight": .1, "transport-alpha": .3,
              "transport-radius": .04, "transport-entropy": .07, "transport-mass": .05,
              "transport-geometry-temperature": .2, "transport-step": 1.,
              "auxiliary-gradient-ratio": .5, "head-alignment-weight": 1., "head-alignment-cap": .1}
    for name, default in integers.items():
        value.add_argument("--" + name, type=int, default=default)
    for name, default in floats.items():
        value.add_argument("--" + name, type=float, default=default)
    value.add_argument("--device", default="auto")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--evaluate-initial", action="store_true")
    value.add_argument("--skip-evaluation", action="store_true", help="Engineering tests only; not benchmark evidence")
    value.add_argument("--snapshot-epochs", nargs="+", type=int, default=[])
    value.add_argument("--evaluate-snapshots", action="store_true", help="Diagnostic curves only; never choose test-best")
    add_arguments(value)
    from dsanet_repair.eventstudy import add_arguments as add_event_arguments
    add_event_arguments(value)
    return value


if __name__ == "__main__":
    train(parser().parse_args())
