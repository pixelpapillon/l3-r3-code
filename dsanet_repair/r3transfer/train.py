"""Train ONE R3-transfer variant; the historical runner remains byte-for-byte intact. No cache construction, upload, dispatch or test-best.

An isolated orchestration shell preserves historical producer/replay hashes.
Data scheduling, optimization, checkpoints, metrics and initialization helpers
come from the existing implementation; no historical runner is monkey-patched.
"""

import copy
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from dsanet_repair.data import UCF_LABEL_MAP, XD_LABEL_MAP
from dsanet_repair.expansion import train as shared
from dsanet_repair.expansion.optimization import optimizer_step
from dsanet_repair.frontier.checkpoints import milestone
from dsanet_repair.eventnext.evaluation import evaluate
from dsanet_repair.revision import train as reference
from dsanet_repair.revision.cache import save_atomic
from dsanet_repair.numerics import clear_text_cache
from dsanet_repair.train_v2 import file_hash, epoch_batches
from dsanet_repair.train_full import official_runtime, make_backbone, load_initial_state, setup_seed
from . import CONFIGS, VARIANTS, PARENTS, validate, legacy_options
from .model import R3Transfer
from dsanet_repair.eventnext import train as historical
from dsanet_repair.eventnext.data import make_loaders as legacy_loaders
from dsanet_repair.eventnext.losses import batch_losses
from dsanet_repair.eventnext.audit import manifest, evaluate_heldout


FORMAT = "dsanet-r3transfer-l1-l3-v1"


def parser():
    value = historical.parser()
    for action in value._actions:
        if action.dest == "variant":
            action.choices = VARIANTS
    value.set_defaults(epochs=9, snapshot_epochs=[3, 6, 9], evaluate_snapshots=True, evaluate_initial=True)
    return value


def source_signature(root):
    values = historical.source_signature(root)
    repo = Path(__file__).resolve().parents[2]
    for path in sorted(Path(__file__).parent.glob("*.py")):
        values[str(path.relative_to(repo))] = file_hash(path)
    # These helpers participate in our lifecycle even though the old source
    # collector did not include all of them.
    for name in ("dsanet_repair/frontier/checkpoints.py", "scheme1_repair/reference.py", "vadcore/types.py"):
        values[name] = file_hash(repo / name)
    for folder in ("eventevolve", "eventrewrite"):
        for path in sorted((repo / "dsanet_repair" / folder).glob("*.py")):
            values[str(path.relative_to(repo))] = file_hash(path)
    return values


def model_for_variant(base, args):
    model = R3Transfer(base, args)
    selected = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    ids = [id(p) for group in model.parameter_partitions().values() for p in group]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for _, p in selected}:
        raise ValueError("Incomplete or overlapping parameter partition")
    return model, selected


def check_cache(metadata, args, root):
    historical.check_cache(metadata, args, root)


def save_checkpoint(path, model, names, optimizer, scheduler, epoch, step, args, metadata,
                    identity, loaders, probe, initial, source, elapsed, joint_sha):
    save_atomic(dict(format=FORMAT, dataset=args.dataset, variant=args.variant, seed=args.seed,
                     epoch=epoch, step=step, identity=identity, anchor_sha256=metadata["anchor_sha256"],
                     cache_identity=metadata["identity"], joint_index_sha256=joint_sha,
                     config=vars(args), trainable_names=names, delta=shared.delta_state(model, names),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     rng=reference.rng_state(loaders), initialization=probe, initial=initial,
                     source=source, training_seconds=elapsed), path)


def train(args):
    validate(args)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Use a new output directory or explicitly resume")
    if args.resume and not (output / "last.pt").is_file():
        raise FileNotFoundError("Resume requires an epoch-committed checkpoint")
    setup_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    root, cfg, test_module, cls, data_module, utils, _ = official_runtime(args.dsanet_root, args.dataset)
    reference.resolve_cfg(cfg, root)
    cfg.seed = args.seed
    labels = UCF_LABEL_MAP if args.dataset == "ucf" else XD_LABEL_MAP
    prompt = utils.get_prompt_text(labels)
    first, second, steps_per_epoch = legacy_loaders(cfg, legacy_options(args), labels, device)
    loaders = first, second
    metadata = first.dataset.metadata
    check_cache(metadata, args, root)
    if second is not None and second.dataset.metadata != metadata:
        raise ValueError("Balanced loaders use different caches")
    if not any(r["supported_queries"] for r in first.dataset.index["rows"].values()):
        raise ValueError("No supported original response queries")
    allocation = manifest(loaders)
    joint_sha = file_hash(Path(args.joint_cache) / "index.json") if args.joint_cache else None
    identity_args = copy.copy(args)
    identity_args.joint_cache = joint_sha  # content-bound, path-portable
    source = source_signature(root)
    identity = reference.run_identity(identity_args, metadata, source)
    base = make_backbone(cls, cfg, device, "stable").to(device)
    base.load_state_dict(load_initial_state(args.init_checkpoint), strict=True)
    model, selected = model_for_variant(base, args)
    model.to(device)
    names = [name for name, _ in selected]
    optimizer = torch.optim.AdamW([dict(params=group, name=name) for name, group in model.parameter_partitions().items()],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * steps_per_epoch, eta_min=args.lr * .1)
    epoch_start = step = 0
    previous_seconds, initial = 0., None
    if args.resume:
        payload = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT or payload.get("identity") != identity:
            raise ValueError("Cannot resume changed L configuration/source/cache/anchor")
        shared.apply_delta(model, payload, names)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        epoch_start, step = payload["epoch"], payload["step"]
        previous_seconds, initial, probe = payload["training_seconds"], payload["initial"], payload["initialization"]
        reference.restore_rng(payload["rng"], loaders)
    else:
        probe = reference.initialization_probe(model, loaders, prompt, device, args)
    environment = dict(python=sys.version.split()[0], torch=str(torch.__version__), numpy=np.__version__,
                       cuda=torch.version.cuda, device=str(device),
                       device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        save_atomic(dict(identity=identity, config=vars(args), cache=metadata, source=source, environment=environment,
                         initialization=probe, joint_index_sha256=joint_sha), output / "run.json")
        save_atomic(allocation, output / "response_holdout.json")
        if args.evaluate_initial and not args.skip_evaluation:
            state = reference.rng_state(loaders)
            initial = evaluate(model, args.dataset, cfg, labels, test_module, data_module, utils, device, output / "initial")
            reference.restore_rng(state, loaders)
    if json.loads((output / "response_holdout.json").read_text()) != allocation:
        raise ValueError("Response holdout allocation changed")
    model.train()
    clear_text_cache(model)
    evaluator = lambda dest: evaluate(model, args.dataset, cfg, labels, test_module, data_module, utils, device, dest)
    if args.resume:
        milestone(output, epoch_start, args, identity, model, loaders, evaluator)
    started, milestone_seconds = time.monotonic(), 0.
    invocation = time.time_ns()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epoch_start, args.epochs):
        for batch_index, batch in enumerate(epoch_batches(*loaders)):
            model.training_step, model.steps_per_epoch = step, steps_per_epoch
            losses = batch_losses(model, batch, prompt, cfg, device, args)
            if not all(torch.isfinite(value).all() for value in losses.values()):
                raise FloatingPointError("Non-finite L loss/diagnostic")
            updates = optimizer_step(model, losses, optimizer, args)
            for name in ("morphology",):
                part = getattr(model.event_core, name, None)
                if part is not None:
                    updates[name + "_gradient_norm"] = sum(
                        float(p.grad.square().sum()) for p in part.parameters() if p.grad is not None) ** .5
            for name in ("slot_evidence", "duration"):
                part = getattr(model, name, None)
                if part is not None:
                    updates[name + "_gradient_norm"] = sum(
                        float(p.grad.square().sum()) for p in part.parameters() if p.grad is not None) ** .5
            if not all(math.isfinite(value) for value in updates.values()):
                raise FloatingPointError("Non-finite L optimizer diagnostic")
            model.commit_references()
            scheduler.step()
            step += 1
            record = dict(phase="train", invocation=invocation, variant=args.variant, epoch=epoch + 1,
                          step=step, batch=batch_index + 1, lr=optimizer.param_groups[0]["lr"],
                          **{key: float(value.detach()) for key, value in losses.items()}, **updates)
            with (output / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if batch_index % args.log_every == 0:
                print(json.dumps(record), flush=True)
        save_checkpoint(output / "last.pt", model, names, optimizer, scheduler, epoch + 1, step, args, metadata,
                        identity, loaders, probe, initial, source,
                        previous_seconds + time.monotonic() - started - milestone_seconds, joint_sha)
        milestone_seconds += milestone(output, epoch + 1, args, identity, model, loaders, evaluator)
    elapsed = previous_seconds + time.monotonic() - started - milestone_seconds
    cached = output / "epochs" / f"epoch_{args.epochs:03d}" / "evaluation.json"
    if not args.skip_evaluation and args.evaluate_snapshots and cached.exists():
        record = json.loads(cached.read_text())
        if record["identity"] != identity or record["evaluation_directory"] != "final":
            raise ValueError("Wrong final evaluation identity")
        final = record["metrics"]
    else:
        final = None if args.skip_evaluation else evaluator(output / "final")
    model.training_step, model.steps_per_epoch = step, steps_per_epoch
    held = evaluate_heldout(model, loaders, prompt, cfg, args, device, allocation)
    save_atomic(held, output / "heldout_response.json")
    summary = dict(variant=args.variant, parent=PARENTS[args.variant], dataset=args.dataset, seed=args.seed,
                   epochs=args.epochs, steps=step, identity=identity, anchor_sha256=metadata["anchor_sha256"],
                   cache_identity=metadata["identity"], joint_index_sha256=joint_sha,
                   response_holdout_sha256=allocation["sha256"], heldout_response=held,
                   configuration=vars(CONFIGS[args.variant]), checkpoint_policy="fixed_final_epoch_not_test_best",
                   initial=initial, final=final, initialization=probe, evaluation_skipped=args.skip_evaluation,
                   trainable_parameters=sum(p.numel() for _, p in selected), training_seconds=elapsed,
                   environment=environment, numerical_mode="stable", snapshot_epochs=args.snapshot_epochs,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   evaluation_inputs={key: file_hash(getattr(cfg, key)) for key in
                       ("test_list", "gt_path", "gt_segment_path", "gt_label_path")} if not args.skip_evaluation else None,
                   runtime_protocol={key: getattr(cfg, key, None) for key in ("visual_length", "temp", "loss2_weight")})
    save_atomic(summary, output / "summary.json")
    return summary


def restore_model(checkpoint, anchor, root_path, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT or payload["anchor_sha256"] != file_hash(anchor):
        raise ValueError("Wrong L checkpoint/anchor")
    args = validate(SimpleNamespace(**payload["config"]))
    root, cfg, module, cls, data, utils, _ = official_runtime(root_path, payload["dataset"])
    if source_signature(root) != payload["source"]:
        raise ValueError("Replay source differs from checkpoint; use the archived source package")
    reference.resolve_cfg(cfg, root)
    base = make_backbone(cls, cfg, device, "stable").to(device)
    base.load_state_dict(load_initial_state(anchor), strict=True)
    model, selected = model_for_variant(base, args)
    model.to(device)
    shared.apply_delta(model, payload, [name for name, _ in selected])
    return model.eval(), payload, cfg, module, data, utils


if __name__ == "__main__":
    print(json.dumps(train(parser().parse_args()), indent=2))
