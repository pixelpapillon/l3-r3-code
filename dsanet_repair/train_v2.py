"""Matched Control/Ours-v2 tail fine-tuning from one verified DSANet anchor."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .adapter import raw_dsanet_logits
from .data import UCF_LABEL_MAP, XD_LABEL_MAP
from .data_v2 import PairedEditDataset
from .decision_constraints import anchor_trust_loss, decision_response_loss
from .numerics import clear_text_cache
from .official_losses import mil_binary, mil_class
from .train_full import (
    evaluate, load_initial_state, make_backbone, official_runtime, setup_seed,
)


TAIL_PREFIXES = ("linear.", "classifier.", "mlp2.")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_tail_parameters(model):
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TAIL_PREFIXES))
        if parameter.requires_grad:
            selected.append((name, parameter))
    if not selected:
        raise RuntimeError("No DSANet tail parameters were selected")
    return selected


def merge_batches(left, right):
    result = {}
    for key in left:
        if key == "identity":
            result[key] = list(left[key]) + list(right[key])
        else:
            result[key] = torch.cat([left[key], right[key]], dim=0)
    return result


def make_loaders(cfg, args, label_map, device):
    common = dict(
        csv_path=cfg.train_list, cache_path=args.response_cache,
        label_map=label_map, visual_length=cfg.visual_length,
        max_rows=args.max_train_rows,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", generator=generator,
        persistent_workers=args.workers > 0, drop_last=args.dataset == "ucf",
    )
    if args.dataset == "ucf":
        normal = DataLoader(PairedEditDataset(subset="normal", **common), **loader_options)
        anomaly_generator = torch.Generator().manual_seed(args.seed + 1)
        anomaly = DataLoader(
            PairedEditDataset(subset="anomaly", **common),
            **{**loader_options, "generator": anomaly_generator},
        )
        return normal, anomaly, min(len(normal), len(anomaly))
    loader = DataLoader(PairedEditDataset(subset=None, **common), **loader_options)
    return loader, None, len(loader)


def epoch_batches(primary, secondary):
    if secondary is None:
        yield from primary
    else:
        for normal, anomaly in zip(primary, secondary):
            yield merge_batches(normal, anomaly)


def gradient_diagnostics(video_loss, response_loss, parameters):
    video = torch.autograd.grad(video_loss, parameters, retain_graph=True, allow_unused=True)
    response = torch.autograd.grad(response_loss, parameters, retain_graph=True, allow_unused=True)
    video_flat = torch.cat([value.detach().reshape(-1) for value in video if value is not None])
    response_flat = torch.cat([value.detach().reshape(-1) for value in response if value is not None])
    video_norm = float(video_flat.norm()) if len(video_flat) else 0.0
    response_norm = float(response_flat.norm()) if len(response_flat) else 0.0
    cosine = 0.0
    if video_norm > 0 and response_norm > 0:
        # Parameter lists have matching order but some entries can be unused by one loss.
        products = []
        for left, right in zip(video, response):
            if left is not None and right is not None:
                products.append((left.detach() * right.detach()).sum())
        if products:
            cosine = float(torch.stack(products).sum() / (video_norm * response_norm))
    return {"video_grad_norm": video_norm, "response_grad_norm": response_norm,
            "gradient_cosine": cosine}


def batch_losses(model, batch, prompt, cfg, device, args):
    features = batch["features"].to(device, non_blocking=True)
    edited = batch["edited"].to(device, non_blocking=True)
    lengths = batch["length"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    batch_size, queries, steps, width = edited.shape
    combined = torch.cat([features, edited.reshape(batch_size * queries, steps, width)], dim=0)
    combined_lengths = torch.cat([lengths, lengths.repeat_interleave(queries)])
    output_all = model(combined, None, prompt, combined_lengths, False)
    binary_all, classes_all = raw_dsanet_logits(output_all)
    raw_all = torch.cat([binary_all.unsqueeze(-1), classes_all], dim=-1)
    original_raw = raw_all[:batch_size]
    edited_raw = raw_all[batch_size:].reshape(batch_size, queries, steps, -1)
    binary_loss = mil_binary(output_all[1][:batch_size], labels, lengths)
    semantic_loss = mil_class(output_all[2][:batch_size], labels, lengths)
    video_loss = binary_loss + cfg.loss2_weight * semantic_loss
    trust = anchor_trust_loss(
        original_raw, batch["anchor"].to(device, non_blocking=True), lengths,
        radius=args.anchor_radius,
    )
    response_loss, response_parts = decision_response_loss(
        original_raw, edited_raw, lengths,
        batch["lower"].to(device, non_blocking=True),
        batch["upper"].to(device, non_blocking=True),
        batch["reliability"].to(device, non_blocking=True),
        batch["null_response"].to(device, non_blocking=True),
        divisor=args.topk_divisor,
    )
    return {
        "video": video_loss, "binary": binary_loss, "semantic": semantic_loss,
        "trust": trust, "response": response_loss, "response_parts": response_parts,
    }


def response_signal_gate(model, primary, secondary, prompt, cfg, device, args, parameters):
    if args.variant == "control":
        return None
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    probe_primary = DataLoader(primary.dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=0, drop_last=False)
    probe_secondary = None
    if secondary is not None:
        probe_secondary = DataLoader(secondary.dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=0, drop_last=False)
    checked = []
    try:
        model.train()
        clear_text_cache(model)
        for batch in epoch_batches(probe_primary, probe_secondary):
            losses = batch_losses(model, batch, prompt, cfg, device, args)
            if float(losses["response_parts"]["supported_mass"]) <= 0:
                continue
            diagnostic = gradient_diagnostics(losses["video"], losses["response"], parameters)
            diagnostic["response_loss"] = float(losses["response"].detach())
            diagnostic["supported_mass"] = float(losses["response_parts"]["supported_mass"])
            checked.append(diagnostic)
            print(json.dumps({"phase": "gradient_probe", "probe": len(checked), **diagnostic}),
                  flush=True)
            if diagnostic["response_grad_norm"] > args.min_response_grad_norm:
                return {"probes": len(checked), **diagnostic}
            if len(checked) >= args.signal_probe_batches:
                break
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        clear_text_cache(model)
    raise RuntimeError(
        f"Ours-v2 found no corrective response gradient in {len(checked)} supported initialization probes"
    )


def save_delta(path, model, trainable_names, optimizer, scheduler, epoch, step, args, metadata):
    state = model.state_dict()
    payload = {
        "format": "dsanet-v2-tail-delta-v1",
        "dataset": args.dataset, "variant": args.variant, "seed": args.seed,
        "epoch": epoch, "step": step, "anchor_checkpoint": args.init_checkpoint,
        "anchor_sha256": metadata["anchor_sha256"],
        "numerical_mode": args.numerical_mode,
        "trainable_names": list(trainable_names),
        "tail_state": {name: state[name].detach().cpu() for name in trainable_names},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "config": vars(args),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    started = time.monotonic()
    setup_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    root, cfg, test_module, model_class, data_module, tools, _stable = official_runtime(
        args.dsanet_root, args.dataset
    )
    for name in ("train_list", "test_list", "gt_path", "gt_segment_path", "gt_label_path"):
        value = Path(getattr(cfg, name))
        setattr(cfg, name, str(value if value.is_absolute() else root / value))
    cfg.seed = args.seed
    label_map = UCF_LABEL_MAP if args.dataset == "ucf" else XD_LABEL_MAP
    prompt = tools.get_prompt_text(label_map)

    model = make_backbone(model_class, cfg, device, args.numerical_mode).to(device)
    model.load_state_dict(load_initial_state(args.init_checkpoint), strict=True)
    anchor_sha256 = file_hash(args.init_checkpoint)
    primary, secondary, batches_per_epoch = make_loaders(cfg, args, label_map, device)
    cache_metadata = primary.dataset.cache_metadata
    if cache_metadata.get("dataset") != args.dataset or int(cache_metadata.get("seed", -1)) != args.seed:
        raise ValueError("V2 cache dataset/seed mismatch")
    if cache_metadata.get("numerical_mode") != args.numerical_mode:
        raise ValueError("V2 cache numerical mode mismatch")
    if cache_metadata.get("anchor_sha256") != anchor_sha256:
        raise ValueError("V2 cache was measured against a different anchor checkpoint")
    if secondary is not None and secondary.dataset.cache_metadata != cache_metadata:
        raise ValueError("Balanced UCF loaders disagree on v2 cache metadata")

    selected = select_tail_parameters(model)
    trainable_names = [name for name, _ in selected]
    parameters = [parameter for _, parameter in selected]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * batches_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.jsonl"

    initial_metrics = None
    if args.evaluate_initial:
        initial_metrics = evaluate(
            model, args.dataset, cfg, label_map, root, test_module, data_module, tools, device
        )
        print(json.dumps({"phase": "initial_evaluate", **initial_metrics}), flush=True)
    model.train()
    clear_text_cache(model)
    global_step = 0
    gradient_record = response_signal_gate(
        model, primary, secondary, prompt, cfg, device, args, parameters,
    )
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(epoch_batches(primary, secondary)):
            losses = batch_losses(model, batch, prompt, cfg, device, args)
            video_loss = losses["video"]
            binary_loss = losses["binary"]
            semantic_loss = losses["semantic"]
            trust = losses["trust"]
            response_loss = losses["response"]
            response_parts = losses["response_parts"]
            if args.variant == "ours-v2":
                total = video_loss + args.anchor_weight * trust + args.response_weight * response_loss
            else:
                # Preserve the same edited forward/loss graph and compute budget while
                # withholding the response gradient from the matched control.
                total = video_loss + args.anchor_weight * trust + response_loss * 0

            if not torch.isfinite(total).all():
                raise FloatingPointError(
                    f"Non-finite v2 loss at epoch={epoch + 1} batch={batch_index + 1}"
                )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, args.clip_grad, error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            record = {
                "phase": "train_v2", "variant": args.variant, "dataset": args.dataset,
                "seed": args.seed, "epoch": epoch + 1, "batch": batch_index + 1,
                "step": global_step, "loss": float(total.detach()),
                "video_loss": float(video_loss.detach()),
                "binary_loss": float(binary_loss.detach()),
                "semantic_loss": float(semantic_loss.detach()),
                "anchor_loss": float(trust.detach()),
                "response_loss": float(response_loss.detach()),
                "supported_mass": float(response_parts["supported_mass"]),
                "gradient_norm": float(gradient_norm),
                "lr": optimizer.param_groups[0]["lr"],
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if batch_index % args.log_every == 0:
                print(json.dumps(record), flush=True)
        save_delta(
            output / "tail_delta.pt", model, trainable_names, optimizer, scheduler,
            epoch + 1, global_step, args, cache_metadata,
        )
    final_metrics = None
    if not args.skip_final_evaluation:
        final_metrics = evaluate(
            model, args.dataset, cfg, label_map, root, test_module, data_module, tools, device
        )
    summary = {
        "method": "Control" if args.variant == "control" else "Ours-v2",
        "variant": args.variant, "dataset": args.dataset, "seed": args.seed,
        "checkpoint_policy": "fixed_final_epoch", "epochs": args.epochs,
        "steps": global_step, "numerical_mode": args.numerical_mode,
        "anchor_sha256": anchor_sha256, "cache_signature_mode": "paired-v2",
        "trainable_names": trainable_names, "initial": initial_metrics,
        "gradient_gate": gradient_record, "final": final_metrics,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", choices=("ucf", "xd"), required=True)
    value.add_argument("--variant", choices=("control", "ours-v2"), required=True)
    value.add_argument("--dsanet-root", default="third_party/DSANet")
    value.add_argument("--response-cache", required=True)
    value.add_argument("--init-checkpoint", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--seed", type=int, default=234)
    value.add_argument("--epochs", type=int, default=3)
    value.add_argument("--batch-size", type=int, default=8)
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--lr", type=float, default=1e-5)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--response-weight", type=float, default=0.5)
    value.add_argument("--anchor-weight", type=float, default=0.1)
    value.add_argument("--anchor-radius", type=float, default=0.25)
    value.add_argument("--clip-grad", type=float, default=1.0)
    value.add_argument("--topk-divisor", type=int, default=16)
    value.add_argument("--min-response-grad-norm", type=float, default=1e-10)
    value.add_argument("--signal-probe-batches", type=int, default=16)
    value.add_argument("--numerical-mode", choices=("legacy", "stable"), default="stable")
    value.add_argument("--device", default="auto")
    value.add_argument("--log-every", type=int, default=20)
    value.add_argument("--evaluate-initial", action="store_true")
    value.add_argument("--max-train-rows", type=int,
                       help="Smoke-test only: cap rows after normal/anomaly filtering.")
    value.add_argument("--skip-final-evaluation", action="store_true",
                       help="Smoke-test only: do not run the benchmark evaluator.")
    return value


if __name__ == "__main__":
    train(parser().parse_args())
