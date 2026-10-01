"""Train Ours-full: DSANet + A1/A2 cached responses + A3 response learning.

Run this module from the project root.  It imports the pinned upstream DSANet
implementation without modifying it, preserves the paper losses/evaluator, and
exports a plain DSANet state dict for final official testing.
"""

import argparse
import importlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CachedResponseDataset, UCF_LABEL_MAP, XD_LABEL_MAP
from .losses import ResponseLossConfig, intervention_response_loss
from .model import ResponseAugmentedDSANet
from .official_losses import paper_loss
from .numerics import fresh_evaluation, stable_model_class


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def official_runtime(dsanet_root, dataset):
    root = Path(dsanet_root).resolve()
    source = root / "src"
    if not (source / "model.py").exists():
        raise FileNotFoundError(f"Not a DSANet checkout: {root}")
    sys.path.insert(0, str(source))
    option = importlib.import_module(f"{dataset}_option")
    test_module = importlib.import_module(f"{dataset}_test")
    model_module = importlib.import_module("model")
    data_module = importlib.import_module("utils.dataset")
    tools = importlib.import_module("utils.tools")
    stable = importlib.import_module("utils.StableAdamW")
    return root, option.parser.parse_args([]), test_module, model_module.DSANet, data_module, tools, stable.StableAdamW


def make_backbone(model_class, cfg, device, numerical_mode="legacy"):
    if numerical_mode == "stable":
        model_class = stable_model_class(model_class)
    elif numerical_mode != "legacy":
        raise ValueError("Unknown numerical mode")
    return model_class(
        cfg.classes_num, cfg.embed_dim, cfg.visual_length, cfg.visual_width,
        cfg.visual_head, cfg.visual_layers, cfg.attn_window, cfg.prompt_prefix,
        cfg.prompt_postfix, cfg, device,
    )


def load_initial_state(path):
    """Load an official, wrapped, or exported DSANet state dictionary.

    The official DSANet release stores a bare ``OrderedDict`` while our
    Ours-full checkpoints store the wrapped model under ``model`` with a
    ``backbone.`` prefix.  Normalizing both forms here lets the same trainer
    run either a paper checkpoint or one of our exported checkpoints without
    changing the model definition.
    """
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            nested = value.get(key)
            if isinstance(nested, dict):
                value = nested
                break
    if not isinstance(value, dict) or not value or not all(isinstance(k, str) for k in value):
        raise ValueError(f"Unsupported DSANet checkpoint format: {path}")
    if any(k.startswith("backbone.") for k in value):
        value = {
            k.removeprefix("backbone."): tensor
            for k, tensor in value.items()
            if k.startswith("backbone.")
        }
    return value


def unpack(batch, device):
    features, _label_text, lengths, identities, targets, weights, labels = batch
    return (features.to(device), lengths.to(device), identities, targets.to(device),
            weights.to(device), labels.to(device))


def merge_batches(normal_batch, anomaly_batch, device):
    normal = unpack(normal_batch, device)
    anomaly = unpack(anomaly_batch, device)
    return tuple(torch.cat([normal[i], anomaly[i]], 0) if i != 2 else tuple(normal[i]) + tuple(anomaly[i])
                 for i in range(len(normal)))


def evaluate(backbone, dataset, cfg, label_map, root, test_module, data_module, tools, device):
    dataset_class = data_module.UCFDataset if dataset == "ucf" else data_module.XDDataset
    test_data = dataset_class(cfg.visual_length, cfg.test_list, True, label_map)
    loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    prompt = tools.get_prompt_text(label_map)
    gt = np.load(cfg.gt_path)
    segments = np.load(cfg.gt_segment_path, allow_pickle=True)
    labels = np.load(cfg.gt_label_path, allow_pickle=True)
    with fresh_evaluation(backbone):
        if dataset == "ucf":
            auc, ap = test_module.test(backbone, loader, cfg.visual_length, prompt, gt, segments,
                                       labels, cfg.DNP_use, device, cfg)
            primary = auc
        else:
            auc, ap, _ = test_module.test(backbone, loader, cfg.visual_length, prompt, gt, segments,
                                          labels, cfg.DNP_use, cfg, device)
            primary = ap
    return {"auc": float(auc), "ap": float(ap), "primary": float(primary)}


def save_checkpoint(path, model, optimizers, schedulers, epoch, step, best, args, cfg):
    payload = {
        "format": "dsanet-ours-full-v1", "seed": args.seed, "dataset": args.dataset,
        "epoch": epoch, "step": step, "best": best,
        "model": model.state_dict(),
        "optimizers": [x.state_dict() for x in optimizers],
        "schedulers": [x.state_dict() for x in schedulers],
        "paper_config": vars(cfg), "method_config": vars(args),
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                "python": random.getstate()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    setup_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    root, cfg, test_module, model_class, data_module, tools, stable_adamw = official_runtime(
        args.dsanet_root, args.dataset
    )
    # Resolve every upstream relative path against the pinned checkout.
    for name in ("train_list", "test_list", "gt_path", "gt_segment_path", "gt_label_path"):
        value = Path(getattr(cfg, name))
        setattr(cfg, name, str(value if value.is_absolute() else root / value))
    cfg.seed = args.seed
    cfg.max_epoch = args.epochs if args.epochs is not None else cfg.max_epoch
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    label_map = UCF_LABEL_MAP if args.dataset == "ucf" else XD_LABEL_MAP
    prompt = tools.get_prompt_text(label_map)

    backbone = make_backbone(model_class, cfg, device, args.numerical_mode)
    if args.init_checkpoint:
        initial_state = load_initial_state(args.init_checkpoint)
        missing, unexpected = backbone.load_state_dict(initial_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Initial DSANet checkpoint is incompatible: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        print(json.dumps({
            "phase": "init_checkpoint",
            "dataset": args.dataset,
            "checkpoint": str(Path(args.init_checkpoint).resolve()),
            "keys": len(initial_state),
        }), flush=True)
    model = ResponseAugmentedDSANet(backbone, args.response_dropout).to(device)
    common = dict(csv_path=cfg.train_list, cache_path=args.response_cache, label_map=label_map,
                  visual_length=cfg.visual_length, require_positive_cache=True)
    generator = torch.Generator().manual_seed(args.seed)
    loader_kw = dict(batch_size=cfg.batch_size, shuffle=True, num_workers=args.workers,
                     pin_memory=device.type == "cuda", generator=generator, drop_last=args.dataset == "ucf")
    if args.dataset == "ucf":
        normal_loader = DataLoader(CachedResponseDataset(subset="normal", **common), **loader_kw)
        anomaly_loader = DataLoader(CachedResponseDataset(subset="anomaly", **common), **loader_kw)
        batches_per_epoch = min(len(normal_loader), len(anomaly_loader))
    else:
        train_loader = DataLoader(CachedResponseDataset(subset=None, **common), **loader_kw)
        batches_per_epoch = len(train_loader)

    refiner, main = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (refiner if "video_anomaly_refiner" in name else main).append(parameter)
    learning_rate = cfg.lr if args.lr is None else args.lr
    optimizer_main = torch.optim.AdamW(main, lr=learning_rate)
    optimizer_refiner = stable_adamw(
        [{"params": refiner}], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4,
        amsgrad=True, eps=1e-10,
    )
    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_main, T_max=cfg.max_epoch)
    # The upstream custom warm scheduler can divide by zero on very short smoke runs;
    # cosine preserves the same terminal LR and is used only for the added runner.
    scheduler_refiner = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_refiner, T_max=max(1, cfg.max_epoch * batches_per_epoch), eta_min=learning_rate * 0.1
    )
    optimizers = (optimizer_main, optimizer_refiner)
    schedulers = (scheduler_main, scheduler_refiner)
    response_cfg = ResponseLossConfig(
        reconstruction_weight=1.0, ranking_weight=args.ranking_weight,
        absent_class_weight=args.absent_class_weight, temporal_weight=args.temporal_weight,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.jsonl"
    best = {"primary": float("-inf"), "epoch": -1, "step": -1}
    global_step = 0
    start_epoch = 0

    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("dataset") != args.dataset or checkpoint.get("seed") != args.seed:
            raise ValueError(
                f"Resume checkpoint mismatch: dataset={checkpoint.get('dataset')!r}, "
                f"seed={checkpoint.get('seed')!r}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("step", 0))
        best = checkpoint.get("best", best)
        # The recovery intentionally resets optimizer moments.  The failed
        # UCF run used an unstable 7e-5 LR; carrying those moments into the
        # stabilized 1e-5 continuation would reintroduce the same spike.
        rng = checkpoint.get("rng", {})
        if "torch" in rng:
            # The checkpoint is loaded onto ``device``; on a CUDA resume this
            # makes the saved CPU RNG byte tensor a CUDA tensor, while
            # torch.set_rng_state requires the CPU generator state.
            torch.set_rng_state(rng["torch"].detach().cpu())
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "python" in rng:
            random.setstate(rng["python"])
        print(json.dumps({
            "phase": "resume", "dataset": args.dataset, "seed": args.seed,
            "checkpoint": str(resume_path), "start_epoch": start_epoch,
            "global_step": global_step, "learning_rate": learning_rate,
        }), flush=True)
        # Continue the cosine schedule at the recovered epoch while retaining
        # freshly initialized optimizer moments.
        if start_epoch > 0:
            scheduler_main.step(start_epoch)
            scheduler_refiner.step(start_epoch * batches_per_epoch)

    for epoch in range(start_epoch, cfg.max_epoch):
        model.train()
        if args.dataset == "ucf":
            batches = (merge_batches(a, b, device) for a, b in zip(normal_loader, anomaly_loader))
        else:
            batches = (unpack(x, device) for x in train_loader)
        for batch_index, (features, lengths, identities, targets, weights, labels) in enumerate(batches):
            try:
                dsanet_output, response = model(features, None, prompt, lengths, cfg.DNP_use)
                base_loss, base_parts = paper_loss(
                    dsanet_output, labels, lengths, cfg.loss2_weight, cfg.DNP_use
                )
                response_loss, response_parts = intervention_response_loss(
                    response, targets, weights, lengths, labels, response_cfg
                )
                loss = base_loss + args.response_weight * response_loss
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch + 1}, batch={batch_index + 1}, "
                        f"global_step={global_step + 1}; reduce learning rate or inspect logits"
                    )
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad, error_if_nonfinite=True
                )
                for optimizer in optimizers:
                    optimizer.step()
            except FloatingPointError as exc:
                if not args.allow_skip_nonfinite:
                    raise
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                skipped = {
                    "phase": "skip_nonfinite", "dataset": args.dataset, "seed": args.seed,
                    "epoch": epoch + 1, "batch": batch_index + 1,
                    "global_step": global_step, "error": str(exc),
                }
                with log_path.open("a") as handle:
                    handle.write(json.dumps(skipped) + "\n")
                print(json.dumps(skipped), flush=True)
                continue
            except RuntimeError as exc:
                message = str(exc).lower()
                if "non-finite" not in message and "nonfinite" not in message:
                    raise
                if not args.allow_skip_nonfinite:
                    raise
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                skipped = {
                    "phase": "skip_nonfinite", "dataset": args.dataset, "seed": args.seed,
                    "epoch": epoch + 1, "batch": batch_index + 1,
                    "global_step": global_step, "error": str(exc),
                }
                with log_path.open("a") as handle:
                    handle.write(json.dumps(skipped) + "\n")
                print(json.dumps(skipped), flush=True)
                continue
            scheduler_refiner.step()
            global_step += 1
            record = {
                "phase": "train", "dataset": args.dataset, "seed": args.seed,
                "epoch": epoch + 1, "batch": batch_index + 1, "global_step": global_step,
                "loss": float(loss.detach()), "paper_loss": float(base_loss.detach()),
                "response_loss": float(response_loss.detach()),
                **{k: float(v) for k, v in response_parts.items()},
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if batch_index % args.log_every == 0:
                print(json.dumps(record), flush=True)

            paper_examples = (batch_index * cfg.batch_size * (2 if args.dataset == "ucf" else 1))
            should_evaluate = args.dataset == "ucf" and paper_examples > 0 and paper_examples % 1280 == 0
            if should_evaluate:
                metrics = evaluate(backbone, args.dataset, cfg, label_map, root, test_module,
                                   data_module, tools, device)
                if metrics["primary"] > best["primary"]:
                    best = {**metrics, "epoch": epoch + 1, "step": global_step}
                    save_checkpoint(output / "best_full.pt", model, optimizers, schedulers,
                                    epoch + 1, global_step, best, args, cfg)
        scheduler_main.step()
        if args.dataset == "xd" or best["epoch"] < 0:
            metrics = evaluate(backbone, args.dataset, cfg, label_map, root, test_module,
                               data_module, tools, device)
            if metrics["primary"] > best["primary"]:
                best = {**metrics, "epoch": epoch + 1, "step": global_step}
                save_checkpoint(output / "best_full.pt", model, optimizers, schedulers,
                                epoch + 1, global_step, best, args, cfg)
        save_checkpoint(output / "last_full.pt", model, optimizers, schedulers,
                        epoch + 1, global_step, best, args, cfg)

    checkpoint = torch.load(output / "best_full.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    # Stable mode has the same keys, but must be replayed with its stable forward.
    torch.save(model.export_backbone_state(), output / f"model_{args.dataset}_ours_full.pth")
    final = evaluate(backbone, args.dataset, cfg, label_map, root, test_module, data_module, tools, device)
    summary = {"method": "Ours-full", "dataset": args.dataset, "seed": args.seed,
               "numerical_mode": args.numerical_mode,
               "allow_skip_nonfinite": args.allow_skip_nonfinite,
               "selection_replay_delta": final["primary"] - checkpoint["best"]["primary"],
               "best": best, "final": final, "elapsed_seconds": time.monotonic() - args._started}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description="DSANet + matched intervention-response learning")
    p.add_argument("--dataset", choices=("ucf", "xd"), required=True)
    p.add_argument("--dsanet-root", default="third_party/DSANet")
    p.add_argument("--response-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=234)
    p.add_argument(
        "--init-checkpoint",
        help="Initialize the official DSANet backbone from a .pth/state-dict checkpoint.",
    )
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume")
    p.add_argument("--numerical-mode", choices=("legacy", "stable"), default="legacy",
                   help="Opt-in pooling repair; apply identically to both comparison rows.")
    p.add_argument("--allow-skip-nonfinite", action="store_true",
                   help="Recovery only; skipped-batch runs are not clean comparisons.")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--response-weight", type=float, default=0.5)
    p.add_argument("--ranking-weight", type=float, default=0.25)
    p.add_argument("--absent-class-weight", type=float, default=0.05)
    p.add_argument("--temporal-weight", type=float, default=0.02)
    p.add_argument("--response-dropout", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=5.0)
    p.add_argument("--log-every", type=int, default=20)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments._started = time.monotonic()
    train(arguments)
