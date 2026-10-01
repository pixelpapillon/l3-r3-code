"""Three controlled, crop-exact DSANet revisions. No remote job dispatch."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dsanet_repair.adapter import raw_dsanet_logits
from dsanet_repair.data import UCF_LABEL_MAP, XD_LABEL_MAP
from dsanet_repair.decision_constraints import anchor_trust_loss
from dsanet_repair.numerics import clear_text_cache, fresh_evaluation
from dsanet_repair.official_losses import mil_binary, mil_class
from dsanet_repair.train_full import official_runtime, make_backbone, load_initial_state, setup_seed
from dsanet_repair.train_v2 import file_hash, select_tail_parameters, epoch_batches
from .cache import ExactCropDataset, SCOPES, measurement_signature, upstream_signature, save_atomic
from .evaluation import evaluate
from .model import NativeCorrection
from .response import scoped_responses, envelope_loss, feasible_envelopes, normal_edit_loss


FORMAT = "dsanet-crop-exact-revision-delta-v1"
VARIANTS = ("control-corrected", "full-corrected", "integrated")


def resolve_cfg(cfg, root):
    for name in ("train_list", "test_list", "gt_path", "gt_segment_path", "gt_label_path"):
        value = Path(getattr(cfg, name))
        setattr(cfg, name, str(value if value.is_absolute() else root / value))
    cfg.DNP_use = False
    return cfg


def model_for_variant(backbone, args):
    if args.variant == "integrated":
        model = NativeCorrection(backbone, args.hidden, args.binary_radius, args.semantic_radius,
                                 args.prototype_temperature)
        return model, [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    return backbone, select_tail_parameters(backbone)


def make_loaders(cfg, args, label_map, device):
    common = dict(csv_path=cfg.train_list, cache_dir=args.response_cache,
                  label_map=label_map, visual_length=cfg.visual_length)
    loaders = []
    for offset, subset in enumerate(("normal", "anomaly") if args.dataset == "ucf" else (None,)):
        dataset = ExactCropDataset(subset=subset, **common)
        generator = torch.Generator().manual_seed(args.seed + offset)
        loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=device.type == "cuda",
                                  generator=generator, persistent_workers=False,
                                  drop_last=args.dataset == "ucf"))
    steps = min(map(len, loaders))
    if steps < 1:
        raise ValueError("Training subset is smaller than the balanced batch size")
    return loaders[0], loaders[1] if len(loaders) == 2 else None, steps


def flatten_decisions(output):
    binary, margins = raw_dsanet_logits(output)
    return torch.cat([binary[..., None], margins], -1)


def batch_losses(model, batch, prompt, cfg, device, args):
    values = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
              for key, value in batch.items()}
    edited, features, lengths = values["edited"], values["features"], values["length"]
    batch_size, queries, steps, width = edited.shape
    combined = torch.cat([features, edited.reshape(-1, steps, width)])
    combined_lengths = torch.cat([lengths, lengths.repeat_interleave(queries)])
    outputs = model(combined, None, prompt, combined_lengths, False)
    raw = flatten_decisions(outputs)
    original, changed = raw[:batch_size], raw[batch_size:].reshape(batch_size, queries, steps, -1)
    binary = mil_binary(outputs[1][:batch_size], values["labels"], lengths)
    semantic = mil_class(outputs[2][:batch_size], values["labels"], lengths)
    video = binary + cfg.loss2_weight * semantic
    trust = anchor_trust_loss(original, values["anchor"], lengths, radius=args.anchor_radius)
    response = scoped_responses(original, changed, lengths, values["intervals"], args.topk_divisor)
    lower, upper, weight, null = (values[name] for name in ("lower", "upper", "weight", "null"))
    rejected, delta_abs, saturation = raw.new_zeros(()), raw.new_zeros(()), raw.new_zeros(())
    normal = response.sum() * 0
    if args.variant == "integrated":
        base = model.last_base_raw
        base_response = scoped_responses(base[:batch_size], base[batch_size:].reshape_as(changed),
                                         lengths, values["intervals"], args.topk_divisor)
        lower, upper, weight, rejected = feasible_envelopes(base_response, lower, upper, weight, null,
                                                           model.radii)
        normal = normal_edit_loss(outputs[1][batch_size:].reshape(batch_size, queries, steps),
                                  outputs[2][batch_size:].reshape(batch_size, queries, steps, -1),
                                  lengths, values["normal_weight"])
        valid = torch.arange(steps, device=device)[None] < combined_lengths[:, None]
        delta = model.last_correction.detach()[valid]
        delta_abs = delta.abs().mean()
        saturation = (delta.abs() >= .95 * model.radii.clamp_min(1e-12)).float().mean()
    scope_losses, diagnostics = [], {}
    for index, name in enumerate(SCOPES):
        loss, parts = envelope_loss(response[:, :, index], lower[:, :, index], upper[:, :, index],
                                    weight[:, :, index], null[:, :, index])
        scope_losses.append(loss)
        diagnostics.update({f"{name}_{key}": value for key, value in parts.items()})
    weighted_response = args.response_weight * scope_losses[0]
    if args.variant == "integrated":
        weighted_response = (weighted_response + args.local_weight * scope_losses[1] +
                             args.outside_weight * scope_losses[2])
    total = video + args.anchor_weight * trust
    if args.variant == "control-corrected":
        total = total + weighted_response * 0  # identical edited forward and response graph
    else:
        total = total + weighted_response
    if args.variant == "integrated":
        total = total + args.normal_weight * normal
    return {"total": total, "video": video, "binary": binary, "semantic": semantic,
            "anchor": trust, "response": weighted_response, "normal_edit": normal,
            **{f"{name}_loss": loss for name, loss in zip(SCOPES, scope_losses)},
            **diagnostics, "rejected_mass": rejected, "delta_abs": delta_abs,
            "delta_saturation_fraction": saturation}


def gradient_diagnostics(video, response, parameters):
    pairs = [torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
             for loss in (video, response)]
    norms = [sum(float(value.detach().square().sum()) for value in row if value is not None) ** .5
             for row in pairs]
    dot = sum(float((a.detach() * b.detach()).sum()) for a, b in zip(*pairs)
              if a is not None and b is not None)
    return {"video_grad_norm": norms[0], "response_grad_norm": norms[1],
            "gradient_cosine": dot / (norms[0] * norms[1]) if min(norms) > 0 else None}


def rng_state(loaders):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loaders": [loader.generator.get_state() for loader in loaders if loader is not None]}


def restore_rng(value, loaders):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if value["cuda"] is not None:
        if not torch.cuda.is_available() or len(value["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all([state.cpu() for state in value["cuda"]])
    active = [loader for loader in loaders if loader is not None]
    if len(active) != len(value["loaders"]):
        raise ValueError("Resume loader count mismatch")
    for loader, state in zip(active, value["loaders"]):
        loader.generator.set_state(state.cpu())


@torch.no_grad()
def initialization_probe(model, loaders, prompt, device, args):
    # Compare deterministic inference, not stochastic training dropout, to the cache.
    states = rng_state(loaders)
    maximum = 0.
    try:
        with fresh_evaluation(model):
            model.eval()
            for loader in loaders:
                if loader is None:
                    continue
                probe = DataLoader(loader.dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=0)
                for index, batch in enumerate(probe):
                    length = batch["length"].to(device)
                    raw = flatten_decisions(model(batch["features"].to(device), None, prompt, length, False))
                    anchor = batch["anchor"].to(device)
                    valid = torch.arange(raw.shape[1], device=device)[None] < length[:, None]
                    error = float((raw[valid] - anchor[valid]).abs().max())
                    maximum = max(maximum, error)
                    if not torch.allclose(raw[valid], anchor[valid], atol=args.init_atol, rtol=1e-5):
                        raise ValueError(f"Anchor/cache initialization mismatch: max_abs={error:.7g}")
                    if index + 1 >= args.init_probe_batches:
                        break
    finally:
        restore_rng(states, loaders)
    return {"max_abs_error": maximum, "atol": args.init_atol, "rtol": 1e-5,
            "batches_per_subset": args.init_probe_batches, "mode": "eval"}


def source_signature(root):
    repo = Path(__file__).resolve().parents[2]
    files = list((repo / "dsanet_repair/revision").glob("*.py"))
    files += [repo / f"dsanet_repair/{name}.py" for name in
              ("numerics", "official_losses", "data", "train_full", "train_v2", "decision_constraints", "adapter")]
    result = {str(path.relative_to(repo)): file_hash(path) for path in sorted(files)}
    for path in sorted((Path(root) / "src").rglob("*.py")):
        result["upstream/" + str(path.relative_to(root))] = file_hash(path)
    return result


def run_identity(args, metadata, source):
    # Output relocation/logging changes are harmless; model/data/optimizer changes aren't.
    ignored = {"output_dir", "resume", "log_every", "diagnose_every", "device", "workers",
               "dsanet_root", "init_checkpoint", "response_cache"}
    config = {name: value for name, value in vars(args).items() if name not in ignored}
    value = {"config": config, "cache": metadata["identity"], "anchor": metadata["anchor_sha256"],
             "source": source}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def delta_state(model, names):
    keys = list(names) + (["radii"] if isinstance(model, NativeCorrection) else [])
    state = model.state_dict()
    return {name: state[name].detach().cpu().clone() for name in keys}


def apply_delta(model, payload, names):
    expected = set(names) | ({"radii"} if isinstance(model, NativeCorrection) else set())
    if set(payload["delta"]) != expected or payload["trainable_names"] != list(names):
        raise ValueError("Checkpoint delta does not exactly match the selected architecture")
    if isinstance(model, NativeCorrection) and not torch.equal(payload["delta"]["radii"].cpu(), model.radii.cpu()):
        raise ValueError("Checkpoint correction budgets disagree with configuration")
    result = model.load_state_dict(payload["delta"], strict=False)
    if result.unexpected_keys or set(result.missing_keys) != set(model.state_dict()) - expected:
        raise ValueError("Incomplete or unexpected delta state")
    clear_text_cache(model)


def save_checkpoint(path, model, names, optimizer, scheduler, epoch, step, args, metadata,
                    identity, loaders, probe, initial, source, training_seconds):
    save_atomic({"format": FORMAT, "dataset": args.dataset, "variant": args.variant,
                 "seed": args.seed, "epoch": epoch, "step": step, "identity": identity,
                 "anchor_sha256": metadata["anchor_sha256"], "cache_identity": metadata["identity"],
                 "config": vars(args), "trainable_names": names, "delta": delta_state(model, names),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "rng": rng_state(loaders), "initialization": probe, "initial": initial,
                 "source": source, "training_seconds": training_seconds}, path)


def train(args):
    if args.lr is None:
        args.lr = 1e-4 if args.variant == "integrated" else 1e-5
    positive = (args.epochs, args.batch_size, args.lr, args.clip_grad, args.topk_divisor,
                args.log_every, args.diagnose_every, args.init_probe_batches, args.init_atol)
    if min(positive) <= 0 or args.workers < 0 or min(args.anchor_weight, args.anchor_radius,
             args.response_weight, args.local_weight, args.outside_weight, args.normal_weight) < 0:
        raise ValueError("Invalid training options")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Nonempty output directory; choose a new one or explicitly --resume")
    if args.resume and not (output / "last.pt").is_file():
        raise FileNotFoundError("Resume requires an epoch-boundary last.pt")
    setup_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    root, cfg, test_module, model_class, data_module, tools, _ = official_runtime(args.dsanet_root, args.dataset)
    resolve_cfg(cfg, root)
    cfg.seed = args.seed
    labels = UCF_LABEL_MAP if args.dataset == "ucf" else XD_LABEL_MAP
    prompt = tools.get_prompt_text(labels)
    primary, secondary, steps_per_epoch = make_loaders(cfg, args, labels, device)
    loaders = (primary, secondary)
    metadata = primary.dataset.metadata
    if (metadata["dataset"] != args.dataset or int(metadata["seed"]) != args.seed or
            metadata["numerical_mode"] != "stable" or metadata["scopes"] != list(SCOPES) or
            metadata["anchor_sha256"] != file_hash(args.init_checkpoint) or
            metadata.get("measurement_source") != measurement_signature() or
            metadata.get("upstream_source") != upstream_signature(root) or
            metadata["generation_config"]["topk_divisor"] != args.topk_divisor):
        raise ValueError("Cache/anchor/response-operator mismatch")
    if secondary is not None and secondary.dataset.metadata != metadata:
        raise ValueError("Balanced loaders must share identical cache metadata")
    if not any(row["supported_queries"] for row in primary.dataset.index["rows"].values()):
        raise ValueError("No supported response queries in the complete cache; do not call this a full method run")
    source = source_signature(root)
    environment = {"python": sys.version.split()[0], "torch": str(torch.__version__),
                   "numpy": np.__version__, "cuda": torch.version.cuda, "device": str(device),
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"}
    identity = run_identity(args, metadata, source)
    backbone = make_backbone(model_class, cfg, device, "stable").to(device)
    backbone.load_state_dict(load_initial_state(args.init_checkpoint), strict=True)
    model, selected = model_for_variant(backbone, args)
    model.to(device)
    names, parameters = [name for name, _ in selected], [value for _, value in selected]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch,
                                                          eta_min=args.lr * .1)
    start_epoch, step, previous_seconds, initial = 0, 0, 0., None
    probe = None
    if args.resume:
        payload = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT or payload.get("identity") != identity:
            raise ValueError("Cannot resume: code/config/cache/anchor identity changed")
        apply_delta(model, payload, names)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch, step = payload["epoch"], payload["step"]
        previous_seconds = payload["training_seconds"]
        probe, initial = payload["initialization"], payload["initial"]
        restore_rng(payload["rng"], loaders)
    else:
        probe = initialization_probe(model, loaders, prompt, device, args)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        save_atomic({"identity": identity, "config": vars(args), "cache": metadata,
                     "source": source, "initialization": probe, "environment": environment}, output / "run.json")
        if args.evaluate_initial:
            state = rng_state(loaders)
            initial = evaluate(model, args.dataset, cfg, labels, test_module, data_module, tools, device,
                               output / "initial")
            restore_rng(state, loaders)
    model.train()
    clear_text_cache(model)
    started = time.monotonic()
    invocation = time.time_ns()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, args.epochs):
        for batch_index, batch in enumerate(epoch_batches(primary, secondary)):
            losses = batch_losses(model, batch, prompt, cfg, device, args)
            if not all(torch.isfinite(value).all() for value in losses.values()):
                raise FloatingPointError(f"Non-finite revision loss/diagnostic at {epoch + 1}:{batch_index + 1}")
            diagnostic = gradient_diagnostics(losses["video"], losses["response"], parameters) if (
                step % args.diagnose_every == 0) else {}
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            record = {"phase": "train", "invocation": invocation, "variant": args.variant,
                      "epoch": epoch + 1, "batch": batch_index + 1, "step": step,
                      **{name: float(value.detach()) for name, value in losses.items()},
                      **diagnostic, "gradient_norm": float(gradient), "lr": optimizer.param_groups[0]["lr"]}
            # An interrupted epoch is rerun from last.pt. Invocation tags disambiguate
            # its provisional log lines; only last.pt commits an epoch's updates.
            with (output / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if batch_index % args.log_every == 0:
                print(json.dumps(record), flush=True)
        save_checkpoint(output / "last.pt", model, names, optimizer, scheduler, epoch + 1, step,
                        args, metadata, identity, loaders, probe, initial, source,
                        previous_seconds + time.monotonic() - started)
    training_seconds = previous_seconds + time.monotonic() - started
    final = None if args.skip_evaluation else evaluate(model, args.dataset, cfg, labels, test_module,
                                                       data_module, tools, device, output / "final")
    summary = {"variant": args.variant, "dataset": args.dataset, "seed": args.seed,
               "identity": identity, "anchor_sha256": metadata["anchor_sha256"],
               "cache_identity": metadata["identity"], "epochs": args.epochs, "steps": step,
               "checkpoint_policy": "fixed_final_epoch_not_test_best", "numerical_mode": "stable",
               "trainable_parameters": sum(value.numel() for value in parameters), "trainable_names": names,
               "initialization": probe, "initial": initial, "final": final,
               "training_seconds": training_seconds, "cache_build_audit": primary.dataset.index.get("audit"),
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "evaluation_skipped": args.skip_evaluation,
               "environment": environment,
               "data_schedule": "balanced min-loader epochs" if args.dataset == "ucf" else "all CSV rows per epoch"}
    save_atomic(summary, output / "summary.json")
    print(json.dumps(summary), flush=True)
    return summary


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", choices=("ucf", "xd"), required=True)
    value.add_argument("--variant", choices=VARIANTS, required=True)
    value.add_argument("--dsanet-root", required=True)
    value.add_argument("--response-cache", required=True)
    value.add_argument("--init-checkpoint", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--seed", type=int, default=234)
    value.add_argument("--epochs", type=int, default=3)
    value.add_argument("--batch-size", type=int, default=8)
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--lr", type=float, help="Default: tails 1e-5; integrated head 1e-4")
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--response-weight", type=float, default=.5)
    value.add_argument("--local-weight", type=float, default=.25)
    value.add_argument("--outside-weight", type=float, default=.1)
    value.add_argument("--normal-weight", type=float, default=.1)
    value.add_argument("--anchor-weight", type=float, default=.1)
    value.add_argument("--anchor-radius", type=float, default=.25)
    value.add_argument("--binary-radius", type=float, default=.5)
    value.add_argument("--semantic-radius", type=float, default=2.)
    value.add_argument("--hidden", type=int, default=128)
    value.add_argument("--prototype-temperature", type=float, default=.1)
    value.add_argument("--clip-grad", type=float, default=1.)
    value.add_argument("--topk-divisor", type=int, default=16)
    value.add_argument("--init-atol", type=float, default=2e-4)
    value.add_argument("--init-probe-batches", type=int, default=2)
    value.add_argument("--diagnose-every", type=int, default=50)
    value.add_argument("--log-every", type=int, default=20)
    value.add_argument("--device", default="auto")
    value.add_argument("--resume", action="store_true", help="Resume committed epoch with unchanged configuration")
    value.add_argument("--evaluate-initial", action="store_true")
    value.add_argument("--skip-evaluation", action="store_true", help="Local engineering tests only, not benchmark evidence")
    return value


if __name__ == "__main__":
    train(parser().parse_args())
