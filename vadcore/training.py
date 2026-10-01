from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import random
import importlib
import json
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from .backbone import BackboneConfig, FeatureVAD, bag_logits
from .types import validate_training, check_disjoint


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    accumulation: int = 8
    clip_grad: float = 5.0
    seed: int = 42
    device: str = "cuda"
    amp: bool = True

    def __post_init__(self):
        if self.epochs < 1 or self.accumulation < 1 or self.lr <= 0:
            raise ValueError("epochs, accumulation and lr must be positive.")


def setup(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU tests require device=cpu.")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.device(device)


def weak_loss(logits, sample):
    labels = sample.labels.to(logits.device)
    if sample.binary_label == 0:
        labels = torch.zeros_like(labels)
    y = torch.cat([logits.new_tensor([sample.binary_label]), labels])
    known = y >= 0
    return F.binary_cross_entropy_with_logits(logits[known], y[known].float())


def amp_context(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=enabled and device.type == "cuda")


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def experiment_signature(samples, validation, config, fixed_text=None):
    """Fingerprint ordered tensors, split membership and optimizer schedule."""
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for name, split in (("train", samples), ("validation", validation)):
        digest.update(name.encode())
        for sample in split:
            digest.update(json.dumps([sample.source_id, sample.binary_label]).encode())
            for tensor in (sample.features, sample.edges, sample.labels):
                x = tensor.detach().cpu().contiguous()
                digest.update(str((tuple(x.shape), x.dtype)).encode())
                digest.update(x.view(torch.uint8).numpy().tobytes())
    if fixed_text is not None:
        x = fixed_text.detach().cpu().contiguous()
        digest.update(str((tuple(x.shape), x.dtype)).encode())
        digest.update(x.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def pack_backbone(model):
    return {"config": asdict(model.config),
            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


def make_backbone(checkpoint):
    cfg = BackboneConfig(**checkpoint["config"])
    fixed = checkpoint["state"].get("fixed_text")
    model = FeatureVAD(cfg, fixed if fixed is not None and fixed.numel() else None)
    model.load_state_dict(checkpoint["state"])
    return model


def train_probe(samples, config, train_config, validation=(), initial=None):
    validate_training(samples, config.classes)
    if validation:
        validate_training(validation, config.classes)
    check_disjoint(samples, validation)
    device = setup(train_config.seed, train_config.device)
    model = (copy.deepcopy(initial) if initial is not None else FeatureVAD(config)).to(device)
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_config.epochs)
    best, best_loss, history = None, float("inf"), []
    for epoch in range(train_config.epochs):
        model.train()
        indices = list(range(len(samples)))
        random.Random(train_config.seed + epoch).shuffle(indices)
        total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, idx in enumerate(indices):
            sample = samples[idx]
            with amp_context(device, train_config.amp):
                out = model(sample.features.to(device))
                loss = weak_loss(bag_logits({k: v[0] for k, v in out.items()}, config.topk_divisor).float(), sample)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite probe loss.")
            start = (step // train_config.accumulation) * train_config.accumulation
            size = min(train_config.accumulation, len(indices) - start)
            (loss / size).backward()
            total += float(loss.detach())
            if (step + 1) % train_config.accumulation == 0 or step + 1 == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            vals = []
            for v in validation:
                out = model(v.features.to(device))
                vals.append(float(weak_loss(bag_logits({k: z[0] for k, z in out.items()}, config.topk_divisor), v)))
        score = sum(vals) / len(vals) if vals else total / len(samples)
        history.append({"epoch": epoch + 1, "train_bag_loss": total / len(samples),
                        "validation_bag_loss": score if vals else None})
        if score < best_loss:
            best_loss, best = score, copy.deepcopy(model.state_dict())
        print(json.dumps({"phase": "probe", **history[-1]}), flush=True)
    model.load_state_dict(best)
    model.eval()
    model.requires_grad_(False)
    return model, history


def load_provider(spec, config):
    module, name = spec.split(":", 1)
    data = getattr(importlib.import_module(module), name)(config)
    if "train" not in data:
        raise ValueError("Provider must return a train sequence.")
    return data
