"""Same E-study sampling/holdout. R6 additionally checks a separate supplement."""

import torch
from torch.utils.data import DataLoader
from dsanet_repair.eventstudy.data import EventDataset
from .joint import JointDataset
from . import CONFIGS


def make_loaders(cfg, args, labels, device):
    loaders = []
    for offset, subset in enumerate(("normal", "anomaly") if args.dataset == "ucf" else (None,)):
        cls = JointDataset if CONFIGS[args.variant].interaction else EventDataset
        extra = {"joint_cache": args.joint_cache} if cls is JointDataset else {}
        data = cls(cfg.train_list, args.response_cache, labels, cfg.visual_length, subset=subset,
                   holdout_fraction=args.response_holdout_fraction, holdout_salt=args.response_holdout_salt, **extra)
        loaders.append(DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=device.type == "cuda", drop_last=args.dataset == "ucf",
                                  generator=torch.Generator().manual_seed(args.seed + offset)))
    if min(map(len, loaders)) < 1:
        raise ValueError("Empty training loader")
    return loaders[0], loaders[1] if len(loaders) > 1 else None, min(map(len, loaders))
