from dataclasses import asdict
import torch
from torch import nn
from vadcore.types import Prediction
from vadcore.training import pack_backbone, make_backbone, atomic_save, load_checkpoint
from .reference import NormalReference
from .repair import SelectiveRepair, RepairConfig


class EvidenceFusion(nn.Module):
    def __init__(self, targets, hidden=32):
        super().__init__()
        self.targets, self.hidden = targets, hidden
        self.target_embedding = nn.Embedding(targets, 8)
        self.net = nn.Sequential(nn.Linear(14, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.temporal = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base, evidence, gate):
        if evidence.shape != (*base.shape, 6) or gate.shape != base.shape or base.shape[1] != self.targets:
            raise ValueError("Fusion tensor shape mismatch.")
        emb = self.target_embedding(torch.arange(self.targets, device=base.device))
        x = torch.cat([evidence, emb[None].expand(len(base), -1, -1)], -1)
        hidden = self.net[:2](x)
        hidden = hidden + self.temporal(hidden.permute(1, 2, 0)).permute(2, 0, 1)
        return base + gate * self.net[-1](hidden).squeeze(-1)


class RepairSystem:
    def __init__(self, engines, fusion):
        if not engines:
            raise ValueError("Repair system needs at least one fitted fold.")
        self.engines, self.fusion = engines, fusion

    def to(self, device):
        self.fusion.to(device)
        for engine in self.engines:
            engine.probe.to(device)
        return self

    @torch.no_grad()
    def logits(self, sample):
        device = next(self.fusion.parameters()).device
        self.fusion.eval()
        logits, info = [], []
        for engine in self.engines:
            e = engine.extract(sample.inference_copy())
            logits.append(self.fusion(e.base.to(device), e.features.to(device), e.gate.to(device)).cpu())
            info.append(e.diagnostics)
        return torch.stack(logits).mean(0), info

    def predict(self, sample):
        logits, info = self.logits(sample)
        prob = logits.sigmoid()
        return Prediction(prob[:, 0], prob[:, 1:], sample.edges.cpu(),
                          {"folds": info, "ensemble": "mean_logits", "used_test_labels": False}).validate()

    def package(self):
        return {
            "format": "selective-repair-v1",
            "engines": [{"probe": pack_backbone(e.probe), "reference": e.reference.package(),
                         "repair_config": asdict(e.config), "trained_source_ids": sorted(e.trained_source_ids)}
                        for e in self.engines],
            "fusion": {"targets": self.fusion.targets, "hidden": self.fusion.hidden,
                       "state": {k: v.detach().cpu().clone() for k, v in self.fusion.state_dict().items()}},
        }

    def save(self, path):
        atomic_save(self.package(), path)

    @classmethod
    def from_package(cls, item, device="cpu"):
        if item.get("format") != "selective-repair-v1":
            raise ValueError("Wrong repair checkpoint.")
        engines = [
            SelectiveRepair(make_backbone(e["probe"]), NormalReference.from_package(e["reference"]),
                            RepairConfig(**e["repair_config"]), e["trained_source_ids"])
            for e in item["engines"]
        ]
        fusion = EvidenceFusion(item["fusion"]["targets"], item["fusion"]["hidden"])
        fusion.load_state_dict(item["fusion"]["state"])
        return cls(engines, fusion).to(device)

    @classmethod
    def load(cls, path, device="cpu"):
        return cls.from_package(load_checkpoint(path), device)
