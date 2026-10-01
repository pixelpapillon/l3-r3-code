"""Modified from VadCLIP model.py/layers.py (Apache-2.0).

Source: nwpu-zxr/VadCLIP@c41067f07d252efcda18008bea367886070c33b0.
Changed: device-neutral graphs, actual lengths, cached text or learned queries,
multilabel margins. No token prompt tuning; not full-paper VadCLIP replication.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class BackboneConfig:
    input_dim: int = 512
    width: int = 512
    classes: int = 13
    heads: int = 8
    layers: int = 1
    window: int = 8
    positions: int = 256
    dropout: float = 0.1
    topk_divisor: int = 16
    max_steps: int = 1024


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


def mlp(w):
    return nn.Sequential(OrderedDict([
        ("c_fc", nn.Linear(w, w * 4)), ("gelu", QuickGELU()),
        ("c_proj", nn.Linear(w * 4, w)),
    ]))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ln_1, self.ln_2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.mlp = mlp(width)

    def forward(self, x, mask):
        q = self.ln_1(x)
        x = x + self.attn(q, q, q, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class GraphConvolution(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, m))
        nn.init.xavier_uniform_(self.weight)
        self.same = n == m
        self.residual = nn.Identity() if self.same else nn.Conv1d(n, m, 5, padding=2)

    def forward(self, x, adj):
        res = self.residual(x) if self.same else self.residual(x.transpose(1, 2)).transpose(1, 2)
        return adj @ (x @ self.weight) + res


class FeatureVAD(nn.Module):
    def __init__(self, config, text_embeddings=None):
        super().__init__()
        self.config = config
        w = config.width
        if w % 2 or w % config.heads or min(config.classes, config.window, config.positions, config.topk_divisor) < 1:
            raise ValueError("Invalid backbone dimensions/window/top-k.")
        self.input_proj = nn.Identity() if config.input_dim == w else nn.Linear(config.input_dim, w)
        self.frame_position_embeddings = nn.Embedding(config.positions, w)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)
        self.temporal = nn.ModuleList([
            ResidualAttentionBlock(w, config.heads, config.dropout) for _ in range(config.layers)
        ])
        self.gc1, self.gc2 = GraphConvolution(w, w // 2), GraphConvolution(w // 2, w // 2)
        self.gc3, self.gc4 = GraphConvolution(w, w // 2), GraphConvolution(w // 2, w // 2)
        self.linear, self.gelu = nn.Linear(w, w), QuickGELU()
        self.mlp1, self.mlp2 = mlp(w), mlp(w)
        self.classifier = nn.Linear(w, 1)
        if text_embeddings is None:
            self.class_queries = nn.Parameter(torch.randn(config.classes + 1, w) / math.sqrt(w))
            self.register_buffer("fixed_text", torch.empty(0))
        else:
            if text_embeddings.shape != (config.classes + 1, w):
                raise ValueError("Text must be [normal+C, width].")
            self.register_parameter("class_queries", None)
            self.register_buffer("fixed_text", text_embeddings.detach().float().clone())

    def _encode_one(self, x):
        t = len(x)
        if t > self.config.max_steps:
            raise ValueError("Feature length exceeds backbone max_steps; provide a declared temporal grid.")
        x = self.input_proj(x.float())[None]
        pos = self.frame_position_embeddings.weight
        pos = pos[:t] if t <= len(pos) else F.interpolate(
            pos.T[None], size=t, mode="linear", align_corners=False)[0].T
        x = x + pos[None]
        i = torch.arange(t, device=x.device)
        local = (i[:, None] // self.config.window) != (i[None, :] // self.config.window)
        for layer in self.temporal:
            x = layer(x, local)
        n = F.normalize(x, dim=-1, eps=1e-6)
        sim = n @ n.transpose(1, 2)
        sim = torch.where(sim > 0.7, sim, torch.zeros_like(sim)).softmax(-1)
        dist = torch.exp(-(i[:, None] - i[None, :]).abs().float() / math.e)[None]
        a = self.gelu(self.gc2(self.gelu(self.gc1(x, sim)), sim))
        b = self.gelu(self.gc4(self.gelu(self.gc3(x, dist)), dist))
        return self.linear(torch.cat([a, b], -1))[0]

    def forward(self, x, lengths=None):
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or x.shape[-1] != self.config.input_dim:
            raise ValueError("Expected [B,T,D] or [T,D].")
        lengths = [x.shape[1]] * len(x) if lengths is None else [int(n) for n in lengths]
        if len(lengths) != len(x) or any(n < 1 or n > x.shape[1] for n in lengths):
            raise ValueError("Invalid lengths.")
        hs, bs, cs = [], [], []
        query = self.fixed_text if self.class_queries is None else self.class_queries
        for row, n in zip(x, lengths):
            h = self._encode_one(row[:n])
            a = self.classifier(h + self.mlp2(h)).squeeze(-1)
            context = F.normalize((a[:, None] * h).sum(0), dim=-1, eps=1e-6)
            txt = query + context[None]
            txt = txt + self.mlp1(txt)
            sim = F.normalize(h, dim=-1, eps=1e-6) @ F.normalize(txt, dim=-1, eps=1e-6).T / 0.07
            c = sim[:, 1:] - sim[:, :1]
            hs.append(F.pad(h, (0, 0, 0, x.shape[1] - n)))
            bs.append(F.pad(a, (0, x.shape[1] - n)))
            cs.append(F.pad(c, (0, 0, 0, x.shape[1] - n)))
        return {"features": torch.stack(hs), "binary": torch.stack(bs), "classes": torch.stack(cs)}

    def import_upstream_visual(self, state):
        own, mapped, skipped = self.state_dict(), {}, []
        for key, value in state.items():
            new = key.replace("temporal.resblocks.", "temporal.")
            if new in own and value.shape == own[new].shape and not key.startswith("clipmodel"):
                mapped[new] = value
            else:
                skipped.append(key)
        if not mapped:
            raise ValueError("No compatible upstream visual weights.")
        result = self.load_state_dict(mapped, strict=False)
        return {"loaded": sorted(mapped), "skipped": skipped, "missing": result.missing_keys}


def bag_logits(output, divisor=16):
    b, c = output["binary"], output["classes"]
    k = min(len(b), max(1, len(b) // divisor + 1))
    return torch.cat([b.topk(k).values.mean()[None], c.topk(k, dim=0).values.mean(0)])
