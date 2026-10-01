"""Per-video category evidence with selective writes, loss and reacquisition.

This is independent of MVLM's image/box tracker and uses no test labels or
cross-video persistent bank. One state exists per abnormal category.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ClassReacquisitionMemory(nn.Module):
    def __init__(self, width, hidden, acquire=.15, max_gap=3, write_rate=.5, decay=.85):
        super().__init__()
        if not 0 < acquire < 1 or max_gap < 1 or not 0 < write_rate <= 1 or not 0 < decay <= 1:
            raise ValueError("Invalid class memory settings")
        self.acquire, self.max_gap = acquire, max_gap
        self.write_rate, self.decay = write_rate, decay
        self.diagnostic_mode = "normal"
        self.project = nn.Linear(width, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, hidden, features, context, base, valid):
        text, binary, semantic = base[:3]
        batch, steps, width = features.shape
        classes = semantic.shape[-1] - 1
        if classes < 1 or text.shape != (classes + 1, width):
            raise ValueError("Class memory needs aligned frozen class text/features")
        with torch.no_grad():
            residual = torch.where(valid[..., None], features - context, 0.)
            text = F.normalize(text[1:], dim=-1)
            alignment = F.normalize(residual, dim=-1) @ text.T
            binary = torch.where(valid[..., None], binary, 0.)
            semantic = torch.where(valid[..., None], semantic, 0.)
            margin = semantic[..., 1:] - semantic[..., :1]
            confidence = (binary.sigmoid() * margin.sigmoid() *
                          (.5 + .5 * alignment.clamp(-1, 1)))
            confidence = torch.where(valid[..., None], confidence, 0.)
        projected = self.project(residual)
        state = hidden.new_zeros(batch, classes, hidden.shape[-1])
        active = torch.zeros(batch, classes, dtype=torch.bool, device=hidden.device)
        seen = active.clone()
        stale = torch.zeros(batch, classes, dtype=torch.long, device=hidden.device)
        writes = confidence.new_zeros(())
        reacquisitions = confidence.new_zeros(())
        active_mass = confidence.new_zeros(())
        additions = []
        for time in range(steps):
            valid_now = valid[:, time, None]
            good = valid_now & (confidence[:, time] >= self.acquire)
            if self.diagnostic_mode == "always_write":
                good = valid_now.expand_as(good)
            elif self.diagnostic_mode == "never_reacquire":
                good = good & (active | ~seen)
            elif self.diagnostic_mode != "normal":
                raise ValueError("Unknown class-memory diagnostic mode")
            was_active = active
            reacquisitions = reacquisitions + (good & ~was_active & seen).sum()
            writes = writes + good.sum()
            stale = torch.where(valid_now, torch.where(good, 0, stale + 1), stale)
            active = torch.where(valid_now, good | (active & (stale < self.max_gap)), active)
            fresh = good & ~was_active
            current = projected[:, time, None].expand(-1, classes, -1)
            updated = torch.where(fresh[..., None], current,
                                  (1 - self.write_rate) * state + self.write_rate * current)
            state = torch.where(good[..., None], updated, self.decay * state)
            state = torch.where(active[..., None], state, 0.)
            seen = seen | good
            weights = torch.where(active, confidence[:, time], 0.)
            read = (weights[..., None] * state).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-8)
            addition = self.output(read)
            additions.append(torch.where(valid_now, addition, 0.))
            active_mass = active_mass + (active & valid_now).sum()
        addition = torch.stack(additions, 1)
        result = torch.where(valid[..., None], hidden + addition, 0.)
        count = (valid.sum() * classes).clamp_min(1)
        return result, {
            "memory_write_fraction": writes / count,
            "memory_active_fraction": active_mass / count,
            "memory_reacquisitions": reacquisitions,
            "memory_output_abs": addition.abs().sum() / (valid.sum().clamp_min(1) * hidden.shape[-1]),
        }
