"""Learnable innovation-gated state estimator for correction logits.

This adapts a state-space primitive, not COGNOS's Gaussian-residual guarantee.
It sees only frozen anchor outputs and the current video's correction sequence.
"""

import torch
from torch import nn
from torch.nn import functional as F


class FixedEWMA(nn.Module):
    """Training-time ordinary smoothing control, with a predetermined scalar gain."""
    def __init__(self, gain=.4):
        super().__init__()
        if not 0 < gain <= 1:
            raise ValueError("Fixed EWMA gain must be in (0,1]")
        self.register_buffer("gain", torch.tensor(float(gain)))

    def forward(self, correction, residual, binary, semantic, valid):
        correction = torch.where(valid[..., None], correction, 0.)
        forward = InnovationFilter._fixed_ewma(correction, valid, self.gain)
        backward = InnovationFilter._fixed_ewma(correction.flip(1), valid.flip(1), self.gain)
        result = torch.where(valid[..., None], .5 * (forward + backward.flip(1)), 0.)
        return result, {"filter_gain": self.gain,
                        "filter_change_abs": (result - correction).abs().sum() / (
                            valid.sum().clamp_min(1) * correction.shape[-1])}


class InnovationFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_process = nn.Parameter(torch.tensor(-2.))
        self.boundary_scale = nn.Parameter(torch.tensor(0.))
        self.base_observation = nn.Parameter(torch.tensor(-1.))
        self.uncertainty_scale = nn.Parameter(torch.tensor(0.))
        self.diagnostic_fixed_gain = False

    @staticmethod
    def _scan(observation, process, noise, valid):
        batch, steps, channels = observation.shape
        state = observation.new_zeros(batch, channels)
        variance = observation.new_zeros(batch, 1)
        seen = torch.zeros(batch, 1, dtype=torch.bool, device=observation.device)
        outputs, gains = [], []
        for time in range(steps):
            active = valid[:, time, None]
            predicted_variance = variance + process[:, time, None]
            gain = predicted_variance / (predicted_variance + noise[:, time, None]).clamp_min(1e-8)
            candidate = state + gain * (observation[:, time] - state)
            state = torch.where(active, torch.where(seen, candidate, observation[:, time]), state)
            variance = torch.where(active, torch.where(seen, (1 - gain) * predicted_variance,
                                                       noise[:, time, None]), variance)
            seen = seen | active
            outputs.append(torch.where(active, state, 0.))
            gains.append(torch.where(active, gain, 0.))
        return torch.stack(outputs, 1), torch.stack(gains, 1)

    @staticmethod
    def _fixed_ewma(observation, valid, gain):
        state = observation.new_zeros(observation.shape[0], observation.shape[-1])
        seen = torch.zeros(observation.shape[0], 1, dtype=torch.bool, device=observation.device)
        outputs = []
        for time in range(observation.shape[1]):
            active = valid[:, time, None]
            candidate = state + gain * (observation[:, time] - state)
            state = torch.where(active, torch.where(seen, candidate, observation[:, time]), state)
            seen = seen | active
            outputs.append(torch.where(active, state, 0.))
        return torch.stack(outputs, 1)

    def forward(self, correction, residual, binary, semantic, valid):
        if correction.ndim != 3 or valid.shape != correction.shape[:2]:
            raise ValueError("InnovationFilter needs [B,T,C] correction and [B,T] mask")
        if correction.shape[1] < 1:
            raise ValueError("Empty correction sequence")
        correction = torch.where(valid[..., None], correction, 0.)
        with torch.no_grad():
            residual = torch.where(valid[..., None], residual, 0.)
            binary = torch.where(valid[..., None], binary, 0.).squeeze(-1)
            semantic = torch.where(valid[..., None], semantic, 0.)
            probability = binary.sigmoid()
            category = semantic[..., 1:].softmax(-1)
            if correction.shape[1] > 1:
                edge = (.5 * (residual[:, 1:] - residual[:, :-1]).square().sum(-1)
                        + (probability[:, 1:] - probability[:, :-1]).abs()
                        + .5 * (category[:, 1:] - category[:, :-1]).abs().sum(-1))
                edge = torch.where(valid[:, 1:] & valid[:, :-1], edge, 0.)
                left = F.pad(edge, (1, 0))
                right = F.pad(edge, (0, 1))
                boundary = torch.maximum(left, right)
            else:
                boundary = correction.new_zeros(valid.shape)
            uncertainty = 4 * probability * (1 - probability) + (1 - category.amax(-1))
        process = F.softplus(self.base_process + F.softplus(self.boundary_scale) * boundary)
        noise = F.softplus(self.base_observation + F.softplus(self.uncertainty_scale) * uncertainty)
        forward, forward_gain = self._scan(correction, process, noise, valid)
        backward, backward_gain = self._scan(correction.flip(1), process.flip(1),
                                             noise.flip(1), valid.flip(1))
        result = torch.where(valid[..., None], .5 * (forward + backward.flip(1)), 0.)
        count = valid.sum().clamp_min(1)
        gain_curve = .5 * (forward_gain + backward_gain.flip(1))
        video_gain = (gain_curve * valid[..., None]).sum(1, keepdim=True) / valid.sum(1).clamp_min(1)[:, None, None]
        temporal_variance = ((gain_curve - video_gain).square() * valid[..., None]).sum() / count
        if self.diagnostic_fixed_gain:
            # Test-only control: each video's mean adaptive gain is retained,
            # but its edge-dependent temporal schedule is removed. No labels.
            gain = ((forward_gain + backward_gain.flip(1)) * valid[..., None]).sum(1) / (
                2 * valid.sum(1, keepdim=True).clamp_min(1))
            fixed_forward = self._fixed_ewma(correction, valid, gain)
            fixed_backward = self._fixed_ewma(correction.flip(1), valid.flip(1), gain)
            result = torch.where(valid[..., None], .5 * (fixed_forward + fixed_backward.flip(1)), 0.)
        diagnostics = {
            "filter_gain": ((forward_gain + backward_gain.flip(1)) * valid[..., None]).sum() / (2 * count),
            "filter_boundary_process": (process * boundary * valid).sum() / (boundary * valid).sum().clamp_min(1),
            "filter_change_abs": (result - correction).abs().sum() / (count * correction.shape[-1]),
            "filter_gain_temporal_std": temporal_variance.sqrt(),
            "filter_process_parameter": self.base_process.detach(),
            "filter_observation_parameter": self.base_observation.detach(),
            "filter_boundary_parameter": self.boundary_scale.detach(),
            "filter_uncertainty_parameter": self.uncertainty_scale.detach(),
        }
        return result, diagnostics
