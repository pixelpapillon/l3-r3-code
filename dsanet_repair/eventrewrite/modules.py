"""Three research operators; no CLIP, teacher edits, or response-cache dependency.

These are hypotheses, not validated improvements. Fixed interval geometry removes
the particular collapse route observed in EventBridge; it does not guarantee
correct event selection. Signed potentials are latent evidence interactions,
NOT measured causal effects of deleting real video events.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class IntervalBasis(nn.Module):
    """Integrate a fixed, multiscale dictionary on the actual feature-bin edges."""

    def __init__(self, centers=8):
        super().__init__()
        if centers < 2:
            raise ValueError("At least two centers are required")
        bounds = []
        for width in (1 / 16, 1 / 4, 1 / 2):
            for start in torch.linspace(0, 1 - width, centers).tolist():
                bounds.append((start, start + width))
        bounds.append((0., 1.))
        self.register_buffer("bounds", torch.tensor(bounds))
        intervals = self.bounds
        overlap = (torch.minimum(intervals[:, None, 1], intervals[None, :, 1]) -
                   torch.maximum(intervals[:, None, 0], intervals[None, :, 0])).clamp_min(0)
        pair = torch.triu(overlap <= 1e-7, diagonal=1).nonzero()
        self.register_buffer("pairs", pair)

    def forward(self, edges):
        # edges [B,T+1], normalized native time. Padded bins have zero width.
        left, right = edges[:, None, :-1], edges[:, None, 1:]
        start, end = self.bounds.T
        center, width = (start + end) / 2, end - start

        def integrate(a, b):
            return (torch.minimum(right, b[None, :, None]) -
                    torch.maximum(left, a[None, :, None])).clamp_min(0)

        mass = integrate(start, end)
        outer = integrate((start - width / 2).clamp_min(0), (end + width / 2).clamp_max(1))
        weights = [mass, integrate(center - width / 4, center + width / 4),
                   integrate(start, center), integrate(center, end), (outer - mass).clamp_min(0)]
        weights = [x / x.sum(-1, keepdim=True).clamp_min(1e-8) for x in weights]
        widths = (right - left).squeeze(1)
        support = mass / widths[:, None].clamp_min(1e-8)
        return weights, support, widths


class MorphologyEvidence(nn.Module):
    """M1: shared interval mean plus core, phase, context and variability contrasts.

    The output is class-specific evidence rendered on interval supports, not a
    separately predicted geometry/moment target. Geometry is deliberately fixed.
    """

    def __init__(self, width, hidden, classes):
        super().__init__()
        self.contrast = nn.Sequential(nn.LayerNorm(4 * width), nn.Linear(4 * width, hidden),
                                      nn.GELU(), nn.Linear(hidden, width))
        self.norm = nn.LayerNorm(width)
        self.unary = nn.Linear(width, classes)
        nn.init.zeros_(self.unary.weight)
        nn.init.zeros_(self.unary.bias)

    def forward(self, h, weights, support):
        envelope, core, early, late, context = [w @ h for w in weights]
        variance = (weights[0] @ h.square() - envelope.square()).clamp_min(0)
        # A whole-video interval has no outer context: use zero contrast, not
        # a fictitious zero-valued background feature.
        has_context = (weights[4].sum(-1, keepdim=True) > 0).to(h)
        descriptor = torch.cat((core - envelope, late - early,
                                (envelope - context) * has_context, variance), dim=-1)
        tokens = self.norm(envelope + self.contrast(descriptor))
        unary = self.unary(tokens)
        render = support.transpose(1, 2)
        render = render / render.sum(-1, keepdim=True).clamp_min(1e-8)
        return tokens, render @ unary, unary


class SignedComposition(nn.Module):
    """M2: symmetric signed low-rank pair potentials, with NO graph smoothing.

    Only disjoint envelope pairs participate. For a fixed token set, retaining
    interval i with a_i multiplies unary terms by a_i and pair (i,j) by a_i*a_j.
    Therefore the latent energy factorial contrast isolates the (i,j) term.
    It is not a claim about causal interactions in the input video.
    """

    def __init__(self, width, classes, rank=8):
        super().__init__()
        self.classes, self.rank = classes, rank
        self.norm = nn.LayerNorm(width)
        self.left = nn.Linear(width, classes * rank, bias=False)
        self.right = nn.Linear(width, classes * rank, bias=False)
        self.coefficient = nn.Parameter(torch.zeros(classes, rank))

    def forward(self, tokens, support, pairs, retention=None):
        batch, count, _ = tokens.shape
        h = self.norm(tokens)
        l = self.left(h).reshape(batch, count, self.classes, self.rank).tanh()
        r = self.right(h).reshape(batch, count, self.classes, self.rank).tanh()
        i, j = pairs.T
        potential = (((l[:, i] * r[:, j] + l[:, j] * r[:, i]) / 2) * self.coefficient).sum(-1)
        potential = potential / math.sqrt(self.rank)
        # The render denominator is defined BEFORE an intervention. Renormalizing
        # after removal would create spurious factorial interactions.
        render = ((support[:, i] + support[:, j]) / 2).transpose(1, 2)
        render = render / render.sum(-1, keepdim=True).clamp_min(1e-8)
        if retention is not None:
            potential = potential * (retention[:, i] * retention[:, j])[..., None]
        return render @ potential, potential, render


def hierarchical_log_probability(binary, semantic, temperature=1.):
    """Exactly the nondegenerate hierarchical DSANet probability, in log space."""
    binary = binary / temperature
    return torch.cat((F.logsigmoid(-binary),
                      F.logsigmoid(binary) + F.log_softmax(semantic[..., 1:] / temperature, -1)), -1)


def reconstruct_probability(log_prior, evidence, widths, strength=.5, iterations=16):
    r"""M3: soft-marginal KL reconstruction with hard per-frame normalization.

    min_Q sum_t w_t KL(Q_t || P_t) - <wQ,E>
          + strength * KL(sum_t w_t Q_t || sum_t w_t P_t),  Q_t in simplex.

    Generalized matrix scaling is unrolled, including gradients through the
    adaptive prior and its marginal. No fixed anomaly mass, no frozen teacher,
    no class-presence label at inference. A zero evidence field is an identity.
    The finite-iteration solver is approximate; report its fixed-point residual.
    """
    if strength < 0 or iterations < 1:
        raise ValueError("Invalid reconstruction settings")
    w = widths / widths.sum(-1, keepdim=True).clamp_min(1e-12)
    log_w = w.clamp_min(1e-30).log().masked_fill(w == 0, -torch.inf)
    marginal = torch.logsumexp(log_prior + log_w[..., None], dim=1)
    kernel = log_prior + evidence
    dual = torch.zeros_like(marginal)
    power = strength / (1 + strength)

    def update(v):
        rows = -torch.logsumexp(kernel + v[:, None], -1)
        columns = torch.logsumexp(kernel + rows[..., None] + log_w[..., None], dim=1)
        return power * (marginal - columns)

    for _ in range(iterations):
        dual = update(dual)
    residual = (update(dual) - dual).abs().amax()
    log_q = F.log_softmax(kernel + dual[:, None], -1)
    mean = (w[..., None] * log_q.exp()).sum(1)
    shift = (mean - marginal.exp()).abs().sum(-1).mean()
    return log_q, {"solver_residual": residual, "marginal_l1": shift}


def official_logits(log_q, temperature):
    """Encode joint Q as the two logits expected by the unchanged evaluator."""
    binary = temperature * (torch.logsumexp(log_q[..., 1:], -1, keepdim=True) - log_q[..., :1])
    return binary, temperature * log_q
