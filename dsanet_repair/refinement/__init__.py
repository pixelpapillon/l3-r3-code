"""Result-driven Q1-Q6 study. One model, three optional mechanisms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Configuration:
    ranking: bool = False
    attention: str = "none"
    flow: bool = False
    binary_rank_temperature: float = 0.
    edge_phase: bool = False
    influence: bool = False
    state_filter: bool = False
    class_memory: bool = False
    no_response: bool = False
    fixed_filter: bool = False
    mechanism: str = "none"


CONFIGS = {
    "q1-gauge": Configuration(),
    "q2-evidence-rank": Configuration(ranking=True),
    "q3-cross-attention": Configuration(ranking=True, attention="add"),
    "q4-contrast-attention": Configuration(ranking=True, attention="contrast"),
    "q5-boundary-flow": Configuration(ranking=True, flow=True),
    "q6-full": Configuration(ranking=True, attention="contrast", flow=True),
}
VARIANTS = tuple(CONFIGS)

# Follow-up variants deliberately do not change the archived Q1-Q6 queue.
# Each variant changes one path relative to its named Q control.
FOLLOWUPS = {
    "r1-rank-soft-tail": Configuration(ranking=True, binary_rank_temperature=5.),
    "r2-event-only": Configuration(ranking=True, attention="event_only"),
    "r3-edge-phase": Configuration(ranking=True, flow=True, edge_phase=True),
    "s4-gradient-agreement": Configuration(influence=True),
    "s5-innovation-filter": Configuration(state_filter=True),
    "s6-class-reacquire": Configuration(class_memory=True),
}
FOLLOWUP_VARIANTS = tuple(FOLLOWUPS)
CONVERGENCE = {
    "c1-long-q1": Configuration(),
    "c2-long-agreement": Configuration(influence=True),
    "c3-long-filter": Configuration(state_filter=True),
    "c4-agreement-filter": Configuration(influence=True, state_filter=True),
    "c5-no-response": Configuration(no_response=True),
    "c6-agreement-ewma": Configuration(influence=True, fixed_filter=True),
}
NOVEL = {
    "n7-duration-assembly": Configuration(mechanism="duration"),
    "n8-capacity-transport": Configuration(mechanism="transport"),
    "n9-crossfit-dynamics": Configuration(mechanism="dynamics"),
    "n10-path-signature": Configuration(mechanism="signature"),
    "n11-wavelet-scattering": Configuration(mechanism="wavelet"),
}
ELEVEN_VARIANTS = tuple(CONVERGENCE) + tuple(NOVEL)
ALL_CONFIGS = {**CONFIGS, **FOLLOWUPS, **CONVERGENCE, **NOVEL}


def add_arguments(parser):
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=3)
    parser.add_argument("--flow-temperature", type=float, default=.2)
    parser.add_argument("--rank-weight", type=float, default=.1)
    parser.add_argument("--semantic-rank-weight", type=float, default=.05)
    parser.add_argument("--rank-margin", type=float, default=.5)
    parser.add_argument("--rank-temperature", type=float, default=1.)
    parser.add_argument("--mechanism-width", type=int, default=16)
    parser.add_argument("--fixed-ewma-gain", type=float, default=.4)
    parser.add_argument("--capacity-iterations", type=int, default=12)
    parser.add_argument("--capacity-entropy", type=float, default=.2)
    parser.add_argument("--dynamics-ridge", type=float, default=.1)
    parser.add_argument("--signature-window", type=int, default=8)


def validate_options(args):
    import math
    positive = (args.memory_slots, args.attention_heads, args.flow_steps,
                args.flow_temperature, args.rank_temperature, args.mechanism_width,
                args.capacity_iterations, args.capacity_entropy, args.dynamics_ridge,
                args.signature_window)
    nonnegative = (args.rank_weight, args.semantic_rank_weight, args.rank_margin)
    if (not all(math.isfinite(float(x)) for x in positive + nonnegative)
            or min(positive) <= 0 or min(nonnegative) < 0):
        raise ValueError("Invalid Q-study options")
    if ALL_CONFIGS[args.variant].attention != "none" and args.hidden % args.attention_heads:
        raise ValueError("Attention width must be divisible by heads")
    if not math.isfinite(args.fixed_ewma_gain) or not 0 < args.fixed_ewma_gain <= 1:
        raise ValueError("EWMA gain must be in (0,1]")
