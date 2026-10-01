"""R1-R6: isolated successors to the immutable E0-E5 event study."""

from dataclasses import dataclass
import copy
import math


@dataclass(frozen=True)
class Configuration:
    morphology: bool = True
    locked: bool = False
    assembly: bool = False
    existence: bool = False
    interaction: bool = False


CONFIGS = {
    "r1-e3-free": Configuration(morphology=False),
    "r2-core-envelope-locked": Configuration(locked=True),
    "r3-core-envelope-free": Configuration(),
    "r4-marginal-assembly": Configuration(assembly=True),
    "r5-existence-readout": Configuration(existence=True),
    "r6-joint-interaction": Configuration(interaction=True),
}
VARIANTS = tuple(CONFIGS)
PARENTS = dict(zip(VARIANTS, ("e3-orderless-events", "e3-orderless-events", VARIANTS[1],
                            VARIANTS[2], VARIANTS[2], VARIANTS[2])))


def add_arguments(parser):
    parser.add_argument("--joint-cache", help="R6-only complete joint measurement supplement")
    parser.add_argument("--assembly-cost", type=float, default=.01)
    parser.add_argument("--assembly-temperature", type=float, default=.1)
    parser.add_argument("--existence-temperature", type=float, default=.5)
    parser.add_argument("--reference-momentum", type=float, default=.95)
    parser.add_argument("--interaction-alpha", type=float, default=.5)
    return parser


def validate(args):
    from dsanet_repair.expansion.train import normalize_options
    if args.variant not in CONFIGS:
        raise ValueError("Unknown R experiment")
    compatible = copy.copy(args)
    compatible.variant = "e3-orderless-events"
    normalize_options(compatible)
    args.snapshot_epochs = compatible.snapshot_epochs
    numbers = (args.assembly_cost, args.assembly_temperature, args.existence_temperature,
               args.reference_momentum, args.interaction_alpha)
    if (not all(math.isfinite(v) for v in numbers) or args.assembly_cost <= 0 or
            min(args.assembly_temperature, args.existence_temperature) <= 0 or
            not 0 <= args.reference_momentum < 1 or not 0 < args.interaction_alpha <= 1):
        raise ValueError("Invalid R-study hyperparameters")
    if CONFIGS[args.variant].interaction != bool(args.joint_cache):
        raise ValueError("Only R6 requires --joint-cache; other variants must not consume it")
    return args
