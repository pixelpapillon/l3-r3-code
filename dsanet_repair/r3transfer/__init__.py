"""Three cumulative upgrades of the historical frozen-DSANet R3 route."""

import copy
from dataclasses import dataclass
from dsanet_repair.eventnext import validate as validate_r

PARENT = "r3-core-envelope-free"


@dataclass(frozen=True)
class Configuration:
    detail: bool = True
    circulation: bool = False
    duration: bool = False


CONFIGS = {
    "l1-r3-detail": Configuration(),
    "l2-r3-detail-area": Configuration(circulation=True),
    "l3-r3-full": Configuration(circulation=True, duration=True),
}
VARIANTS = tuple(CONFIGS)
PARENTS = dict(zip(VARIANTS, (PARENT, VARIANTS[0], VARIANTS[1])))


def legacy_options(args):
    result = copy.copy(args)
    result.variant = PARENT
    return result


def validate(args):
    if args.variant not in CONFIGS:
        raise ValueError("Unknown R3-transfer experiment")
    legacy = validate_r(legacy_options(args))
    args.snapshot_epochs = legacy.snapshot_epochs
    return args
