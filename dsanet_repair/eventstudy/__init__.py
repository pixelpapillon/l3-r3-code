"""Six paired event-structure experiments; no implicit execution or cache edits."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Configuration:
    events: bool = True
    category_support: bool = True
    ordered: bool = True
    locked: bool = True
    generic: bool = False


CONFIGS = {
    "e0-c1-control": Configuration(events=False, locked=False),
    "e1-full-123": Configuration(),
    "e2-shared-support": Configuration(category_support=False),
    "e3-orderless-events": Configuration(ordered=False),
    "e4-free-response": Configuration(locked=False),
    "e5-class-attention": Configuration(locked=False, generic=True),
}
VARIANTS = tuple(CONFIGS)


def add_arguments(parser):
    parser.add_argument("--event-slots", type=int, default=8)
    parser.add_argument("--event-rank", type=int, default=8)
    parser.add_argument("--event-samples", type=int, default=8)
    parser.add_argument("--locked-alpha", type=float, default=.5)
    parser.add_argument("--response-holdout-fraction", type=float, default=.1)
    parser.add_argument("--response-holdout-salt", default="eventstudy-query-v1")
    return parser


def validate(args):
    if min(args.event_slots, args.event_rank) < 1 or args.event_samples < 3:
        raise ValueError("Positive slots/rank and at least three event samples are required")
    if (not math.isfinite(args.locked_alpha) or not 0 < args.locked_alpha <= 1 or
            not math.isfinite(args.response_holdout_fraction) or
            not 0 <= args.response_holdout_fraction < 1):
        raise ValueError("Invalid locked response or holdout fraction")
    if not args.response_holdout_salt:
        raise ValueError("An immutable query-holdout salt is required")

