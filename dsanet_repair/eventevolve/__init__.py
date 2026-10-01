"""E7-E14: event detail, ordered circulation and duration-marginal evidence."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Experiment:
    name: str
    detail: bool
    circulation: bool
    readout: str
    order_blind: bool
    comparison: str
    question: str

    # The shared weak-label objective consumes final joint probabilities in
    # EVERY E7-E14 run, including the raw-logit E8 control.
    probability: bool = True


EXPERIMENTS = {
    "e7": Experiment("full-no-marginal", False, False, "pointwise", False, "E5",
                     "Does the marginal KL term help or restrict correction?"),
    "e8": Experiment("raw-readout-joint-loss", False, False, "raw", False, "E6 / E5",
                     "Separate the joint-probability objective from the reconstruction readout."),
    "e9": Experiment("multiresolution-detail", True, False, "kl", False, "E5",
                     "Do signed within-event details outperform interval-constant evidence?"),
    "e10": Experiment("ordered-circulation", False, True, "kl", False, "E5",
                      "Does ordered non-collinear evolution add to symmetric event pairs?"),
    "e11": Experiment("duration-marginal-readout", False, False, "duration", False, "E7 (primary) / E5",
                      "Can marginalizing duration hypotheses turn local evidence into better decisions?"),
    "e12": Experiment("detail-and-circulation", True, True, "kl", False, "E9 / E10 / E5",
                      "Do within-event detail and chronological composition complement each other?"),
    "e13": Experiment("full-event-calculus", True, True, "duration", False, "E12 / E11",
                      "Do upgraded M1, M2 and M3 cooperate?"),
    "e14": Experiment("full-orientation-blind", True, True, "duration", True, "E13",
                      "Does the sign of temporal circulation matter beyond its magnitude?"),
}


def matrix():
    return [{"id": key, **asdict(value)} for key, value in EXPERIMENTS.items()]
