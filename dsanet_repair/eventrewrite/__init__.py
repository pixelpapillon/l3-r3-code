"""EventRewrite: six matched, cache-free DSANet downstream experiments."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Experiment:
    name: str
    morphology: bool
    interaction: bool
    probability: bool
    question: str


EXPERIMENTS = {
    "e1": Experiment("matched-dsa", False, False, False,
                     "What does matched downstream fine-tuning alone achieve?"),
    "e2": Experiment("decision-aligned", False, False, True,
                     "Does training the evaluated joint probability help?"),
    "e3": Experiment("morphology-decision", True, False, True,
                     "Does multiscale core/phase/context evidence help without interactions?"),
    "e4": Experiment("interaction-decision", False, True, True,
                     "Do signed second-order potentials help with plain interval tokens?"),
    "e5": Experiment("full-three-modules", True, True, True,
                     "Do morphology, signed composition and probability reconstruction cooperate?"),
    "e6": Experiment("full-without-decision", True, True, False,
                     "Does the full evidence model need the new readout/objective?"),
}


def matrix():
    return [{"id": key, **asdict(value)} for key, value in EXPERIMENTS.items()]
