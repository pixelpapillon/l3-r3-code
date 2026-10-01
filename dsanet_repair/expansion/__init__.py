"""Six fixed, feature-only DSANet research configurations; no automatic jobs."""

VARIANTS = (
    "p1-center-full", "p2-center-no-response", "p3-center-split",
    "p4-dilated-context", "p5-structured-transport", "p6-primary-aligned",
)

SPLIT_VARIANTS = frozenset(VARIANTS[2:5])

