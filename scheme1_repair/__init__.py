"""Scheme 1: calibrated, class-selective normal feature repair."""
from .reference import NormalReference, ReferenceConfig
from .repair import SelectiveRepair, RepairConfig
from .model import EvidenceFusion, RepairSystem
