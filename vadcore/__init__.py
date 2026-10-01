"""Shared tensor contracts and adapted VadCLIP feature backend."""
from .types import VideoSample, Prediction
from .backbone import FeatureVAD, BackboneConfig

__all__ = ["VideoSample", "Prediction", "FeatureVAD", "BackboneConfig"]
