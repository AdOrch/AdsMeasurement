"""sk-causal: Causal inference algorithms with scikit-learn compatible APIs."""

from .base import BaseCausalEstimator
from .estimators import DoublyRobust, InversePropensityWeighting, PropensityScoreMatching

__version__ = "0.1.0"

__all__ = [
    "BaseCausalEstimator",
    "PropensityScoreMatching",
    "InversePropensityWeighting",
    "DoublyRobust",
]
