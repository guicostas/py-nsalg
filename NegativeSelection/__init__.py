"""
This module implements Negative Selection-based learning algorithms.
Training algorithms do not utilize labeled data, just normal data applied to
anomaly detection tasks. In tests, samples and trackers are needed in order
to generate labels that categorize normal or anomaly data.
"""

from ._classical import ClassicalNegativeSelection
from ._vdetector import VDetector
from .match_detector import match_detector

__all__ = ["ClassicalNegativeSelection", "VDetector", "match_detector"]
