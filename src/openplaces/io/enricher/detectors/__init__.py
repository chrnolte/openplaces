"""Image-based building attribute detectors for enrichment."""

from .n_stories import NStoriesDetector
from .occupancy import OccupancyClassifier
from .roof_shape import RoofShapeClassifier

__all__ = [
    'NStoriesDetector',
    'OccupancyClassifier',
    'RoofShapeClassifier',
]
