"""
Occupancy classifier for street-level building imagery.
"""

from __future__ import annotations

from pathlib import Path

from openplaces.io.enricher.models import get_model
from openplaces.io.scrapers.types import ImageSet

_MODEL_URL = 'https://zenodo.org/record/7272099/files/trained_model_occupancy_v1.pth'
_MODEL_FILENAME = 'OccupancyClassifier_v1.pth'
_CLASSES = ['Other', 'Residential']


class OccupancyClassifier:
    """Classify building occupancy from street-level imagery."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model_path = Path(model_path) if model_path else None

    def predict(self, images: ImageSet, **kwargs) -> dict:
        """Predict occupancy classes for the provided image set.

        Keyword arguments (e.g. batch_size, device, verbose) are passed
        through to predict_classes.
        """
        from openplaces.io.enricher.detectors.classify import predict_classes

        model_path = get_model(
            _MODEL_URL,
            _MODEL_FILENAME,
            self._model_path,
            label='occupancy classifier model',
        )

        return predict_classes(images, model_path, _CLASSES, **kwargs)
