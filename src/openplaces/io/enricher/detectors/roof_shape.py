"""
Roof-shape classifier for satellite building imagery.
"""

from __future__ import annotations

from pathlib import Path

from openplaces.io.enricher.models import get_model
from openplaces.io.scrapers.types import ImageSet

_MODEL_URL = 'https://zenodo.org/record/7271554/files/trained_model_rooftype.pth'
_MODEL_FILENAME = 'roofTypeClassifier_v1.pth'
_CLASSES = ['Flat', 'Gable', 'Hip']


class RoofShapeClassifier:
    """Classify roof shape from satellite imagery."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model_path = Path(model_path) if model_path else None

    def predict(self, images: ImageSet, **kwargs) -> dict:
        """Predict roof-shape classes for the provided image set.

        Keyword arguments (e.g. batch_size, device, verbose) are passed
        through to predict_classes.
        """
        from openplaces.io.enricher.detectors.classify import predict_classes

        model_path = get_model(
            _MODEL_URL,
            _MODEL_FILENAME,
            self._model_path,
            label='roof-shape classifier model',
        )

        return predict_classes(images, model_path, _CLASSES, **kwargs)
