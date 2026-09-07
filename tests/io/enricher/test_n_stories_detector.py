"""The story-count detector works on in-memory images and stays quiet.

The EfficientDet engine is not bundled (see `detectors/n_stories.py`), so
inference itself is stubbed. What is checked is the contract around it:
the detector consumes an `ImageSet` as the scrapers build it, fails with
the intended explanation when no engine is present, and importing it does
not change the process-wide warning filters.
"""

from __future__ import annotations

import importlib
import warnings
from io import BytesIO

import pytest
from PIL import Image as PILImage

from openplaces.io.enricher.detectors import n_stories
from openplaces.io.scrapers.types import Image, ImageSet


def _png_bytes() -> bytes:
    buffer = BytesIO()
    PILImage.new('RGB', (8, 8)).save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def image_set() -> ImageSet:
    images = ImageSet()
    images.add_image('F1', Image('footprint_F1.png', payload=_png_bytes()))
    images.add_image('F2', Image('footprint_F2.png'))  # fetch failed
    return images


@pytest.fixture
def model_on_disk(monkeypatch, tmp_path):
    path = tmp_path / 'weights.pth'
    path.write_bytes(b'')
    monkeypatch.setattr(n_stories, 'get_model', lambda *a, **k: path)
    return path


def test_importing_the_detector_leaves_warnings_enabled():
    """No module-level blanket ignore: reloading must not silence warnings.

    The module is reloaded so its top-level code runs inside this test,
    where pytest restores the filters afterwards; an import cached from an
    earlier test would hide a filter installed at import time.
    """
    importlib.reload(n_stories)
    with warnings.catch_warnings(record=True) as caught:
        warnings.warn('probe', stacklevel=1)
    assert caught, 'a warning raised after the import was swallowed'


def test_predict_without_an_engine_reports_that_it_is_not_bundled(
    image_set, model_on_disk
):
    """The absent engine, not a missing ImageSet attribute, is the error."""
    n_stories._load_infer.cache_clear()
    try:
        with pytest.raises(RuntimeError, match='not bundled'):
            n_stories.NStoriesDetector().predict(image_set)
    finally:
        n_stories._load_infer.cache_clear()


def test_predict_reads_images_from_memory(monkeypatch, image_set, model_on_disk):
    """An in-memory image is detected on; one with no pixels yields None."""

    class _Engine:
        def predict(self, path, threshold):
            return None, None, []  # no boxes: zero stories

    monkeypatch.setattr(n_stories, '_load_infer', lambda *a, **k: _Engine())
    predictions = n_stories.NStoriesDetector().predict(image_set)
    assert predictions == {'F1': 0, 'F2': None}
