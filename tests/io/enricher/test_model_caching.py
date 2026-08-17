"""Loaded detector models are cached per (model_path, device), not reloaded
on every admin unit processed in a pipeline run.

The EfficientDet story-count engine is no longer bundled (LGPL-3.0, see
`detectors/n_stories.py`), so its loader is covered by asserting that it
fails loudly rather than by exercising a cache it no longer has.
"""

import pytest
import torch

from openplaces.io.enricher.detectors import classify, n_stories


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self


def test_classifier_model_loaded_once_per_path_and_device(monkeypatch):
    classify._load_model.cache_clear()
    calls = []

    def fake_load(path, map_location=None, weights_only=None):
        calls.append(path)
        return _FakeModel()

    monkeypatch.setattr(torch, 'load', fake_load)
    try:
        m1 = classify._load_model('model_a.pth', torch.device('cpu'))
        m2 = classify._load_model('model_a.pth', torch.device('cpu'))
        m3 = classify._load_model('model_b.pth', torch.device('cpu'))

        assert m1 is m2
        assert m1 is not m3
        assert calls == ['model_a.pth', 'model_b.pth']
    finally:
        classify._load_model.cache_clear()


def test_efficientdet_infer_reports_that_it_is_not_bundled():
    """The removed engine must fail with an explanation, not an ImportError.

    A bare ImportError would read as a broken install; the message has to
    say why the engine is absent and what still supplies story counts.
    """
    n_stories._load_infer.cache_clear()
    try:
        with pytest.raises(RuntimeError, match='not bundled'):
            n_stories._load_infer('model_a.pth', False)
    finally:
        n_stories._load_infer.cache_clear()


def test_efficientdet_removal_is_complete():
    """The LGPL-3.0 subtree must not come back without a NOTICE carve-out."""
    with pytest.raises(ImportError):
        import openplaces.io.enricher.detectors.efficientdet_lib  # noqa: F401
