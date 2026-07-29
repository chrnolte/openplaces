"""Loaded detector models are cached per (model_path, device), not reloaded
on every admin unit processed in a pipeline run."""

import torch

from openplaces.io.enricher.detectors import classify, n_stories
from openplaces.io.enricher.detectors.efficientdet_lib import infer as infer_module


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


class _FakeInfer:
    def __init__(self):
        self.load_calls = 0

    def load_model(self, path, classes, use_gpu=False):
        self.load_calls += 1


def test_efficientdet_infer_loaded_once_per_path_and_device(monkeypatch):
    n_stories._load_infer.cache_clear()
    created = []

    def fake_infer():
        instance = _FakeInfer()
        created.append(instance)
        return instance

    monkeypatch.setattr(infer_module, 'Infer', fake_infer)
    try:
        infer1 = n_stories._load_infer('model_a.pth', False)
        infer2 = n_stories._load_infer('model_a.pth', False)
        infer3 = n_stories._load_infer('model_a.pth', True)

        assert infer1 is infer2
        assert infer1 is not infer3
        assert len(created) == 2
        assert infer1.load_calls == 1
    finally:
        n_stories._load_infer.cache_clear()
