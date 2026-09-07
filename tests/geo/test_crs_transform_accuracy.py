"""An exact PROJ operation (accuracy 0.0) must outrank an approximate one."""

from __future__ import annotations

from dataclasses import dataclass

from openplaces.geo.crs_transforms import _accuracy_sort_key


@dataclass
class _FakeTransformer:
    accuracy: float | None
    description: str


def test_zero_accuracy_sorts_first():
    exact = _FakeTransformer(0.0, 'exact operation')
    approximate = _FakeTransformer(1.0, 'approximate operation')

    assert min([approximate, exact], key=_accuracy_sort_key) is exact


def test_unknown_accuracy_sorts_last():
    unknown = _FakeTransformer(None, 'unrated operation')
    negative = _FakeTransformer(-1.0, 'unrated operation too')
    rated = _FakeTransformer(5.0, 'rated operation')

    assert min([unknown, negative, rated], key=_accuracy_sort_key) is rated
