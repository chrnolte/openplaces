"""Tests for the complete-coverage guard's error handling.

The guard exists so that a missing reference which declares complete
coverage raises instead of being soft-skipped. A blanket except turned
every load failure into the tolerant answer, so a broken recipe was
skipped precisely where the guard was meant to stop the run.
"""

import pytest

from openplaces import recipe as recipe_module
from openplaces.recipe import coverage_is_complete, raise_if_coverage_complete


def test_missing_recipe_stays_tolerant(monkeypatch):
    def _missing(recipe_id, **kwargs):
        raise OSError('Not found: fabricated')

    monkeypatch.setattr(recipe_module, 'get_recipe_by_id', _missing)
    assert coverage_is_complete('US_building-fabricated-2026') is False


def test_unloadable_recipe_raises(monkeypatch):
    def _broken(recipe_id, **kwargs):
        raise ValueError('recipe does not validate')

    monkeypatch.setattr(recipe_module, 'get_recipe_by_id', _broken)
    with pytest.raises(ValueError):
        coverage_is_complete('US_building-fabricated-2026')
    with pytest.raises(ValueError):
        raise_if_coverage_complete('US_building-fabricated-2026', 'US-NC-AL')
