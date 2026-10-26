"""Cells set from a reviewed-corrections sidecar, with their provenance."""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.io.curator import CurateState, reconcilers
from openplaces.io.curator.reconcilers import apply_reviewed_corrections
from openplaces.recipe import get_recipe_by_id


def _state(frame):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=None,
        verbose=False,
        timer=None,
        curated=frame,
    )


def _sidecar(monkeypatch, rows):
    table = pd.DataFrame(rows, columns=['footprint_id', 'column', 'value', 'reason'])
    monkeypatch.setattr(
        'openplaces.recipe.get_recipe_by_id', lambda *_a, **_k: table.astype(object)
    )


def test_listed_cells_are_set_and_say_so(monkeypatch):
    _sidecar(
        monkeypatch,
        [
            ['AAA', 'occupancy_type', 'Commercial-Residential', 'fabricated'],
            ['AAA', 'n_dwellings', None, 'fabricated'],
        ],
    )
    frame = pd.DataFrame(
        {
            'occupancy_type': pd.Categorical(['Multi-Family', 'Single-Family']),
            'occupancy_type_source': ['overture_count', 'nsi'],
            'n_dwellings': [2.0, 1.0],
            'n_dwellings_source': ['overture', 'nsi'],
        },
        index=pd.Index(['AAA', 'BBB'], name='footprint_id'),
    )
    out = apply_reviewed_corrections(
        _state(frame), corrections_id='any', id_column='footprint_id'
    ).curated
    assert out.loc['AAA', 'occupancy_type'] == 'Commercial-Residential'
    assert out.loc['AAA', 'occupancy_type_source'] == 'reviewed_correction'
    assert pd.isna(out.loc['AAA', 'n_dwellings'])
    assert pd.isna(out.loc['AAA', 'n_dwellings_source'])
    # Unlisted rows keep their values.
    assert out.loc['BBB', 'occupancy_type'] == 'Single-Family'
    assert out.loc['BBB', 'n_dwellings'] == 1.0


def test_an_id_not_in_the_table_is_reported(monkeypatch):
    _sidecar(monkeypatch, [['GONE', 'n_dwellings', '3', 'fabricated']])
    frame = pd.DataFrame(
        {'n_dwellings': [2.0]}, index=pd.Index(['AAA'], name='footprint_id')
    )
    with pytest.warns(UserWarning, match='GONE'):
        apply_reviewed_corrections(
            _state(frame), corrections_id='any', id_column='footprint_id'
        )


def test_cambridge_sidecar_names_columns_the_step_reads():
    table = get_recipe_by_id(
        'US-MA-CAM_building-openplaces-2026_reviewed-corrections', dtype=str
    )
    assert list(table.columns) == ['footprint_id', 'column', 'value', 'reason']
    assert reconcilers.REVIEWED_CORRECTION_SOURCE == 'reviewed_correction'
