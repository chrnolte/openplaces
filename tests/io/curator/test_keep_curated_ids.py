"""keep_curated_ids drops ids the parcel curate no longer carries.

The parcel curate drops slivers under 1 m2; a building or footprint can
still name one (transition review 3.5). All ids are fabricated.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.io.curator import CurateState, evidence


def _state(curated):
    return CurateState(
        recipe={'entity': {'entity_type': 'building'}},
        entity_recipe={},
        admin_id='US-NC-BR',
        verbose=False,
        timer=None,
        curated=curated,
    )


@pytest.fixture
def curated_parcels(tmp_path, monkeypatch):
    """Curated parcels a, b and c, with parcel_id as the index."""
    path = tmp_path / 'parcels.parquet'
    pd.DataFrame(
        {'area_ha': [0.1, 0.2, 0.3]}, index=pd.Index(['a', 'b', 'c'], name='parcel_id')
    ).to_parquet(path)
    monkeypatch.setattr(evidence, 'get_output_path', lambda *a, **k: path)
    return path


def _run(curated):
    return evidence.keep_curated_ids(
        _state(curated), recipe_id='US_parcel-openplaces-2026'
    ).curated


def test_dropped_ids_leave_both_columns(curated_parcels):
    curated = pd.DataFrame(
        {
            'parcel_id': ['a', 'x', 'b', 'x', 'c'],
            'parcel_id_all': [None, 'x|b', 'b|x|c', None, 'c|y'],
        },
        index=pd.Index(['b1', 'b2', 'b3', 'b4', 'b5'], name='building_id'),
    )
    got = _run(curated)
    # Untouched where every id is curated.
    assert got.loc['b1', 'parcel_id'] == 'a'
    assert pd.isna(got.loc['b1', 'parcel_id_all'])
    # A dropped dominant lot is not replaced by the next one.
    assert pd.isna(got.loc['b2', 'parcel_id'])
    assert got.loc['b2', 'parcel_id_all'] == 'b'
    # A list keeps the curated ids in order.
    assert got.loc['b3', 'parcel_id_all'] == 'b|c'
    # Nothing left: missing.
    assert pd.isna(got.loc['b4', 'parcel_id'])
    assert pd.isna(got.loc['b4', 'parcel_id_all'])
    # A list reduced to the base id is left missing, as collect_link_ids
    # leaves it.
    assert got.loc['b5', 'parcel_id'] == 'c'
    assert pd.isna(got.loc['b5', 'parcel_id_all'])


def test_an_absent_column_is_skipped(curated_parcels):
    curated = pd.DataFrame(
        {'parcel_id': ['a', 'x']}, index=pd.Index(['b1', 'b2'], name='building_id')
    )
    got = _run(curated)
    assert 'parcel_id_all' not in got.columns
    assert got['parcel_id'].tolist()[0] == 'a'
    assert pd.isna(got.loc['b2', 'parcel_id'])


def test_the_reference_must_be_a_curate_recipe(curated_parcels):
    curated = pd.DataFrame({'parcel_id': ['a']}, index=pd.Index(['b1']))
    with pytest.raises(ValueError, match="stage 'curate'"):
        evidence.keep_curated_ids(_state(curated), recipe_id='US_parcel-geospine-2026')
