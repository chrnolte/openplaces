"""A curated reference a unit saved empty reads as "none of that entity".

Harmonize saves a table without rows for a unit that has nothing to
build from, and a curate of it saves a table without rows or columns.
A later curate reading that table as a reference finds no id column;
it must join nothing rather than fail. A table with rows but without
the id column is still a broken reference and raises.
"""

import geopandas as gpd
import pandas as pd
import pytest

import openplaces.io.curator.evidence as evidence
from openplaces.core.schema import AdminId, Entity
from openplaces.io import read_parquet, readers, save_parquet
from openplaces.io.curator import CurateState
from openplaces.recipe import get_output_path


def _state(curated):
    return CurateState(
        recipe={'recipe_id': 'US_building-x-2026'},
        entity_recipe={},
        admin_id=AdminId('US-XX-YY'),
        verbose=False,
        timer=None,
        curated=curated,
    )


def _buildings():
    return pd.DataFrame(
        {'parcel_id': ['p1', 'p2', None]},
        index=pd.Index(['b1', 'b2', 'b3'], name='building_id'),
    )


@pytest.fixture
def reference(monkeypatch, tmp_path):
    """Point the curated reference at a file the test writes."""
    path = tmp_path / 'US-XX-YY_parcel-x-2026.parquet'
    monkeypatch.setattr(
        evidence, 'get_recipe_by_id', lambda *a, **k: {'stage': 'curate'}
    )
    monkeypatch.setattr(evidence, 'get_output_path', lambda *a, **k: path)
    return path


def test_an_empty_reference_links_nothing(reference):
    # What the curate of a unit without parcels saves.
    save_parquet(pd.DataFrame(), reference)

    state = evidence.copy_from_curated_entity(
        _state(_buildings()), 'US_parcel-x-2026', columns={'use_group': 'use_parcel'}
    )

    assert state.curated['use_parcel'].isna().all()
    assert list(state.curated.index) == ['b1', 'b2', 'b3']


def test_an_empty_reference_keeps_no_id(reference):
    save_parquet(pd.DataFrame(), reference)

    state = evidence.empty_unknown_ids(_state(_buildings()), 'US_parcel-x-2026')

    assert state.curated['parcel_id'].isna().all()


def test_an_empty_reference_apportions_nothing(reference):
    # No overlay link exists in the test: the step must not need one.
    save_parquet(pd.DataFrame(), reference)

    state = evidence.split_curated_values(
        _state(_buildings()),
        'US_parcel-x-2026',
        columns={'improvement_value': 'structure_value'},
    )

    assert state.curated['structure_value'].isna().all()


def test_an_empty_reference_with_geometry_links_nothing(reference):
    # What harmonize saves for a geometry recipe that built nothing.
    empty = gpd.GeoDataFrame(
        {'geometry': gpd.GeoSeries([], dtype='geometry')}, crs='EPSG:4326'
    )
    save_parquet(empty, reference)

    state = evidence.copy_from_curated_entity(
        _state(_buildings()), 'US_parcel-x-2026', columns={'use_group': 'use_parcel'}
    )

    assert state.curated['use_parcel'].isna().all()


def test_a_reference_with_rows_but_no_id_still_raises(reference):
    save_parquet(pd.DataFrame({'use_group': ['residential']}), reference)

    with pytest.raises(ValueError, match="has no 'parcel_id' column"):
        evidence.copy_from_curated_entity(
            _state(_buildings()),
            'US_parcel-x-2026',
            columns={'use_group': 'use_parcel'},
        )
    with pytest.raises(ValueError, match="has no 'parcel_id' column"):
        evidence.empty_unknown_ids(_state(_buildings()), 'US_parcel-x-2026')


def test_a_table_without_rows_or_sidecar_reads_with_geometry(tmp_path):
    # A curate of an empty unit saves a plain table: no `_geo` sidecar.
    path = tmp_path / 'empty.parquet'
    save_parquet(pd.DataFrame(), path)

    frame = read_parquet(path, geom=True)

    assert isinstance(frame, gpd.GeoDataFrame)
    assert frame.empty
    assert 'geometry' in frame.columns


GEOSPINE = {
    'recipe_id': 'US_parcel-geotest-2026',
    'admin_id': AdminId('US'),
    'entity': Entity('parcel', 'geotest', '2026'),
    'save_to': {'admin_level': 3, 'data_dir': 'core'},
}
ATTRIBUTES = {
    'recipe_id': 'US_parcel-test-2026',
    'admin_id': AdminId('US'),
    'entity': Entity('parcel', 'test', '2026'),
    'entity_recipe': 'US_parcel-geotest-2026',
    'save_to': {'admin_level': 3, 'data_dir': 'core', 'geometry': False},
}


@pytest.fixture
def empty_attribute_unit(mock_data_root, monkeypatch):
    """An attribute-only recipe whose unit was saved as a placeholder."""
    monkeypatch.setattr(readers, 'get_recipe_by_id', lambda *a, **k: GEOSPINE)
    save_parquet(pd.DataFrame(), get_output_path(ATTRIBUTES, 'US-NC-AAA'))


def test_get_entities_reads_a_placeholder_with_geometry(empty_attribute_unit):
    data = readers.get_entities(ATTRIBUTES, admin_id='US-NC-AAA', geom=True)

    assert isinstance(data, gpd.GeoDataFrame)
    assert data.empty
    assert 'geometry' in data.columns


def test_get_entities_reads_named_columns_of_a_placeholder(empty_attribute_unit):
    data = readers.get_entities(
        ATTRIBUTES, admin_id='US-NC-AAA', columns=['parcel_id', 'land_value']
    )

    assert data.empty


def test_a_table_with_rows_and_no_sidecar_still_raises(tmp_path):
    path = tmp_path / 'rows.parquet'
    save_parquet(pd.DataFrame({'x': [1]}), path)

    with pytest.raises(FileNotFoundError):
        read_parquet(path, geom=True)
