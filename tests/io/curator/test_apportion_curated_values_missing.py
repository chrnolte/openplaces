"""Tests for the apportion_curated_values curate step."""

import pandas as pd
import pytest

from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.geo.link import get_entity_link_path
from openplaces.io import to_parquet
from openplaces.io.curator import CurateState
from openplaces.io.curator.evidence import apportion_curated_values
from openplaces.recipe import get_output_path, get_recipe_by_id

SPINE = 'US_footprint-spine-2026'
# The sidecar is keyed by the recipe that ran the link steps -- the
# geospine under the split (get_link_owner_recipe_id resolves it from the
# spine's entity_recipe chain).
GEOSPINE = 'US_footprint-geospine-2026'
PARCEL_CURATE = 'US_parcel-openplaces-2026'
COUNTY = 'US-FL-LA'


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    dirs = dict(cfg.config['directories'])
    dirs['data_root'] = tmp_path
    for name in ('core', 'external', 'raw', 'cache', 'out', 'share'):
        dirs[name] = tmp_path / 'data' / name
    dirs['heap'] = tmp_path / 'data/cache/_heap'
    dirs['logs'] = tmp_path / 'data/cache/_logs'
    monkeypatch.setitem(cfg.config, 'directories', dirs)
    return tmp_path


def _write_sidecar():
    links = pd.DataFrame(
        {
            'footprint_id': ['f1', 'f2'],
            'parcel_id': ['P1', 'P2'],
            'area_intersection_m2': [40.0, 30.0],
            'link': ['unique parcel', 'unique parcel'],
        }
    )
    path = get_entity_link_path(GEOSPINE, PARCEL_CURATE, admin_id=COUNTY)
    to_parquet(links, path)


def _write_curated_parcels(with_improvement_value: bool):
    parcels = pd.DataFrame(
        {
            'parcel_id': ['P1', 'P2'],
            'land_value': [100.0, 200.0],
            **({'improvement_value': [10.0, 20.0]} if with_improvement_value else {}),
        }
    )
    ref_recipe = get_recipe_by_id(PARCEL_CURATE)
    to_parquet(parcels, get_output_path(ref_recipe, AdminId(COUNTY)))


def _state():
    curated = pd.DataFrame(
        {'parcel_id': ['P1', 'P2']},
        index=pd.Index(['f1', 'f2'], name='footprint_id'),
    )
    return CurateState(
        recipe={'recipe_id': 'US_footprint-openplaces-2026'},
        entity_recipe=get_recipe_by_id(SPINE),
        admin_id=AdminId(COUNTY),
        verbose=True,
        timer=None,
        curated=curated,
    )


def test_missing_ref_column_skipped_not_raised(data_root):
    _write_sidecar()
    _write_curated_parcels(with_improvement_value=False)
    state = apportion_curated_values(
        _state(),
        recipe_id=PARCEL_CURATE,
        columns={
            'improvement_value': 'improvement_value_parcel',
            'land_value': 'land_value_parcel',
        },
        link_recipe_id=PARCEL_CURATE,
    )
    # Present column apportions normally.
    assert state.curated.loc['f1', 'land_value_parcel'] == 100.0
    assert state.curated.loc['f2', 'land_value_parcel'] == 200.0
    # Missing column is left null rather than raising.
    assert state.curated['improvement_value_parcel'].isna().all()


def test_present_columns_still_apportion(data_root):
    _write_sidecar()
    _write_curated_parcels(with_improvement_value=True)
    state = apportion_curated_values(
        _state(),
        recipe_id=PARCEL_CURATE,
        columns={
            'improvement_value': 'improvement_value_parcel',
            'land_value': 'land_value_parcel',
        },
        link_recipe_id=PARCEL_CURATE,
    )
    assert state.curated.loc['f1', 'improvement_value_parcel'] == 10.0
    assert state.curated.loc['f2', 'improvement_value_parcel'] == 20.0
