"""The placeslab fmv recipes index every table by the parcel key,
whether a delivery stores `geometry_sha3_224` as the pandas index or as
an ordinary column beside a RangeIndex, so the join pairs rows by key
and never by row position. All keys and values are fabricated."""

import importlib

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces import cfg
from openplaces.io import read_parquet, save_parquet
from openplaces.io.aggregate import join_partitions_by_index
from openplaces.io.transform import index_by_key_column
from openplaces.recipe import get_output_path, get_recipe_by_id

KEY = 'geometry_sha3_224'
ADMIN_ID = 'US-MA-MI'
KEYS = ['hash_a', 'hash_b', 'hash_c']
RECIPE_IDS = ['US_parcel-placeslab-fmv2026', 'US_parcel-placeslab-fmv2026open']


@pytest.fixture
def cache_to_tmp(tmp_path, monkeypatch):
    """Redirect the 'cache' data dir so tests never touch real outputs."""
    monkeypatch.setitem(cfg.config['directories'], 'cache', tmp_path)
    return tmp_path


def _as_delivered(df, layout):
    """Lay `df` (indexed by KEY) out as a delivery would."""
    if layout == 'index':
        return df
    return df.reset_index()


def _raw_tables(layout):
    """Fabricated raw tables; predictors arrive in another row order."""
    boundaries = gpd.GeoDataFrame(
        {'name': ['n_a', 'n_b', 'n_c']},
        geometry=[box(i, i, i + 1, i + 1) for i in range(3)],
        crs='EPSG:4326',
        index=pd.Index(KEYS, name=KEY),
    )
    predictions = pd.DataFrame(
        {'lnusd_ha': [1.0, 2.0, 3.0]}, index=pd.Index(KEYS, name=KEY)
    )
    shuffled = [KEYS[2], KEYS[0], KEYS[1]]
    predictors = pd.DataFrame(
        {'slope': [30.0, 10.0, 20.0]}, index=pd.Index(shuffled, name=KEY)
    )
    return {
        name: _as_delivered(df, layout)
        for name, df in (
            ('boundaries', boundaries),
            ('predictions', predictions),
            ('predictors', predictors),
        )
    }


def _apply_create_index(recipe, df):
    """Run the recipe's own `create_index` directive on a raw table."""
    config = recipe['create_index']
    module_name, function_name = config['function'].rsplit('.', 1)
    function = getattr(importlib.import_module(module_name), function_name)
    return function(df, **config.get('args', {}))


def _write(recipe, table_name, df):
    save_parquet(df, get_output_path(recipe, ADMIN_ID, partition_id=table_name))


def test_key_column_is_promoted_and_dropped():
    raw = pd.DataFrame({KEY: ['hash_b', 'hash_a'], 'x': [2, 1]})

    result = index_by_key_column(raw, column=KEY, name='parcel_id_int')

    assert result.index.name == 'parcel_id_int'
    assert list(result.index) == ['hash_b', 'hash_a']
    assert KEY not in result.columns


def test_existing_key_index_is_kept_and_renamed():
    raw = pd.DataFrame({'x': [1]}, index=pd.Index(['hash_a'], name=KEY))

    result = index_by_key_column(raw, column=KEY, name='parcel_id_int')

    assert result.index.name == 'parcel_id_int'
    assert list(result.index) == ['hash_a']


def test_row_numbers_without_key_column_raise():
    with pytest.raises(ValueError, match='RangeIndex'):
        index_by_key_column(pd.DataFrame({'x': [1]}), column=KEY, name='k')


@pytest.mark.parametrize('recipe_id', RECIPE_IDS)
@pytest.mark.parametrize('layout', ['index', 'column'])
def test_join_pairs_tables_by_key(cache_to_tmp, recipe_id, layout):
    recipe = get_recipe_by_id(recipe_id)
    for table_name, raw in _raw_tables(layout).items():
        _write(recipe, table_name, _apply_create_index(recipe, raw))

    join_partitions_by_index(
        recipe,
        table_names=['boundaries', 'predictions', 'predictors'],
        admin_ids=ADMIN_ID,
        join_key_name='parcel_id_int',
    )

    result = read_parquet(get_output_path(recipe, ADMIN_ID), geom=True)
    by_key = result.set_index('parcel_id_int')
    assert by_key.loc[KEYS, 'lnusd_ha'].tolist() == [1.0, 2.0, 3.0]
    assert by_key.loc[KEYS, 'slope'].tolist() == [10.0, 20.0, 30.0]
    assert by_key.loc[KEYS, 'name'].tolist() == ['n_a', 'n_b', 'n_c']
    assert KEY not in result.columns


@pytest.mark.parametrize('layout', ['index', 'column'])
def test_crosswalk_references_match_the_joined_index(cache_to_tmp, layout):
    """A crosswalk county (no boundaries) is keyed by the parcel hash as
    a string, the form its crosswalk's reference_parcel_id takes."""
    recipe = get_recipe_by_id(RECIPE_IDS[0])
    raw = _raw_tables(layout)
    for table_name in ('predictions', 'predictors'):
        _write(recipe, table_name, _apply_create_index(recipe, raw[table_name]))
    crosswalk = pd.DataFrame(
        {'parcel_id_new': ['new_1', 'new_2', 'new_3'], 'reference_parcel_id': KEYS}
    )
    crosswalk_path = get_output_path(recipe, ADMIN_ID, partition_id='crosswalk')
    crosswalk_path.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_parquet(crosswalk_path)

    join_partitions_by_index(
        recipe,
        table_names=['boundaries', 'predictions', 'predictors', 'crosswalk'],
        admin_ids=ADMIN_ID,
        join_key_name='parcel_id_int',
        sidecar_tables=['crosswalk'],
    )

    result = read_parquet(get_output_path(recipe, ADMIN_ID))
    assert set(crosswalk['reference_parcel_id']) == set(result.index.astype(str))
    assert result.loc[KEYS, 'slope'].tolist() == [10.0, 20.0, 30.0]
