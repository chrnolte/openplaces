"""`Ingester` integration of `join_partitions_by_index`: the `join_partitions_by`
recipe field opts a `download_by: {partition: table}` recipe into automatically
column-joining its per-table outputs after ingest (see
`io.aggregate.join_partitions_by_index`, already tested directly in
`tests/io/test_aggregate_join_by_index.py`; this file covers the new `Ingester`
wiring: the `_is_joined_table_partition` property, `_join_table_partitions`'s
delegation, and `_early_warnings`'s validation)."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.ingester as ingester_module
from openplaces import cfg
from openplaces.io import read_parquet, save_parquet
from openplaces.io.ingester import ingest
from openplaces.recipe import get_output_path, get_recipe_by_id

RECIPE_ID = 'US_parcel-placeslab-fmv2026'
ADMIN_ID = 'US-NY-AL'


@pytest.fixture
def cache_to_tmp(tmp_path, monkeypatch):
    """Redirect the 'cache' data dir so tests never touch real outputs."""
    monkeypatch.setitem(cfg.config['directories'], 'cache', tmp_path)
    return tmp_path


def _bare_ingester(recipe, **extra):
    ing = ingester_module.Ingester.__new__(ingester_module.Ingester)
    ing.recipe = recipe
    for key, value in extra.items():
        setattr(ing, key, value)
    return ing


def test_is_joined_table_partition_true_when_recipe_declares_it():
    ing = _bare_ingester({'join_partitions_by': {'join_key_name': 'parcel_id_int'}})
    assert ing._is_joined_table_partition is True


def test_is_joined_table_partition_false_when_absent():
    ing = _bare_ingester({})
    assert ing._is_joined_table_partition is False


def test_join_table_partitions_calls_join_partitions_by_index_with_recipe_config(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        'openplaces.io.aggregate.join_partitions_by_index',
        lambda *a, **k: calls.append((a, k)),
    )

    recipe = {
        'download_by': {
            'partition': 'table',
            'table_names': ['boundaries', 'predictions'],
        },
        'join_partitions_by': {'join_key_name': 'parcel_id_int', 'keep_original': True},
    }
    ing = _bare_ingester(recipe, admin_ids_to_save=['US-MA-MI'], verbose=True)

    ing._join_table_partitions()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is recipe
    assert kwargs['table_names'] == ['boundaries', 'predictions']
    assert kwargs['admin_ids'] == ['US-MA-MI']
    assert kwargs['join_key_name'] == 'parcel_id_int'
    assert kwargs['keep_original'] is True
    assert kwargs['verbose'] is True


def test_join_table_partitions_defaults_keep_original_false(monkeypatch):
    calls = []
    monkeypatch.setattr(
        'openplaces.io.aggregate.join_partitions_by_index',
        lambda *a, **k: calls.append(k),
    )

    recipe = {
        'download_by': {'partition': 'table', 'table_names': ['boundaries']},
        'join_partitions_by': {'join_key_name': 'parcel_id_int'},
    }
    ing = _bare_ingester(recipe, admin_ids_to_save=[], verbose=False)

    ing._join_table_partitions()

    assert calls[0]['keep_original'] is False


def test_early_warnings_raises_without_table_partition():
    recipe = {
        'join_partitions_by': {'join_key_name': 'parcel_id_int'},
        'download_by': {'partition': 'year'},
    }
    ing = _bare_ingester(recipe)

    with pytest.raises(ValueError, match='join_partitions_by'):
        ing._early_warnings()


def test_early_warnings_raises_without_table_names():
    recipe = {
        'join_partitions_by': {'join_key_name': 'parcel_id_int'},
        'download_by': {'partition': 'table'},
    }
    ing = _bare_ingester(recipe)

    with pytest.raises(ValueError, match='join_partitions_by'):
        ing._early_warnings()


def test_early_warnings_passes_for_valid_join_partitions_by_config():
    recipe = {
        'join_partitions_by': {'join_key_name': 'parcel_id_int'},
        'download_by': {'partition': 'table', 'table_names': ['boundaries']},
    }
    ing = _bare_ingester(recipe)

    ing._early_warnings()  # does not raise


def test_early_warnings_no_op_without_join_partitions_by():
    ing = _bare_ingester({'download_by': {'partition': 'year'}})

    ing._early_warnings()  # does not raise


def test_placeslab_recipe_is_joined_table_partition():
    recipe = get_recipe_by_id('US_parcel-placeslab-fmv2026')
    ing = _bare_ingester(recipe)

    assert ing._is_joined_table_partition is True
    ing._early_warnings()  # the real recipe's config must itself be valid


def test_population_acs_recipe_is_not_joined_table_partition():
    """US_population-acs-2024 also uses `partition: table` (18 independently
    consumed ACS tables), but must NOT opt into auto-joining."""
    recipe = get_recipe_by_id('US_population-acs-2024')
    ing = _bare_ingester(recipe)

    assert ing._is_joined_table_partition is False


def _seed_table_partitions(recipe, keys):
    boundaries = gpd.GeoDataFrame(
        {'name': [f'parcel-{k}' for k in keys]},
        geometry=[box(i, i, i + 1, i + 1) for i in range(len(keys))],
        crs='EPSG:4326',
        index=pd.Index(keys, name='parcel_id_int'),
    )
    predictions = pd.DataFrame(
        {'lnusd_ha': [1.1 * (i + 1) for i in range(len(keys))]},
        index=pd.Index(keys, name='parcel_id_int'),
    )
    predictors = pd.DataFrame(
        {'slope': [2.0 * (i + 1) for i in range(len(keys))]},
        index=pd.Index(keys, name='parcel_id_int'),
    )
    for table_name, df in (
        ('boundaries', boundaries),
        ('predictions', predictions),
        ('predictors', predictors),
    ):
        save_parquet(df, get_output_path(recipe, ADMIN_ID, partition_id=table_name))


def test_ingest_auto_joins_table_partitions_end_to_end(cache_to_tmp):
    """Real `ingest()` call for the real recipe: with all 3 table partitions
    already on disk (standing in for a completed download/process loop, so no
    network access happens), the join fires automatically and the per-table
    partitions are cleaned up -- no manual `join_partitions_by_index` call."""
    recipe = get_recipe_by_id(RECIPE_ID)
    keys = ['k1', 'k2']
    _seed_table_partitions(recipe, keys)

    ingest(RECIPE_ID, admin_ids=[ADMIN_ID], verbose=False)

    out_path = get_output_path(recipe, ADMIN_ID)
    assert out_path.exists()
    result = read_parquet(out_path, geom=True)
    assert set(result['parcel_id_int']) == set(keys)
    assert set(result['lnusd_ha']) == {1.1, 2.2}
    assert set(result['slope']) == {2.0, 4.0}

    for table_name in ('boundaries', 'predictions', 'predictors'):
        assert not get_output_path(recipe, ADMIN_ID, partition_id=table_name).exists()


def test_ingest_second_call_is_a_no_op(cache_to_tmp, monkeypatch):
    """A second `ingest()` call without `reprocess` doesn't re-enter partition
    resolution/download for a county whose joined output already exists: the
    admin_id resolution excludes it upstream, so `_join_table_partitions` runs
    (the property gate only checks recipe shape) but delegates with an empty
    `admin_ids` list -- a safe no-op, not a special-cased skip."""
    recipe = get_recipe_by_id(RECIPE_ID)
    _seed_table_partitions(recipe, ['k1'])
    ingest(RECIPE_ID, admin_ids=[ADMIN_ID], verbose=False)

    calls = []
    monkeypatch.setattr(
        'openplaces.io.aggregate.join_partitions_by_index',
        lambda *a, **k: calls.append(k),
    )

    ingest(RECIPE_ID, admin_ids=[ADMIN_ID], verbose=False)

    assert calls == [
        {
            'table_names': ['boundaries', 'predictions', 'predictors'],
            'admin_ids': [],
            'join_key_name': 'parcel_id_int',
            'keep_original': False,
            'verbose': False,
        }
    ]
