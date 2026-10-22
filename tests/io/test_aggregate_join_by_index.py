"""Tests for `join_partitions_by_index`, the column-join counterpart to
`aggregate_partitions` (which only row-concatenates)."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces import cfg
from openplaces.io import read_parquet, save_parquet
from openplaces.io.aggregate import join_partitions_by_index
from openplaces.recipe import get_output_path, get_recipe_by_id

RECIPE_ID = 'US_parcel-placeslab-fmv2026'
ADMIN_ID = 'US-MA-MI'


@pytest.fixture
def cache_to_tmp(tmp_path, monkeypatch):
    """Redirect the 'cache' data dir so tests never touch real outputs."""
    monkeypatch.setitem(cfg.config['directories'], 'cache', tmp_path)
    return tmp_path


def _boundaries_df(keys):
    df = gpd.GeoDataFrame(
        {'name': [f'parcel-{k}' for k in keys]},
        geometry=[box(i, i, i + 1, i + 1) for i in range(len(keys))],
        crs='EPSG:4326',
        index=pd.Index(keys, name='geo_id_source'),
    )
    return df


def _attribute_df(keys, column, values):
    return pd.DataFrame({column: values}, index=pd.Index(keys, name='geo_id_source'))


def _write_partition(recipe, admin_id, partition_id, df):
    save_parquet(df, get_output_path(recipe, admin_id, partition_id=partition_id))


def test_joins_three_partitions_with_geo_id_and_provenance_column(cache_to_tmp):
    recipe = get_recipe_by_id(RECIPE_ID)
    keys = ['k1', 'k2']
    _write_partition(recipe, ADMIN_ID, 'boundaries', _boundaries_df(keys))
    _write_partition(
        recipe, ADMIN_ID, 'predictions', _attribute_df(keys, 'lnusd_ha', [1.1, 2.2])
    )
    _write_partition(
        recipe, ADMIN_ID, 'predictors', _attribute_df(keys, 'slope', [3.0, 4.0])
    )

    join_partitions_by_index(
        recipe,
        table_names=['boundaries', 'predictions', 'predictors'],
        admin_ids=ADMIN_ID,
        join_key_name='geo_id_source',
        verbose=False,
    )

    out_path = get_output_path(recipe, ADMIN_ID)
    assert out_path.exists()
    result = read_parquet(out_path, geom=True)

    assert result.index.name == 'parcel_id'
    assert 'geo_id' in result.columns
    assert set(result['geo_id_source']) == set(keys)
    assert set(result['lnusd_ha']) == {1.1, 2.2}
    assert set(result['slope']) == {3.0, 4.0}
    assert result.geometry.notna().all()

    # Per-table partitions deleted after a successful join (keep_original=False).
    for table_name in ('boundaries', 'predictions', 'predictors'):
        assert not get_output_path(recipe, ADMIN_ID, partition_id=table_name).exists()


def test_omitting_join_key_name_drops_the_original_index_column(cache_to_tmp):
    recipe = get_recipe_by_id(RECIPE_ID)
    keys = ['k1']
    _write_partition(recipe, ADMIN_ID, 'boundaries', _boundaries_df(keys))

    join_partitions_by_index(
        recipe, table_names=['boundaries'], admin_ids=ADMIN_ID, verbose=False
    )

    result = read_parquet(get_output_path(recipe, ADMIN_ID), geom=True)
    assert 'geo_id_source' not in result.columns


def test_keep_original_retains_partition_files(cache_to_tmp):
    recipe = get_recipe_by_id(RECIPE_ID)
    keys = ['k1']
    _write_partition(recipe, ADMIN_ID, 'boundaries', _boundaries_df(keys))

    join_partitions_by_index(
        recipe,
        table_names=['boundaries'],
        admin_ids=ADMIN_ID,
        keep_original=True,
        verbose=False,
    )

    assert get_output_path(recipe, ADMIN_ID, partition_id='boundaries').exists()


def test_missing_base_table_skips_admin_unit(cache_to_tmp, capsys):
    recipe = get_recipe_by_id(RECIPE_ID)
    _write_partition(recipe, ADMIN_ID, 'predictions', _attribute_df(['k1'], 'x', [1.0]))

    join_partitions_by_index(
        recipe,
        table_names=['boundaries', 'predictions'],
        admin_ids=ADMIN_ID,
        verbose=True,
    )

    assert not get_output_path(recipe, ADMIN_ID).exists()
    assert 'boundaries' in capsys.readouterr().out


def test_sidecar_stands_in_for_a_missing_base_table(cache_to_tmp):
    """A unit delivered with a crosswalk instead of boundaries is joined
    without geometry, keyed by the join key; the sidecar is kept."""
    recipe = get_recipe_by_id(RECIPE_ID)
    keys = ['k1', 'k2']
    _write_partition(
        recipe, ADMIN_ID, 'predictions', _attribute_df(keys, 'lnusd_ha', [1.1, 2.2])
    )
    _write_partition(
        recipe, ADMIN_ID, 'predictors', _attribute_df(keys, 'slope', [3.0, 4.0])
    )
    crosswalk = pd.DataFrame(
        {'parcel_id_new': ['new_1', 'new_1'], 'reference_parcel_id': keys}
    )
    crosswalk_path = get_output_path(recipe, ADMIN_ID, partition_id='crosswalk')
    crosswalk_path.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_parquet(crosswalk_path)
    before = crosswalk_path.read_bytes()

    written = join_partitions_by_index(
        recipe,
        table_names=['boundaries', 'predictions', 'predictors', 'crosswalk'],
        admin_ids=ADMIN_ID,
        join_key_name='geo_id_source',
        sidecar_tables=['crosswalk'],
        verbose=False,
    )

    assert written == [ADMIN_ID]
    out_path = get_output_path(recipe, ADMIN_ID)
    assert not out_path.with_stem(out_path.stem + '_geo').exists()
    result = read_parquet(out_path)
    assert result.index.name == 'parcel_id'
    assert list(result.index) == keys
    assert list(result['geo_id_source']) == keys
    assert list(result['slope']) == [3.0, 4.0]
    assert 'geo_id' not in result.columns
    assert crosswalk_path.read_bytes() == before
    for table_name in ('predictions', 'predictors'):
        assert not get_output_path(recipe, ADMIN_ID, partition_id=table_name).exists()


def test_base_table_wins_and_sidecar_is_not_joined(cache_to_tmp):
    """A unit with boundaries joins as before; its sidecar stays apart."""
    recipe = get_recipe_by_id(RECIPE_ID)
    keys = ['k1']
    _write_partition(recipe, ADMIN_ID, 'boundaries', _boundaries_df(keys))
    _write_partition(recipe, ADMIN_ID, 'crosswalk', _attribute_df(keys, 'pair', [1]))

    join_partitions_by_index(
        recipe,
        table_names=['boundaries', 'crosswalk'],
        admin_ids=ADMIN_ID,
        sidecar_tables=['crosswalk'],
        verbose=False,
    )

    result = read_parquet(get_output_path(recipe, ADMIN_ID), geom=True)
    assert 'pair' not in result.columns
    assert result.geometry.notna().all()
    assert get_output_path(recipe, ADMIN_ID, partition_id='crosswalk').exists()


def test_unknown_sidecar_table_raises(cache_to_tmp):
    with pytest.raises(ValueError, match='sidecar_tables'):
        join_partitions_by_index(
            get_recipe_by_id(RECIPE_ID),
            table_names=['boundaries'],
            admin_ids=ADMIN_ID,
            sidecar_tables=['crosswalk'],
        )


def test_missing_attribute_table_still_joins_the_rest(cache_to_tmp, capsys):
    recipe = get_recipe_by_id(RECIPE_ID)
    keys = ['k1']
    _write_partition(recipe, ADMIN_ID, 'boundaries', _boundaries_df(keys))
    _write_partition(
        recipe, ADMIN_ID, 'predictors', _attribute_df(keys, 'slope', [3.0])
    )
    # 'predictions' never written for this admin unit.

    join_partitions_by_index(
        recipe,
        table_names=['boundaries', 'predictions', 'predictors'],
        admin_ids=ADMIN_ID,
        verbose=True,
    )

    result = read_parquet(get_output_path(recipe, ADMIN_ID), geom=True)
    assert 'slope' in result.columns
    assert 'predictions' in capsys.readouterr().out


def test_column_name_collision_raises(cache_to_tmp):
    recipe = get_recipe_by_id(RECIPE_ID)
    keys = ['k1']
    boundaries = _boundaries_df(keys)
    boundaries['dup'] = [1]
    _write_partition(recipe, ADMIN_ID, 'boundaries', boundaries)
    _write_partition(recipe, ADMIN_ID, 'predictions', _attribute_df(keys, 'dup', [2]))

    with pytest.raises(ValueError, match='dup'):
        join_partitions_by_index(
            recipe,
            table_names=['boundaries', 'predictions'],
            admin_ids=ADMIN_ID,
            verbose=False,
        )
