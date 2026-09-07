"""Tests for `aggregate_partitions`, the row-wise partition roll-up.

The regression these cover: a roll-up whose group key equals one of its own
partition ids resolved its output to a path that was also an input, rewrote
that file from itself, and then deleted it as an original.
"""

import pandas as pd
import pytest

from openplaces import cfg
from openplaces.io import read_parquet, save_parquet
from openplaces.io.aggregate import aggregate_partitions
from openplaces.recipe import get_output_path, get_recipe_by_id

# A year-partitioned recipe: its partition ids are years, so a by='year'
# roll-up group key is identical to the partition id it groups.
RECIPE_ID = 'US-NC-NHA_transaction-nhcgov-2026'
ADMIN_ID = 'US-NC-NHA'


@pytest.fixture
def cache_to_tmp(tmp_path, monkeypatch):
    """Redirect the 'cache' data dir so tests never touch real outputs."""
    monkeypatch.setitem(cfg.config['directories'], 'cache', tmp_path)
    return tmp_path


def _write_partition(recipe, partition_id, values):
    path = get_output_path(recipe, ADMIN_ID, partition_id=partition_id)
    save_parquet(pd.DataFrame({'sale_price': values}), path)
    return path


def test_by_year_rollup_of_year_partitions_refuses_instead_of_deleting(cache_to_tmp):
    recipe = get_recipe_by_id(RECIPE_ID)
    path_2021 = _write_partition(recipe, '2021', [100, 200])

    with pytest.raises(ValueError, match='into itself'):
        aggregate_partitions(
            recipe, by='year', admin_ids=ADMIN_ID, partition_ids=['2021']
        )

    # The partition file must still be there, with its rows intact.
    assert path_2021.exists()
    assert list(read_parquet(path_2021)['sale_price']) == [100, 200]


def test_single_file_rollup_of_year_partitions_still_works(cache_to_tmp):
    recipe = get_recipe_by_id(RECIPE_ID)
    path_2021 = _write_partition(recipe, '2021', [100])
    path_2022 = _write_partition(recipe, '2022', [200])

    aggregate_partitions(
        recipe,
        single_file=True,
        admin_ids=ADMIN_ID,
        partition_ids=['2021', '2022'],
    )

    all_path = get_output_path(recipe, ADMIN_ID, partition_id='all')
    assert sorted(read_parquet(all_path)['sale_price']) == [100, 200]
    assert not path_2021.exists()
    assert not path_2022.exists()


def test_partition_discovery_skips_sidecars_and_previous_rollups(cache_to_tmp):
    from openplaces.io.aggregate import _existing_partition_ids

    recipe = get_recipe_by_id(RECIPE_ID)
    partition_path = _write_partition(recipe, '2021', [100])
    _write_partition(recipe, '2022', [200])
    # A geometry sidecar and a simplified sidecar beside a partition.
    for suffix in ('_geo', '_geo_simplified'):
        partition_path.with_stem(partition_path.stem + suffix).touch()
    # A previous roll-up, recognizable by the coverage it records.
    rollup_path = get_output_path(recipe, ADMIN_ID, partition_id='2019')
    save_parquet(
        pd.DataFrame({'sale_price': [1]}),
        rollup_path,
        file_metadata={'openplaces:partitions': '["2019"]'},
    )

    assert _existing_partition_ids(recipe, ADMIN_ID) == ['2021', '2022']
