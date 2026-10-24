"""Year-partitioned roll-ups that stream and replace years exactly.

A Florida DOR roll is one file per county holding every year's rows.
Writing it held every year in one frame (US-FL-LA peaked at 38.9 GB),
and a re-run of some years was refused because a full-row union would
keep each changed old row beside its new version. With
`aggregate_by: partition_column`, the years of a run replace exactly
their own rows, one year in memory at a time. All values here are
fabricated.
"""

import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

import openplaces.io.aggregate as aggregate_module
from openplaces import cfg
from openplaces.io import read_parquet, save_parquet
from openplaces.io.aggregate import (
    aggregate_partitions,
    read_file_metadata,
    read_partition_coverage,
)
from openplaces.path import RECIPE_ROOTS_METADATA_KEY
from openplaces.recipe import get_output_path, get_recipe_by_id

RECIPE_ID = 'US-FL_property-fldor-2026'
ADMIN_ID = 'US-FL-UN'


@pytest.fixture
def cache_to_tmp(tmp_path, monkeypatch):
    """Redirect the 'cache' data dir so tests never touch real outputs."""
    monkeypatch.setitem(cfg.config['directories'], 'cache', tmp_path)
    return tmp_path


@pytest.fixture
def recipe():
    return get_recipe_by_id(RECIPE_ID)


def _year_frame(year, n, value=100):
    """A fabricated roll vintage of *n* rows for *year*."""
    return pd.DataFrame(
        {
            'parcel_id_assessor': [f'P-{i:04d}' for i in range(n)],
            'tax_year': [year] * n,
            'market_value': [value + i for i in range(n)],
            'roll_type': pd.Categorical(['R'] * n),
            'note': [None] * n if year == 2010 else ['x'] * n,
        }
    )


def _write_year(recipe, year, n, value=100):
    path = get_output_path(recipe, ADMIN_ID, partition_id=str(year))
    save_parquet(_year_frame(year, n, value), path)
    return path


def _all_path(recipe):
    return get_output_path(recipe, ADMIN_ID, partition_id='all')


def _roll_up(recipe, **kwargs):
    aggregate_partitions(
        recipe,
        single_file=True,
        admin_ids=ADMIN_ID,
        partition_column='tax_year',
        **kwargs,
    )


def test_full_run_streams_in_year_order_with_footer(cache_to_tmp, recipe, capsys):
    for year, n in ((2010, 3), (2011, 4), (2012, 2)):
        _write_year(recipe, year, n)

    _roll_up(recipe, verbose=True)

    assert 'streamed' in capsys.readouterr().out
    final = _all_path(recipe)
    result = read_parquet(final)
    assert result['tax_year'].tolist() == [2010] * 3 + [2011] * 4 + [2012] * 2
    assert read_partition_coverage(final) == {'2010', '2011', '2012'}
    assert RECIPE_ROOTS_METADATA_KEY in read_file_metadata(final)
    leftovers = sorted(p.name for p in final.parent.glob('*.parquet'))
    assert leftovers == [final.name]


def test_streamed_full_run_matches_the_in_memory_file(
    cache_to_tmp, recipe, monkeypatch
):
    """Same columns, values and order as the in-memory path writes.

    Arrow may store a text column as `string` on one path and
    `large_string` on the other; both read back as the same values.
    """
    years = ((2010, 3), (2011, 4), (2012, 2))
    for year, n in years:
        _write_year(recipe, year, n)
    _roll_up(recipe, keep_original=True)
    final = _all_path(recipe)
    streamed = pq.read_table(final)
    final.unlink()

    monkeypatch.setattr(aggregate_module, '_can_stream_aggregate', lambda **k: False)
    _roll_up(recipe)
    in_memory = pq.read_table(final)

    assert streamed.column_names == in_memory.column_names
    pd.testing.assert_frame_equal(streamed.to_pandas(), in_memory.to_pandas())


def test_type_drift_across_vintages_still_streams(
    cache_to_tmp, recipe, monkeypatch, capsys
):
    """A column read as float in one vintage and text in another streams.

    Every Florida roll has one: a CSV vintage whose address cells are
    all empty is read as float, so the vintages' schemas did not unify
    and every county fell back to holding all its years in one frame.
    The text written must be the in-memory path's, float cells included.
    """
    frames = {
        2010: pd.DataFrame(
            {
                'tax_year': [2010, 2010],
                'address': [float('nan'), float('nan')],
                'land_value': [1, 2],
                'note': [1.0, float('nan')],
            }
        ),
        2011: pd.DataFrame(
            {
                'tax_year': [2011, 2011],
                'address': ['1 Fabricated Way', None],
                'land_value': [3.5, None],
                'note': ['x', 'y'],
            }
        ),
    }
    for year, df in frames.items():
        save_parquet(df, get_output_path(recipe, ADMIN_ID, partition_id=str(year)))
    _roll_up(recipe, keep_original=True, verbose=True)
    assert 'streamed' in capsys.readouterr().out
    final = _all_path(recipe)
    streamed = read_parquet(final)
    final.unlink()

    monkeypatch.setattr(aggregate_module, '_can_stream_aggregate', lambda **k: False)
    _roll_up(recipe)
    in_memory = read_parquet(final)

    assert streamed['note'].tolist()[:1] == ['1.0']
    pd.testing.assert_frame_equal(
        streamed.astype(object).where(streamed.notna(), None),
        in_memory.astype(object).where(in_memory.notna(), None),
    )


def test_partial_rerun_replaces_two_years_exactly(cache_to_tmp, recipe):
    for year, n in ((2010, 3), (2011, 4), (2012, 2), (2013, 3)):
        _write_year(recipe, year, n)
    _roll_up(recipe)
    final = _all_path(recipe)
    before = read_parquet(final)
    before_schema = pq.read_schema(final)

    # Re-run 2011 and 2012: one more row in 2011, one fewer in 2012,
    # and every value changed, so a full-row union would keep the old
    # rows beside the new ones.
    _write_year(recipe, 2011, 5, value=900)
    _write_year(recipe, 2012, 1, value=900)
    _roll_up(recipe)

    after = read_parquet(final)
    assert after.groupby('tax_year').size().to_dict() == {
        2010: 3,
        2011: 5,
        2012: 1,
        2013: 3,
    }
    assert not after.duplicated(['parcel_id_assessor', 'tax_year']).any()
    assert after.loc[after['tax_year'].isin([2011, 2012]), 'market_value'].min() == 900
    kept = after[after['tax_year'].isin([2010, 2013])].reset_index(drop=True)
    old = before[before['tax_year'].isin([2010, 2013])].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, old)
    assert read_partition_coverage(final) == {'2010', '2011', '2012', '2013'}
    assert RECIPE_ROOTS_METADATA_KEY in read_file_metadata(final)
    assert pq.read_schema(final).equals(before_schema, check_metadata=False)
    join_ids = pq.read_table(final, columns=['_join_id'])['_join_id'].to_pylist()
    assert join_ids == list(range(1, len(after) + 1))
    assert not list(final.parent.glob('*.split'))
    leftovers = sorted(p.name for p in final.parent.glob('*.parquet'))
    assert leftovers == [final.name]


def test_partial_rerun_matches_a_full_run_of_the_same_years(
    cache_to_tmp, recipe, tmp_path_factory, monkeypatch
):
    """Row order after a replacement is the order a full run writes."""
    for year, n in ((2010, 3), (2011, 4), (2012, 2)):
        _write_year(recipe, year, n)
    _roll_up(recipe)
    _write_year(recipe, 2011, 2, value=500)
    _roll_up(recipe)
    replaced = pq.read_table(_all_path(recipe))

    fresh_root = tmp_path_factory.mktemp('fresh')
    monkeypatch.setitem(cfg.config['directories'], 'cache', fresh_root)
    _write_year(recipe, 2010, 3)
    _write_year(recipe, 2011, 2, value=500)
    _write_year(recipe, 2012, 2)
    _roll_up(recipe)
    fresh = pq.read_table(_all_path(recipe))

    assert replaced.equals(fresh)


def test_a_new_year_is_added_in_year_order(cache_to_tmp, recipe):
    _write_year(recipe, 2011, 2)
    _write_year(recipe, 2012, 2)
    _roll_up(recipe)
    _write_year(recipe, 2010, 1)
    _roll_up(recipe)

    final = _all_path(recipe)
    assert read_parquet(final)['tax_year'].tolist() == [2010, 2011, 2011, 2012, 2012]
    assert read_partition_coverage(final) == {'2010', '2011', '2012'}


def test_a_row_naming_another_year_is_refused(cache_to_tmp, recipe):
    _write_year(recipe, 2010, 2)
    _write_year(recipe, 2011, 2)
    _roll_up(recipe)
    final = _all_path(recipe)
    before = final.read_bytes()

    bad = _year_frame(2011, 2)
    bad.loc[1, 'tax_year'] = 2010
    save_parquet(bad, get_output_path(recipe, ADMIN_ID, partition_id='2011'))
    with pytest.raises(ValueError, match='names another partition'):
        _roll_up(recipe)

    assert final.read_bytes() == before


def test_without_a_partition_column_union_is_unchanged(cache_to_tmp, recipe):
    """A recipe that names no column keeps the full-row union."""
    _write_year(recipe, 2010, 2)
    aggregate_partitions(recipe, single_file=True, admin_ids=ADMIN_ID)
    _write_year(recipe, 2010, 2)
    aggregate_partitions(recipe, single_file=True, admin_ids=ADMIN_ID)

    final = _all_path(recipe)
    assert len(read_parquet(final)) == 2
    assert json.loads(read_file_metadata(final)['openplaces:partitions']) == ['2010']
