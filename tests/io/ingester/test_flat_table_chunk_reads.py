"""A flat source file chunked by `process_by.admin_id_column` is parsed once
per download partition, not once per admin unit, and a duplicated crosswalk
position never writes the same source row twice.

Every admin unit of a statewide CSV/xlsx/fixed-width roll re-entered the read
with the same path and the same columns, so the whole file was parsed again for
each of them. The row set each unit keeps is a positional slice of that one
parse.
"""

from pathlib import Path

import pandas as pd

from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.timing import Timer


def _make_roll(tmp_path):
    """A fabricated three-county roll: two rows per county."""
    path = tmp_path / 'roll.csv'
    pd.DataFrame(
        {
            'county': ['AA', 'AA', 'BB', 'BB', 'CC', 'CC'],
            'parcel': [1, 2, 3, 4, 5, 6],
        }
    ).to_csv(path, index=False)
    return path


def _ingester(path, partition, counter):
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = {'entity': None, 'dataset': 'roll'}
    ingester.download_partition = partition
    ingester.processing_chunk = {}
    ingester.recipe_heap_dir = path.parent
    ingester.timer = Timer('test')
    ingester.verbose = False

    original = ingester._read_flat_table

    def counted(*args, **kwargs):
        counter.append(1)
        return original(*args, **kwargs)

    ingester._read_flat_table = counted
    return ingester


def test_the_file_is_parsed_once_for_all_admin_chunks(tmp_path):
    path = _make_roll(tmp_path)
    partition = {'data_path': path}
    counter = []
    ingester = _ingester(path, partition, counter)

    first = ingester._read_recipe_data(fids=[0, 1])
    second = ingester._read_recipe_data(fids=[2, 3])
    third = ingester._read_recipe_data(fids=[4, 5])

    assert len(counter) == 1
    assert first['parcel'].tolist() == [1, 2]
    assert second['parcel'].tolist() == [3, 4]
    assert third['parcel'].tolist() == [5, 6]


def test_a_repeated_position_is_read_once(tmp_path):
    path = _make_roll(tmp_path)
    partition = {'data_path': path}
    ingester = _ingester(path, partition, [])

    gdf = ingester._read_recipe_data(fids=[2, 3, 3])

    assert gdf['parcel'].tolist() == [3, 4]


def test_each_chunk_gets_its_own_frame_to_mutate(tmp_path):
    path = _make_roll(tmp_path)
    partition = {'data_path': path}
    ingester = _ingester(path, partition, [])

    first = ingester._read_recipe_data(fids=[0, 1])
    first['parcel'] = 'overwritten'
    second = ingester._read_recipe_data(fids=[0, 1])

    assert second['parcel'].tolist() == [1, 2]


def test_a_different_file_is_parsed_again(tmp_path):
    path = _make_roll(tmp_path)
    other = tmp_path / 'other.csv'
    Path(other).write_text('county,parcel\nDD,9\n', encoding='utf-8')
    partition = {'data_path': path}
    counter = []
    ingester = _ingester(path, partition, counter)

    ingester._read_recipe_data(fids=[0])
    partition['data_path'] = other
    gdf = ingester._read_recipe_data(fids=[0])

    assert len(counter) == 2
    assert gdf['parcel'].tolist() == [9]
