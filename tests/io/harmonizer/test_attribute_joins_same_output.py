"""add_source_evidence' joins give what pandas' call per group gave.

plans/slow-steps-same-output-faster.md, R1: the point reference's text
columns and collected ids, the polygon reference's `parcel_id_all`, and
`_join_distinct`'s ' + ' summaries. The previous expressions are kept
below; old and new run on the same fabricated crosswalks and are
compared exactly (values, value types, dtype, index, column order).
The joined columns take the text dtype ('str') pandas infers from the
joined strings; the tests pin that the new code arrives at the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openplaces.io.harmonizer.attributes import _shared
from openplaces.io.harmonizer.attributes.point import _joined_text_columns
from openplaces.table import join_distinct_text, reduce_groups


def _old_joined_text_columns(crosswalk, key, columns):
    def _join_unique(s: pd.Series) -> str | None:
        seen = dict.fromkeys(str(v) for v in s if pd.notna(v))
        return '; '.join(seen) if seen else None

    return crosswalk.groupby(key)[columns].agg(_join_unique)


def _old_ids(frame, key, column):
    return frame.groupby(key)[column].agg(
        lambda s: '|'.join(s.dropna().astype(str).unique())
    )


def _old_join_distinct(areas, spine_id_col, col):
    grouped = areas.groupby(spine_id_col)[col]
    joined = grouped.apply(' + '.join)
    return joined.where(grouped.nunique() > 1)


def _types(series):
    return [type(v) for v in series.to_numpy(dtype=object)]


def _same_frame(new, old):
    pd.testing.assert_frame_equal(new, old, check_exact=True, check_dtype=True)
    for column in old.columns:
        assert _types(new[column]) == _types(old[column]), column


def _same_series(new, old):
    pd.testing.assert_series_equal(new, old, check_exact=True, check_dtype=True)
    pd.testing.assert_index_equal(new.index, old.index, exact=True)
    assert _types(new) == _types(old)


# b1 one point; b2 two points; b3 the same values twice; b4 nothing
# stated; an unlinked point; b5 an empty text.
CROSSWALK = {
    'building_id': ['b1', 'b2', 'b2', 'b3', 'b3', 'b4', None, 'b5'],
    'source': ['nsi', 'nsi', 'esri', 'nsi', 'nsi', None, 'nsi', ''],
    'foundation_type': ['S', 'C', 'C', None, 'B', None, 'S', 'S'],
}


@pytest.mark.parametrize('dtype', [object, 'str', 'string', 'category'])
@pytest.mark.parametrize('key_dtype', [object, 'str'])
def test_text_columns(dtype, key_dtype):
    crosswalk = pd.DataFrame(CROSSWALK).astype(
        {'source': dtype, 'foundation_type': dtype, 'building_id': key_dtype}
    )
    columns = ['source', 'foundation_type']
    _same_frame(
        _joined_text_columns(crosswalk, 'building_id', columns),
        _old_joined_text_columns(crosswalk, 'building_id', columns),
    )


def test_text_columns_all_missing_and_empty():
    crosswalk = pd.DataFrame(
        {'building_id': ['b1', 'b1', 'b2'], 'source': [None, None, None]}
    ).astype({'source': object})
    _same_frame(
        _joined_text_columns(crosswalk, 'building_id', ['source']),
        _old_joined_text_columns(crosswalk, 'building_id', ['source']),
    )
    empty = pd.DataFrame(
        {
            'building_id': pd.Series([], dtype='str'),
            'source': pd.Series([], dtype='str'),
        }
    )
    _same_frame(
        _joined_text_columns(empty, 'building_id', ['source']),
        _old_joined_text_columns(empty, 'building_id', ['source']),
    )


@pytest.mark.parametrize('dtype', [object, 'str', 'string', 'category'])
def test_collected_ids(dtype):
    frame = pd.DataFrame(
        {
            'building_id': ['b1', 'b2', 'b2', 'b3', 'b3', 'b4', None],
            'nsi_id': pd.Series(
                ['n1', 'n2', 'n9', 'n4', 'n4', None, 'n7'], dtype=dtype
            ),
        }
    )
    _same_series(
        reduce_groups(frame['nsi_id'], frame['building_id'], join_distinct_text),
        _old_ids(frame, 'building_id', 'nsi_id'),
    )


def _random_crosswalk(n_rows, seed=13):
    rng = np.random.default_rng(seed)
    sizes = rng.choice([1, 1, 1, 1, 2, 2, 3, 5], size=n_rows)
    keys = np.repeat(np.arange(len(sizes)), sizes)[:n_rows]
    order = rng.permutation(n_rows)
    source = rng.choice(np.array(['nsi', 'esri', 'hifld', None], dtype=object), n_rows)
    foundation = rng.choice(np.array(['S', 'C', 'B', 'P', None], dtype=object), n_rows)
    parcel = pd.Series(keys + rng.integers(0, 3, n_rows)).map('p{:05d}'.format)
    return pd.DataFrame(
        {
            'building_id': pd.Series(keys[order]).map('b{:05d}'.format),
            'source': pd.Series(source[order], dtype='str'),
            'foundation_type': pd.Series(foundation[order], dtype='str'),
            'parcel_id': pd.Series(parcel.to_numpy()[order], dtype='str'),
        }
    )


def test_random_crosswalk():
    crosswalk = _random_crosswalk(4000)
    columns = ['source', 'foundation_type']
    _same_frame(
        _joined_text_columns(crosswalk, 'building_id', columns),
        _old_joined_text_columns(crosswalk, 'building_id', columns),
    )
    _same_series(
        reduce_groups(
            crosswalk['parcel_id'], crosswalk['building_id'], join_distinct_text
        ),
        _old_ids(crosswalk, 'building_id', 'parcel_id'),
    )


# _join_distinct reads the per-(entity, value) area table, unique pairs.
AREAS = {
    'footprint_id': ['f1', 'f2', 'f2', 'f3', 'f4', 'f4', 'f4'],
    'use_group_combined': ['res', 'res', 'com', 'ind', 'res', 'com', 'ind'],
}


@pytest.mark.parametrize('dtype', [object, 'str', 'category'])
def test_join_distinct(dtype):
    areas = pd.DataFrame(AREAS).astype({'use_group_combined': dtype})
    _same_series(
        _shared._join_distinct(areas, 'footprint_id', 'use_group_combined'),
        _old_join_distinct(areas, 'footprint_id', 'use_group_combined'),
    )


@pytest.mark.parametrize('dtype', [object, 'str', 'category'])
def test_join_distinct_without_any_group_of_several(dtype):
    areas = pd.DataFrame(
        {'footprint_id': ['f1', 'f2'], 'use_group_combined': ['res', 'com']}
    ).astype({'use_group_combined': dtype})
    _same_series(
        _shared._join_distinct(areas, 'footprint_id', 'use_group_combined'),
        _old_join_distinct(areas, 'footprint_id', 'use_group_combined'),
    )


def test_join_distinct_empty():
    areas = pd.DataFrame(
        {
            'footprint_id': pd.Series([], dtype='str'),
            'use_group_combined': pd.Series([], dtype='str'),
        }
    )
    _same_series(
        _shared._join_distinct(areas, 'footprint_id', 'use_group_combined'),
        _old_join_distinct(areas, 'footprint_id', 'use_group_combined'),
    )


def test_join_distinct_random():
    crosswalk = _random_crosswalk(4000, seed=17).dropna(subset=['source'])
    areas = crosswalk[['building_id', 'source']].drop_duplicates()
    _same_series(
        _shared._join_distinct(areas, 'building_id', 'source'),
        _old_join_distinct(areas, 'building_id', 'source'),
    )
