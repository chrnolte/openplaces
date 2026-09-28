"""Key dtypes when a crosswalk table is read.

A county or class code is text with meaningful leading zeros. Read as a
number, '037' becomes 37, matches nothing in the column it maps, and the
all-NaN result raises nothing.
"""

import pandas as pd
import pytest

from openplaces.io import transform
from openplaces.io.transform import (
    _apply_remap_file,
    _read_crosswalk_table,
    get_crosswalk,
)


def _write_csv(path, rows):
    path.write_text('code,value\n' + '\n'.join(rows) + '\n', encoding='utf-8')
    return path


def test_zero_padded_keys_survive_as_text(tmp_path):
    path = _write_csv(tmp_path / 'remap.csv', ['037,Alleghany', '119,Mecklenburg'])

    result = _apply_remap_file(pd.Series(['037', '119']), str(path))

    assert result.tolist() == ['Alleghany', 'Mecklenburg']


def test_plain_numeric_keys_keep_their_numeric_values(tmp_path):
    # No leading zeros: the table is left exactly as pandas typed it, so a
    # crosswalk carrying numbers still returns numbers.
    path = _write_csv(tmp_path / 'remap.csv', ['37,1500', '119,2400'])

    table = _read_crosswalk_table(lambda dtype: pd.read_csv(path, dtype=dtype))

    assert table['code'].tolist() == [37, 119]
    assert table['value'].tolist() == [1500, 2400]


def _admin_layer(monkeypatch, keys):
    """An admin layer as `get_admin` returns it: the spine, outer-joined."""
    frame = pd.DataFrame(
        {'admin2_id_wikidata': keys},
        index=pd.Index([f'BW-{i:02d}' for i in range(len(keys))], name='admin2_id'),
    )
    monkeypatch.setattr(transform, 'get_admin', lambda *a, **k: frame)
    return {
        'admin_id': 'BW',
        'admin_level': 2,
        'admin_id_column': 'admin2_id_wikidata',
        'admin_recipe_id': 'admin-wikidata-2026_admin2',
    }


def test_units_the_layer_does_not_cover_are_dropped_not_collided(monkeypatch):
    # `get_admin` outer-joins the named layer onto the spine, so every
    # unit it does not cover arrives with an empty key. While the world
    # moves off its old source a country at a time that is most of them,
    # and Botswana's six not-yet-replaced districts aborted its level-3
    # ingest as a duplicate index.
    crosswalk_dict = _admin_layer(monkeypatch, ['Q1', '', 'Q2', '', '', ''])

    crosswalk = get_crosswalk(crosswalk_dict, flip=True)

    assert sorted(crosswalk.index) == ['Q1', 'Q2']


def test_a_real_duplicate_key_is_still_an_error(monkeypatch):
    crosswalk_dict = _admin_layer(monkeypatch, ['Q1', 'Q1', ''])

    with pytest.raises(ValueError, match='duplicated indices'):
        get_crosswalk(crosswalk_dict, flip=True)
