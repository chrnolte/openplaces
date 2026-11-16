"""A property spine cell names the source that supplied it.

The spine's `source` labels a row with the roll it came from. A
supplement (a table declaring `supplements:` for that roll) fills or
overwrites cells of the row, and two sources' rows sharing an account
number merge into one. In both cases the cell's `{column}_source` names
the source that supplied it, and an empty sidecar keeps meaning "the
row's `source`". Values, ids and source names are fabricated.
"""

import pandas as pd

import openplaces.io.harmonizer.links as links
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.entity_ids import merge_on_id
from tests.links_patching import patch_links

_ROLL = 'US-XX-YY_property-openroll-2026'
_CITY = 'US-XX-YY_property-cityroll-2026'
_KEY = 'parcel_id_assessor'


def _state(spine):
    state = HarmonizeState(
        recipe={}, admin_id=AdminId('US-XX-YY'), verbose=False, timer=None, spine=spine
    )
    state.metadata['spine_source_recipe_ids'] = {_ROLL}
    return state


def _join(monkeypatch, spine, city, track_provenance='all'):
    match = {
        'recipe_id': _CITY,
        'layer': None,
        'key': _KEY,
        'aggregation_function': None,
        'supplements': _ROLL,
        'supplements_key': _KEY,
    }
    patch_links(monkeypatch, '_discover_link_sources', lambda *a, **k: [match])
    patch_links(monkeypatch, 'get_entities', lambda *a, **k: city)
    patch_links(monkeypatch, 'restrict_to_admin_by_name', lambda df, *a: df)
    patch_links(monkeypatch, '_apply_remap_csvs', lambda state, recipe_id: state)
    return links.link_by_id(
        _state(spine),
        auto_discover=True,
        entity_type='property',
        supplements_only=True,
        count_as=False,
        columns=['year_built', 'n_bedrooms'],
        track_provenance=track_provenance,
    ).spine


def _spine():
    return pd.DataFrame(
        {
            _KEY: ['101', '102', '103'],
            'year_built': [1950.0, 1960.0, 1970.0],
            'n_bedrooms': [None, None, 2.0],
            'source': ['openroll'] * 3,
        },
        index=pd.Index(['p1', 'p2', 'p3'], name='property_id'),
    )


def test_tracking_all_names_every_column_the_join_writes():
    pairs = [('beds', 'n_bedrooms'), ('yr', 'year_built')]
    assert links.by_id._tracked_columns('all', pairs) == {'n_bedrooms', 'year_built'}
    assert links.by_id._tracked_columns(['year_built'], pairs) == {'year_built'}
    assert links.by_id._tracked_columns(None, pairs) == set()


def test_a_supplement_names_the_cells_it_fills_and_overwrites(monkeypatch):
    # The city table covers two of three rows: a majority, so it
    # overwrites year_built where it states one (an equal value too:
    # the cell was taken from it), and fills bedrooms.
    city = pd.DataFrame(
        {
            _KEY: ['101', '102'],
            'year_built': [1955.0, 1960.0],
            'n_bedrooms': [3.0, 4.0],
        }
    )
    spine = _join(monkeypatch, _spine(), city)

    assert spine['year_built'].tolist() == [1955.0, 1960.0, 1970.0]
    assert spine['year_built_source'].tolist()[:2] == ['cityroll', 'cityroll']
    # The roll's own values carry no label: empty means the row's source.
    assert pd.isna(spine.loc['p3', 'year_built_source'])
    assert spine['n_bedrooms_source'].tolist()[:2] == ['cityroll', 'cityroll']
    assert pd.isna(spine.loc['p3', 'n_bedrooms_source'])
    # The row still names its roll.
    assert set(spine['source']) == {'openroll'}


def test_without_tracking_the_join_writes_no_label(monkeypatch):
    city = pd.DataFrame({_KEY: ['101'], 'year_built': [1955.0], 'n_bedrooms': [3.0]})
    spine = _join(monkeypatch, _spine(), city, track_provenance=None)
    assert not [c for c in spine.columns if c.endswith('_source')]


def _merge(rows):
    ids = pd.Series(rows.pop('id').tolist())
    return merge_on_id(rows, ids, 'property_id')


def test_a_merged_cell_keeps_the_label_of_the_row_it_came_from():
    rows = pd.DataFrame(
        {
            'id': ['A', 'A', 'B'],
            'n_rooms': [5.0, 9.0, 6.0],
            'n_bedrooms': [None, 3.0, 2.0],
            # The second row's bedrooms came from a supplement of its roll.
            'n_bedrooms_source': [None, 'cityroll', None],
            'source': ['county', 'state', 'state'],
        }
    )
    merged = _merge(rows)

    assert merged.loc['A', 'source'] == 'county+state'
    assert merged.loc['A', 'n_rooms'] == 5.0
    assert merged.loc['A', 'n_rooms_source'] == 'county'
    assert merged.loc['A', 'n_bedrooms'] == 3.0
    assert merged.loc['A', 'n_bedrooms_source'] == 'cityroll'
    # A row nothing merged into keeps its own sidecars.
    assert pd.isna(merged.loc['B', 'n_rooms_source'])
    assert pd.isna(merged.loc['B', 'n_bedrooms_source'])


def test_a_sidecar_follows_its_cell_not_the_first_label_stated():
    # Two copies of one account from one roll; only the second had a
    # supplement fill its year. The first row's year wins the cell, so
    # the supplement's label must not ride along with it.
    rows = pd.DataFrame(
        {
            'id': ['A', 'A'],
            'year_built': [1950.0, 1960.0],
            'year_built_source': [None, 'cityroll'],
            'source': ['openroll', 'openroll'],
        }
    )
    merged = _merge(rows)

    assert merged.loc['A', 'year_built'] == 1950.0
    assert pd.isna(merged.loc['A', 'year_built_source'])
