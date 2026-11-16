"""A gathered value names the property rows that supplied it.

`aggregate_from_entities` labels a curated cell with the tokens of the
reference rows its value came from: for `max`, `min` and `first` the
row holding the value (rows of several sources tied on it each count),
for a sum or a mean every row stating one. On a property spine a cell's
`{column}_source` names a source other than the row's own (a supplement,
a merged row's other source), and an empty one means the row's
`source`. A delivery withholds by these labels, so a restricted
supplement's value is withheld although its row's roll is open.
Values, ids and source names are fabricated, except `massgis`, which
stands for an open roll the recipe catalog declares.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState, aggregation
from openplaces.io.curator.provenance import record_sources
from openplaces.io.delivery import export_delivery
from openplaces.io.delivery.redaction import WITHHELD
from tests.io.test_delivery import _county, _recipe
from tests.io.test_delivery_restriction_gate import _source


def _state(curated):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US-NC-AL'),
        verbose=False,
        timer=None,
        curated=curated,
    )


def _patch(monkeypatch, rows):
    monkeypatch.setattr(
        aggregation,
        'get_entities',
        lambda recipe_id, admin_id, columns=None, missing='raise': rows[
            [c for c in columns if c in rows.columns]
        ].copy(),
    )
    monkeypatch.setattr(
        aggregation,
        'describe_recipe',
        lambda recipe_id, admin_id: pd.DataFrame(index=rows.columns),
    )


def _spine():
    """Three lots: L1 two rows of one roll, L2 a tie, L3 a single row."""
    return pd.DataFrame(
        {
            'parcel_id_local': ['L1', 'L1', 'L2', 'L2', 'L3'],
            'n_stories': [3.0, 2.0, 2.0, 2.0, 1.0],
            # The first row's stories came from a supplement.
            'n_stories_source': ['cityroll', None, None, None, None],
            'year_built': [1990.0, 1950.0, 1980.0, 1970.0, None],
            # And the second row's year.
            'year_built_source': [None, 'cityroll', None, None, None],
            'living_area_sqft': [1000.0, 500.0, 800.0, 700.0, 900.0],
            'source': [
                'massgis',
                'massgis',
                'US-NC-AL_property-countyroll-2026',
                'otherroll',
                'massgis',
            ],
        },
        index=pd.Index(['p1', 'p2', 'p3', 'p4', 'p5'], name='property_id'),
    )


def _gather(monkeypatch, columns, curated=None, **kwargs):
    _patch(monkeypatch, _spine())
    if curated is None:
        curated = pd.DataFrame({'parcel_id_local': ['L1', 'L2', 'L3']})
    state = aggregation.aggregate_from_entities(
        _state(curated), 'US_property-spine-2026', columns=columns, **kwargs
    )
    return state.curated


def test_max_names_the_row_holding_the_value(monkeypatch):
    curated = _gather(monkeypatch, {'n_stories': 'max'})
    # L1's three stories are the supplement's, not the roll's.
    assert curated['n_stories_source'].tolist() == [
        'cityroll',
        'countyroll+otherroll',
        'massgis',
    ]


def test_min_and_first_name_their_supplying_row(monkeypatch):
    curated = _gather(monkeypatch, {'year_built': 'min'})
    assert curated['year_built_source'].tolist()[:2] == ['cityroll', 'otherroll']
    # No row states a year on L3: no value, no label.
    assert pd.isna(curated['year_built_source'].iloc[2])

    curated = _gather(monkeypatch, {'n_stories': 'first'})
    assert curated['n_stories_source'].tolist() == [
        'cityroll',
        'countyroll',
        'massgis',
    ]


def test_first_follows_order_by(monkeypatch):
    curated = _gather(
        monkeypatch, {'n_stories': 'first'}, order_by={'year_built': 'ascending'}
    )
    assert curated['n_stories'].tolist() == [2.0, 2.0, 1.0]
    assert curated['n_stories_source'].tolist()[:2] == ['massgis', 'otherroll']


def test_a_sum_names_every_contributing_row(monkeypatch):
    curated = _gather(monkeypatch, {'n_stories': 'sum', 'living_area_sqft': 'sum'})
    assert curated['n_stories_source'].tolist() == [
        'cityroll+massgis',
        'countyroll+otherroll',
        'massgis',
    ]
    assert curated['living_area_sqft_source'].iloc[0] == 'massgis'


def test_prefer_replaces_the_label_of_the_value_it_replaced(monkeypatch):
    curated = pd.DataFrame(
        {
            'parcel_id_local': ['L1', 'L2', 'L3'],
            'n_stories': [9.0, 9.0, 9.0],
            'n_stories_source': ['parcelroll', 'parcelroll', 'parcelroll'],
        }
    )
    curated = _gather(
        monkeypatch, {'n_stories': 'max'}, curated=curated, prefer=['n_stories']
    )
    assert curated['n_stories'].tolist() == [3.0, 2.0, 1.0]
    assert curated['n_stories_source'].tolist() == [
        'cityroll',
        'countyroll+otherroll',
        'massgis',
    ]


def test_prefer_clears_a_label_when_the_rows_carry_none(monkeypatch):
    rows = pd.DataFrame({'parcel_id_local': ['L1', 'L2'], 'n_stories': [3.0, None]})
    _patch(monkeypatch, rows)
    curated = pd.DataFrame(
        {
            'parcel_id_local': ['L1', 'L2'],
            'n_stories': [9.0, 9.0],
            'n_stories_source': ['parcelroll', 'parcelroll'],
        }
    )
    state = aggregation.aggregate_from_entities(
        _state(curated), 'r', columns={'n_stories': 'max'}, prefer=['n_stories']
    )
    # The written cell lost a label that named the value it replaced;
    # the cell left as it was keeps its own.
    assert state.curated['n_stories'].tolist() == [3.0, 9.0]
    assert pd.isna(state.curated['n_stories_source'].iloc[0])
    assert state.curated['n_stories_source'].iloc[1] == 'parcelroll'


def test_record_sources_clears_only_when_asked():
    frame = pd.DataFrame({'x': [1, 2], 'x_source': ['old', 'old']})
    tokens = pd.Series([None, 'new'])
    record_sources(frame, 'x', tokens)
    assert frame['x_source'].tolist() == ['old', 'new']
    record_sources(frame, 'x', tokens, clear_missing=True)
    assert pd.isna(frame['x_source'].iloc[0])
    # A frame without the sidecar gains none from clearing alone.
    bare = pd.DataFrame({'x': [1]})
    record_sources(bare, 'x', pd.Series([None]), clear_missing=True)
    assert 'x_source' not in bare.columns


def test_a_token_that_is_not_a_recipe_id_is_kept():
    labels = pd.Series(['not_on_roll', 'US-XX-YY_property-someroll-2026+x:units'])
    assert aggregation._row_source_tokens(labels).tolist() == [
        'not_on_roll',
        'someroll+x',
    ]


# The delivery: a restricted supplement on an open roll.

CLOSED_CITY = _source('cityroll', 'US-NC-AL', ['year_built'], 'property')


@pytest.fixture
def restricted(monkeypatch):
    import openplaces.io.delivery as delivery_module

    monkeypatch.setattr(
        delivery_module, 'restricted_inputs', lambda *_a, **_k: [CLOSED_CITY]
    )


def _delivered(labels):
    """Two curated rows in the supplement's county, years 1950 and 1985."""
    _county(
        ['a', 'b'],
        admin3_id='US-NC-AL',
        year_built=[1950, 1985],
        extra={'year_built_source': labels},
    )
    paths = export_delivery(
        _recipe(), 'US-NC', admin_ids=['US-NC-AL'], allow_stale=True
    )
    return pd.read_parquet(paths['canonical']), pd.read_parquet(paths['evidence'])


def test_the_supplements_value_is_withheld_though_its_roll_is_open(
    monkeypatch, mock_data_root, restricted
):
    # L1's earliest year is the supplement's, on a row of the open roll.
    curated = _gather(monkeypatch, {'year_built': 'min'})
    assert curated['year_built'].iloc[0] == 1950.0
    labels = [curated['year_built_source'].iloc[0], 'massgis']
    assert labels == ['cityroll', 'massgis']

    canonical, evidence = _delivered(labels)

    assert pd.isna(canonical.loc['a', 'year_built'])
    assert canonical.loc['b', 'year_built'] == 1985
    assert evidence.loc['a', 'year_built_source'] == WITHHELD
    assert evidence.loc['b', 'year_built_source'] == 'massgis'


def test_the_row_label_alone_let_it_ship(mock_data_root, restricted):
    # The contrast: labeled with the row's roll, as before, the cell
    # read as the open roll's and was not withheld.
    canonical, _ = _delivered(['massgis', 'massgis'])
    assert canonical.loc['a', 'year_built'] == 1950
