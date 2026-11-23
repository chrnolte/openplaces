"""A county spine refuses a statewide source whose rows name no county.

The Indiana shape: a transaction recipe saved as one statewide file
(no process_by) whose rows carry neither a county id nor a county name.
Read for a county by union_spine_sources, it used to come back whole,
and every county's spine held the state's sales (Ohio County, Indiana:
1,017,881 spine rows for 847 sales). It raises now. Every value is
fabricated.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId, Entity
from openplaces.io import readers
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import spine as spine_module
from openplaces.io.harmonizer.spine import union_spine_sources
from openplaces.io.readers import UnnarrowableSourceError

RECIPE_ID = 'US-XA_transaction-statewide-2026'
RECIPE = {
    'recipe_id': RECIPE_ID,
    'admin_id': AdminId('US-XA'),
    'entity': Entity('transaction', 'statewide', '2026'),
    'save_to': {'data_dir': 'cache'},
}


class _Path:
    def __init__(self, key):
        self.key = key

    def exists(self):
        return True

    def __str__(self):
        return f'<{self.key}>'


@pytest.fixture
def statewide(monkeypatch):
    frames = {}

    monkeypatch.setattr(readers, 'get_recipe_by_id', lambda rid, **kwargs: dict(RECIPE))
    monkeypatch.setattr(
        readers,
        'get_output_path',
        lambda recipe, admin_id=None, **kwargs: _Path(str(admin_id)),
    )
    monkeypatch.setattr(
        readers,
        'read_parquet',
        lambda path, **kwargs: frames[path.key].copy(),
    )
    monkeypatch.setattr(
        readers,
        'get_admin',
        lambda admin_id, level=None, **kwargs: pd.DataFrame(
            index=pd.Index(['US-XA-AAA', 'US-XA-BBB'], name='admin3_id')
        ),
    )
    monkeypatch.setattr(
        spine_module, '_expand_auto_discover', lambda sources, state: sources
    )
    monkeypatch.setattr(spine_module, 'require_source_ingested', lambda *a: None)
    return frames


def _state(admin_id):
    return HarmonizeState(
        recipe={'admin_id': AdminId('US'), 'entity': None},
        admin_id=AdminId(admin_id),
        verbose=False,
        timer=None,
        spine=None,
    )


def test_a_county_spine_raises_instead_of_taking_the_state(statewide):
    statewide['US-XA'] = pd.DataFrame(
        {'price': [1.0, 2.0, 3.0], 'instrument_number': ['a', 'b', 'c']},
        index=pd.Index(['t1', 't2', 't3'], name='transaction_id'),
    )
    sources = [{'recipe_id': RECIPE_ID, 'label': 'statewide'}]
    with pytest.raises(UnnarrowableSourceError, match='US-XA-AAA'):
        union_spine_sources(_state('US-XA-AAA'), sources=sources)


def test_a_county_column_gives_the_county_its_rows_only(statewide):
    statewide['US-XA'] = pd.DataFrame(
        {
            'price': [1.0, 2.0, 3.0],
            'admin3_id': ['US-XA-AAA', 'US-XA-BBB', 'US-XA-BBB'],
        },
        index=pd.Index(['t1', 't2', 't3'], name='transaction_id'),
    )
    sources = [{'recipe_id': RECIPE_ID, 'label': 'statewide'}]
    spine = union_spine_sources(_state('US-XA-AAA'), sources=sources).spine
    assert len(spine) == 1
    assert spine['price'].tolist() == [1.0]
