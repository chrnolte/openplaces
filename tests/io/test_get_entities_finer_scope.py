"""get_entities() narrows to the union of what was requested.

A request finer than a recipe's save level is truncated to the file it
lives in, so the read file can cover units the caller never asked for and
has to be narrowed afterwards. That narrowing used to be a chain of
successive filters, which dropped every row of a unit requested whole and
returned nothing at all for a request naming two different finer levels.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId, Entity
from openplaces.io import readers
from openplaces.io.readers import get_entities

RECIPE = {
    'recipe_id': 'US_footprint-test-2026',
    'admin_id': AdminId('US'),
    'entity': Entity('footprint', 'test', '2026'),
    'save_to': {'admin_level': 3, 'data_dir': 'cache'},
}


@pytest.fixture
def county_files(monkeypatch):
    """Two county files, one of them carrying finer admin id columns."""
    frames = {
        'US-NC-WAK': pd.DataFrame(
            {'value': [1, 2]},
            index=pd.Index(['w1', 'w2'], name='footprint_id'),
        ),
        'US-NC-BRU': pd.DataFrame(
            {
                'value': [3, 4, 5],
                'admin4_id': [
                    'US-NC-BRU-BLD',
                    'US-NC-BRU-OTH',
                    'US-NC-BRU-OTH',
                ],
                'admin5_id': [
                    'US-NC-BRU-BLD-AAA',
                    'US-NC-BRU-OTH-SUB',
                    'US-NC-BRU-OTH-ZZZ',
                ],
            },
            index=pd.Index(['b1', 'b2', 'b3'], name='footprint_id'),
        ),
    }

    class _Path:
        def __init__(self, key):
            self.key = key

        def exists(self):
            return self.key in frames

        def __str__(self):
            return f'<{self.key}>'

    def fake_get_output_path(recipe, admin_id=None, **kwargs):
        return _Path(str(admin_id))

    def fake_read_parquet(path, geom=False, columns=None, bbox=None, **kwargs):
        return frames[path.key].copy()

    monkeypatch.setattr(readers, 'get_output_path', fake_get_output_path)
    monkeypatch.setattr(readers, 'read_parquet', fake_read_parquet)
    return frames


class TestNarrowingKeepsUnitsRequestedWhole:
    def test_a_county_named_beside_a_town_survives(self, county_files):
        data = get_entities(
            RECIPE, admin_id=['US-NC-WAK', 'US-NC-BRU-BLD'], missing='ignore'
        )
        assert set(data.index) == {'w1', 'w2', 'b1'}

    def test_a_town_alone_still_narrows_its_county(self, county_files):
        data = get_entities(RECIPE, admin_id='US-NC-BRU-BLD', missing='ignore')
        assert set(data.index) == {'b1'}


class TestNarrowingCombinesLevelsAsAUnion:
    def test_two_finer_levels_return_both_units(self, county_files):
        data = get_entities(
            RECIPE,
            admin_id=['US-NC-BRU-BLD', 'US-NC-BRU-OTH-SUB'],
            missing='ignore',
        )
        assert set(data.index) == {'b1', 'b2'}


class TestOutputAdminIdsReportWholeUnits:
    def test_a_save_level_request_is_reported_whole(self):
        output_ids, finer, whole = readers._get_output_admin_ids(
            RECIPE, ['US-NC-WAK', 'US-NC-BRU-BLD']
        )
        assert output_ids == [AdminId('US-NC-WAK'), AdminId('US-NC-BRU')]
        assert finer == [AdminId('US-NC-BRU-BLD')]
        assert whole == [AdminId('US-NC-WAK')]
