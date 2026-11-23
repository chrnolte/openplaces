"""A per-unit read of a coarser file narrows to the unit or raises.

A county read of a recipe saved one file per state used to return the
whole state with a warning when the rows named no county: every Indiana
county's transaction spine received the state's 1.2 million parcel-sale
rows. The read now narrows by an admin id column, by geometry, or by the
unit being its state's only county, and raises otherwise unless the
caller asks for the whole file. All values are fabricated.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from openplaces.core.schema import AdminId, Entity
from openplaces.io import readers
from openplaces.io.readers import (
    UNNARROWED_ATTR,
    UnnarrowableSourceError,
    get_entities,
)


def _recipe(admin_id='US-XA'):
    return {
        'recipe_id': f'{admin_id}_transaction-test-2026',
        'admin_id': AdminId(admin_id),
        'entity': Entity('transaction', 'test', '2026'),
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
def state_file(monkeypatch):
    """Patch the reader to serve one statewide frame per state key."""
    frames = {}

    def fake_get_output_path(recipe, admin_id=None, **kwargs):
        return _Path(str(admin_id))

    def fake_read_parquet(path, geom=False, columns=None, bbox=None, **kwargs):
        frame = frames[path.key].copy()
        if columns is not None:
            keep = [c for c in columns if c in frame.columns]
            if isinstance(frame, gpd.GeoDataFrame) and geom:
                keep.append('geometry')
            frame = frame[keep]
        return frame

    def fake_schema(path):
        return set(frames[path.key].columns)

    # Two counties in the fabricated state XA, one in XB.
    children = {
        ('US-XA', 3): ['US-XA-AAA', 'US-XA-BBB'],
        ('US-XB', 3): ['US-XB-ONE'],
    }

    def fake_get_admin(admin_id, level=None, columns=None, geom=False, **kwargs):
        if geom:
            ids = list(admin_id)
            shapes = {
                'US-XA-AAA': box(0, 0, 1, 1),
                'US-XA-BBB': box(1, 0, 2, 1),
            }
            return gpd.GeoDataFrame(
                geometry=[shapes[i] for i in ids],
                index=pd.Index(ids, name=f'admin{level}_id'),
                crs='EPSG:4326',
            )
        ids = children[(str(admin_id), level)]
        return pd.DataFrame(index=pd.Index(ids, name=f'admin{level}_id'))

    monkeypatch.setattr(readers, 'get_output_path', fake_get_output_path)
    monkeypatch.setattr(readers, 'read_parquet', fake_read_parquet)
    monkeypatch.setattr(readers, '_parquet_column_names', fake_schema)
    monkeypatch.setattr(readers, 'get_admin', fake_get_admin)
    monkeypatch.setattr(readers, 'is_placeholder_output', lambda path: False)
    return frames


def _no_county_column():
    return pd.DataFrame(
        {'price': [100.0, 200.0, 300.0]},
        index=pd.Index(['t1', 't2', 't3'], name='transaction_id'),
    )


class TestAStatewideFileWithoutCountyRaises:
    def test_a_county_read_raises(self, state_file):
        state_file['US-XA'] = _no_county_column()
        with pytest.raises(UnnarrowableSourceError) as caught:
            get_entities(_recipe(), 'US-XA-AAA')
        message = str(caught.value)
        assert 'US-XA_transaction-test-2026' in message
        assert 'US-XA-AAA' in message
        assert 'admin2' in message
        assert 'process_by' in message

    def test_the_error_is_not_a_value_error(self):
        # Callers skip an unavailable source on ValueError; this one
        # must not be skipped.
        assert not issubclass(UnnarrowableSourceError, ValueError)


class TestTheWholeFileOnlyWhenAsked:
    def test_unnarrowable_whole_returns_the_file_and_says_so(self, state_file):
        state_file['US-XA'] = _no_county_column()
        data = get_entities(_recipe(), 'US-XA-AAA', unnarrowable='whole')
        assert set(data.index) == {'t1', 't2', 't3'}
        assert data.attrs[UNNARROWED_ATTR] == {
            'recipe_id': 'US-XA_transaction-test-2026',
            'admin_ids': ['US-XA-AAA'],
            'save_level': 2,
        }

    def test_a_state_read_needs_no_narrowing(self, state_file):
        state_file['US-XA'] = _no_county_column()
        data = get_entities(_recipe(), 'US-XA')
        assert set(data.index) == {'t1', 't2', 't3'}
        assert UNNARROWED_ATTR not in data.attrs

    def test_an_unknown_policy_is_refused(self, state_file):
        state_file['US-XA'] = _no_county_column()
        with pytest.raises(ValueError, match='unnarrowable'):
            get_entities(_recipe(), 'US-XA-AAA', unnarrowable='warn')


class TestACountyColumnNarrows:
    def _with_county(self):
        frame = _no_county_column()
        frame['admin3_id'] = ['US-XA-AAA', 'US-XA-BBB', 'US-XA-AAA']
        return frame

    def test_rows_of_the_county_only(self, state_file):
        state_file['US-XA'] = self._with_county()
        data = get_entities(_recipe(), 'US-XA-AAA')
        assert set(data.index) == {'t1', 't3'}

    def test_a_column_read_still_narrows(self, state_file):
        # The county column is read for the narrowing although the
        # caller did not ask for it, and dropped again.
        state_file['US-XA'] = self._with_county()
        data = get_entities(_recipe(), 'US-XA-AAA', columns=['price'])
        assert set(data.index) == {'t1', 't3'}
        assert list(data.columns) == ['price']


class TestGeometryNarrows:
    def test_points_are_overlaid_with_the_county(self, state_file):
        state_file['US-XA'] = gpd.GeoDataFrame(
            {'price': [100.0, 200.0]},
            geometry=[Point(0.5, 0.5), Point(1.5, 0.5)],
            index=pd.Index(['t1', 't2'], name='transaction_id'),
            crs='EPSG:4326',
        )
        data = get_entities(_recipe(), 'US-XA-BBB', geom=True)
        assert set(data.index) == {'t2'}


class TestASoleCountyIsItsState:
    def test_the_only_county_reads_the_whole_file(self, state_file):
        state_file['US-XB'] = _no_county_column()
        data = get_entities(_recipe('US-XB'), 'US-XB-ONE')
        assert set(data.index) == {'t1', 't2', 't3'}
        assert UNNARROWED_ATTR not in data.attrs
