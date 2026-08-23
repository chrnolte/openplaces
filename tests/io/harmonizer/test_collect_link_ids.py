"""Tests for the collect_link_ids curate step (parcel_id_all)."""

import pandas as pd
import pytest

from openplaces.config import cfg
from openplaces.core.attribute_registry import get_data_type, load_registry
from openplaces.core.schema import AdminId
from openplaces.geo.link import get_entity_link_path
from openplaces.io import to_parquet
from openplaces.io.curator import CurateState
from openplaces.io.curator.evidence import collect_link_ids
from openplaces.recipe import get_recipe_by_id

SPINE = 'US_footprint-spine-2026'
# The sidecar is keyed by the recipe that ran the link steps -- the
# geospine under the split (get_link_owner_recipe_id resolves it from the
# spine's entity_recipe chain).
GEOSPINE = 'US_footprint-geospine-2026'
PARCEL = 'US-NC_parcel-nconemap-2025'
COUNTY = 'US-NC-BR'


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    dirs = dict(cfg.config['directories'])
    dirs['data_root'] = tmp_path
    for name in ('core', 'external', 'raw', 'cache', 'out', 'share'):
        dirs[name] = tmp_path / 'data' / name
    dirs['heap'] = tmp_path / 'data/cache/_heap'
    dirs['logs'] = tmp_path / 'data/cache/_logs'
    monkeypatch.setitem(cfg.config, 'directories', dirs)
    return tmp_path


def _write_sidecar():
    """f1: single parcel. f2: two kept parcels. f3: kept P1 + sliver P2.
    f4: no parcel."""
    links = pd.DataFrame(
        {
            'footprint_id': ['f1', 'f2', 'f2', 'f3', 'f3', 'f4'],
            'parcel_id': ['P1', 'P1', 'P2', 'P1', 'P2', None],
            'area_intersection_m2': [40.0, 30.0, 60.0, 50.0, 2.0, None],
            'link': [
                'unique parcel',
                'multi-parcel footprint',
                'multi-parcel footprint',
                'unique parcel (dropping small neighbor)',
                None,  # trimmed-out sliver pair
                'no parcel',
            ],
        }
    )
    path = get_entity_link_path(GEOSPINE, PARCEL, admin_id=COUNTY)
    to_parquet(links, path)
    return path


def _state():
    curated = pd.DataFrame(
        {'parcel_id': ['P1', 'P2', 'P1', None]},
        index=pd.Index(['f1', 'f2', 'f3', 'f4'], name='footprint_id'),
    )
    return CurateState(
        recipe={'recipe_id': 'US_footprint-cheer-2026'},
        entity_recipe=get_recipe_by_id(SPINE),
        admin_id=AdminId(COUNTY),
        verbose=False,
        timer=None,
        curated=curated,
    )


def test_collects_dominant_first(data_root):
    _write_sidecar()
    state = collect_link_ids(_state(), entity_type='parcel')
    out = state.curated['parcel_id_all']
    # Identical to parcel_id -> left missing (adds no information)
    assert pd.isna(out['f1'])
    # Dominant (largest intersection) first
    assert out['f2'] == 'P2|P1'
    assert out['f2'].split('|')[0] == state.curated.loc['f2', 'parcel_id']
    # Sliver pair excluded by default -> single id equal to parcel_id
    assert pd.isna(out['f3'])
    assert pd.isna(out['f4'])


def test_single_id_kept_when_it_differs_from_parcel_id(data_root):
    _write_sidecar()
    state = _state()
    # f1's dominant parcel was assigned differently upstream; the collected
    # single id is new information and must not be blanked.
    state.curated.loc['f1', 'parcel_id'] = 'P9'
    state = collect_link_ids(state, entity_type='parcel')
    assert state.curated['parcel_id_all']['f1'] == 'P1'


def test_include_below_threshold(data_root):
    _write_sidecar()
    state = collect_link_ids(
        _state(), entity_type='parcel', include_below_threshold=True
    )
    assert state.curated['parcel_id_all']['f3'] == 'P1|P2'


def test_missing_sidecar_raises(data_root):
    with pytest.raises(FileNotFoundError, match='save_link'):
        collect_link_ids(_state(), entity_type='parcel')


def test_explicit_link_recipe_id(data_root):
    _write_sidecar()
    state = collect_link_ids(_state(), link_recipe_id=PARCEL)
    assert state.curated['parcel_id_all']['f2'] == 'P2|P1'


def test_registry_row():
    assert 'parcel_id_all' in load_registry().index
    assert get_data_type('parcel_id_all') == 'string'
