"""Tests for the canonical entity-link path resolver."""

from openplaces.geo.link import get_entity_link_path
from openplaces.recipe import get_output_path

SPINE = 'US_footprint-spine-2026'
PARCEL = 'US-NC_parcel-nconemap-2025'
ADMIN = 'US-NC-BS'


def test_argument_order_invariance():
    path_ab = get_entity_link_path(SPINE, PARCEL, admin_id=ADMIN)
    path_ba = get_entity_link_path(PARCEL, SPINE, admin_id=ADMIN)
    assert path_ab == path_ba


def test_link_stored_beside_finer_entity():
    # footprint is finer than parcel: the link lives beside the spine output
    link_path = get_entity_link_path(SPINE, PARCEL, admin_id=ADMIN)
    spine_path = get_output_path(SPINE, admin_id=ADMIN)
    assert link_path.parent == spine_path.parent
    assert link_path.name == f'{spine_path.stem}_{PARCEL}.parquet'


def test_tile_finer_than_admin():
    tile = 'US_tile-census-2025_tract'
    admin = 'US_admin-census-2021_admin3'
    link_path = get_entity_link_path(admin, tile)
    tile_path = get_output_path(tile)
    assert link_path.parent == tile_path.parent
    assert link_path.name == f'{tile_path.stem}_{admin}.parquet'


def test_same_entity_type_falls_back_to_lexicographic():
    a, b = 'footprint-obm-2025', 'US_footprint-microsoft-v2'
    path_ab = get_entity_link_path(a, b)
    path_ba = get_entity_link_path(b, a)
    assert path_ab == path_ba
    owner_id = sorted([a, b])[0]
    assert path_ab.stem.endswith(f'_{sorted([a, b])[1]}') or owner_id in path_ab.stem


def test_admin_id_truncated_to_owner_save_level():
    # The parcel recipe saves at county level (3); a finer town-level
    # admin_id must resolve to the county's link path.
    admin = 'US_admin-census-2021_admin3'
    town = get_entity_link_path(PARCEL, admin, admin_id='US-NC-BS-SP')
    county = get_entity_link_path(PARCEL, admin, admin_id='US-NC-BS')
    assert town == county
