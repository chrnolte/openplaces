"""Tests for get_recipe_dependencies against the committed recipe tree."""

from openplaces.recipe import get_recipe_dependencies


def _upstream_ids(edges):
    return {e.upstream_recipe_id for e in edges if e.upstream_recipe_id}


def test_curate_recipe_edges():
    edges = get_recipe_dependencies('US_footprint-cheer-2026')
    upstream = _upstream_ids(edges)
    assert 'US_footprint-spine-2026' in upstream
    assert 'US_parcel-openplaces-2026' in upstream
    assert 'US_footprint_built-roof-shape-brails-2026' in upstream
    assert 'US_footprint_built-n-stories-brails-2026' in upstream
    kinds = {e.kind for e in edges}
    assert 'entity_recipe' in kinds


def test_footprint_spine_literal_edges():
    # The geometry phase (geospine) hosts resolve_spine and the spatial
    # links, so the raw footprint sources are its literal edges now.
    edges = get_recipe_dependencies('US_footprint-geospine-2026')
    upstream = _upstream_ids(edges)
    assert 'footprint-obm-2025' in upstream
    assert 'US_footprint-microsoft-v2' in upstream
    assert 'US_building-nsi-2026' in upstream
    assert 'dwelling-overture-2025' in upstream
    # Value crosswalks are not data dependencies
    assert 'US_building-nsi-2026_occupancy-type-remap' not in upstream

    # The attribute recipe's own edges: the geospine plus the sources its
    # reconcile/link_by_id steps still name directly.
    edges = get_recipe_dependencies('US_footprint-spine-2026')
    upstream = _upstream_ids(edges)
    assert 'US_footprint-geospine-2026' in upstream
    assert 'US_building-nsi-2026' in upstream
    assert 'dwelling-overture-2025' in upstream


def test_footprint_spine_auto_discover_unresolved_without_admin():
    edges = get_recipe_dependencies('US_footprint-spine-2026')
    unresolved = [e for e in edges if not e.resolved]
    assert unresolved, 'auto_discover without admin_id must yield unresolved edges'
    assert all(e.upstream_recipe_id is None for e in unresolved)


def test_footprint_spine_auto_discover_resolves_for_admin():
    # The geometry phase (geospine) hosts resolve_spine and the parcel
    # overlay, so the auto-discovered sources are its edges now.
    edges = get_recipe_dependencies('US_footprint-geospine-2026', admin_id='US-MA-MI')
    upstream = _upstream_ids(edges)
    # State footprint source discovered by the resolve_spine sentinel
    assert 'US-MA_footprint-massgis-2026' in upstream
    # Parcel reference discovered by link_to_reference (entity_type: parcel)
    assert 'US-MA_parcel-massgis-2025' in upstream
    # The attribute recipe reaches those sources through the geospine.
    edges = get_recipe_dependencies('US_footprint-spine-2026', admin_id='US-MA-MI')
    assert 'US_footprint-geospine-2026' in _upstream_ids(edges)


def test_enrich_recipe_edges():
    edges = get_recipe_dependencies('US_footprint_built-n-stories-brails-2026')
    by_kind = {e.kind: e for e in edges}
    assert by_kind['image_recipe'].upstream_recipe_id == ('image-googlestreetview-2026')
    # Dynamic spine resolution mirrors the enricher
    assert by_kind['entity_recipe'].upstream_recipe_id == 'US_footprint-spine-2026'


def test_ingest_recipe_admin_and_tile_edges():
    edges = get_recipe_dependencies('footprint-obm-2025')
    by_kind = {}
    for e in edges:
        by_kind.setdefault(e.kind, set()).add(e.upstream_recipe_id)
    assert 'tile-obm-2025' in by_kind.get('tile_recipe_id', set())
    assert 'US_admin-census-2021_admin3' in by_kind.get('admin_recipe_id', set())


def test_admin_id_crosswalk_is_not_an_edge():
    edges = get_recipe_dependencies('US-CT_parcel-ctgov-2024')
    upstream = _upstream_ids(edges)
    assert 'US-CT_parcel-ctgov-2024_admin4-crosswalk' not in upstream


def test_edges_carry_consumer_id():
    edges = get_recipe_dependencies('US_footprint-cheer-2026')
    assert all(e.recipe_id == 'US_footprint-cheer-2026' for e in edges)


def test_admin3_create_index_declares_admin2_edge():
    edges = get_recipe_dependencies('US_admin-census-2021_admin3')
    by_kind = {}
    for e in edges:
        by_kind.setdefault(e.kind, set()).add(e.upstream_recipe_id)
    assert by_kind.get('admin2_recipe_id') == {'US_admin-census-2021_admin2'}


def test_tile_entity_links_declare_admin_edges():
    edges = get_recipe_dependencies('tile-obm-2025')
    upstream = _upstream_ids(edges)
    # The harmonized layers, not GADM: a persisted tile-to-admin
    # crosswalk keyed on GADM identifiers is data openplaces cannot ship.
    assert {
        'admin-openplaces-2026_admin1',
        'admin-openplaces-2026_admin2',
        'admin-openplaces-2026_admin3',
        'US_admin-census-2021_admin2',
        'US_admin-census-2021_admin3',
    } <= upstream
    steps = {e.step for e in edges if e.kind == 'recipe_id'}
    assert 'entity_links' in steps


def test_reference_parcel_recipe_id_is_a_real_edge():
    edges = get_recipe_dependencies('US_parcel_parcel-placeslab-fmv2026')
    by_kind = {e.kind: e for e in edges}
    assert (
        by_kind['reference_parcel_recipe_id'].upstream_recipe_id
        == 'US_parcel-placeslab-fmv2026'
    )


def test_exclude_recipe_ids_suppresses_edge():
    edges = get_recipe_dependencies(
        'US_parcel-openplaces-2026',
        exclude_recipe_ids={'US_parcel_parcel-placeslab-fmv2026'},
    )
    upstream = _upstream_ids(edges)
    assert 'US_parcel_parcel-placeslab-fmv2026' not in upstream


def test_exclude_recipe_ids_default_is_unaffected():
    upstream = _upstream_ids(get_recipe_dependencies('US_parcel-openplaces-2026'))
    assert 'US_parcel_parcel-placeslab-fmv2026' in upstream
