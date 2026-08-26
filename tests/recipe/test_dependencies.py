"""Tests for get_recipe_dependencies against the committed recipe tree."""

from openplaces.recipe import get_recipe_by_id, get_recipe_dependencies


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
    """A tile-partitioned ingest recipe depends on its tile index and on
    the admin layer it resolves units against.

    Both edges are checked against what the recipe itself declares rather
    than against a literal recipe id: this test pinned
    `US_admin-census-2021_admin3` until the re-mint repointed recipes at
    the openplaces admin layer, which is a change in what the tree says,
    not a break in what this function derives.
    """
    recipe_id = 'footprint-obm-2025'
    recipe = get_recipe_by_id(recipe_id)
    edges = get_recipe_dependencies(recipe_id)
    by_kind = {}
    for e in edges:
        by_kind.setdefault(e.kind, set()).add(e.upstream_recipe_id)

    declared = recipe['download_by']
    assert declared['tile_recipe_id'] in by_kind.get('tile_recipe_id', set())
    assert declared['tile_admin_recipe_id'] in by_kind.get(
        'tile_admin_recipe_id', set()
    )

    admin_edges = by_kind.get('admin_recipe_id', set())
    assert admin_edges, 'no admin_recipe_id edge was derived'
    assert all('admin' in upstream for upstream in admin_edges), admin_edges


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
    """Every recipe named in `entity_links` becomes an edge.

    Checked against what the recipe declares rather than a literal list:
    this test pinned the 2021 census vintage until the tile grid moved to
    2025, which is a change in the tree, not in what this function
    derives. What the assertion is really for is that all of them arrive
    and that they are admin layers - a persisted tile-to-admin crosswalk
    keyed on GADM identifiers is data openplaces cannot ship.
    """
    recipe_id = 'tile-obm-2025'
    recipe = get_recipe_by_id(recipe_id)
    declared = {link['recipe_id'] for link in recipe['entity_links']}
    assert declared, 'the tile recipe declares no entity_links'
    assert all('admin' in upstream for upstream in declared), declared

    edges = get_recipe_dependencies(recipe_id)
    assert declared <= _upstream_ids(edges)
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
