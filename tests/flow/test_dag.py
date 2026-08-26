"""Tests for RecipeDAG against the committed CHEER recipe tree."""

import pytest

from openplaces.flow import RecipeDAG
from openplaces.geo.link import get_entity_link_path
from openplaces.io.delivery import (
    delivery_members,
    delivery_paths,
    delivery_regions,
)
from openplaces.recipe import get_output_path

TARGET = 'US_footprint-openplaces-2026'
COUNTY = 'US-NC-BRU'
# The recipe ships several regions; these tests exercise the Carolina one,
# which COUNTY belongs to.
REGION = 'cheer-eastern-nc'


@pytest.fixture(scope='module')
def dag():
    return RecipeDAG(TARGET, admin_ids=[COUNTY])


@pytest.fixture(scope='module')
def shipping_dag():
    """A run that covers the declared region, so the bundle is built.

    Forced rather than requested region-wide: expanding a state request to
    its 100 counties walks the whole recipe tree per county, which is far
    too slow for a test. `deliver=True` exercises the same node.
    """
    return RecipeDAG(TARGET, admin_ids=[COUNTY], deliver=True)


def test_nodes_cover_the_worked_example(dag):
    by_id = {node.recipe_id: node for node in dag.nodes()}
    # Terminal curate node
    assert by_id[TARGET].stage == 'curate'
    assert by_id[TARGET].admin_id == COUNTY
    # Harmonize spines
    assert by_id['US_footprint-spine-2026'].stage == 'harmonize'
    assert by_id['US_parcel-spine-2026'].stage == 'harmonize'
    # Parcel curation lane
    assert by_id['US_parcel-openplaces-2026'].stage == 'curate'
    # Ingest inputs (literal + auto-discovered for the county)
    for recipe_id in (
        'US_building-nsi-2026',
        'footprint-obm-2025',
        'US_footprint-microsoft-v2',
        'dwelling-overture-2025',
        'US-NC_parcel-nconemap-2025',
    ):
        assert by_id[recipe_id].stage == 'ingest', recipe_id
    # Enrichment evidence + its image ingest
    assert by_id['US_footprint_built-n-stories-brails-2026'].stage == 'enrich'
    assert by_id['image-googlestreetview-2026'].stage == 'ingest'


def test_output_path_matches_recipe_layer(dag):
    assert dag.output_path('harmonize', 'US_footprint-spine-2026', COUNTY) == (
        get_output_path('US_footprint-spine-2026', admin_id=COUNTY)
    )


def test_extra_outputs_include_link_sidecar(dag):
    # Persistence is default-on: every link_to_reference step that does not
    # opt out with save_link: false declares its sidecar, the spatial_point
    # joins (NSI, Overture) alongside the parcel overlay.
    extras = dag.extra_outputs('harmonize', 'US_footprint-geospine-2026', COUNTY)
    parcel_sidecar = get_entity_link_path(
        'US_footprint-geospine-2026', 'US-NC_parcel-nconemap-2025', COUNTY
    )
    assert parcel_sidecar in extras
    assert len(extras) == 3
    assert len(set(extras)) == 3
    # The attribute recipe runs no link steps of its own.
    assert dag.extra_outputs('harmonize', 'US_footprint-spine-2026', COUNTY) == []


def test_input_paths_of_curate_include_spine_and_sidecar(dag):
    inputs = dag.input_paths('curate', TARGET, COUNTY)
    spine_path = get_output_path('US_footprint-spine-2026', admin_id=COUNTY)
    # The sidecar curate actually opens is keyed by the geospine (the
    # spine's link owner), reached through the spine edge.
    sidecar = get_entity_link_path(
        'US_footprint-geospine-2026', 'US-NC_parcel-nconemap-2025', COUNTY
    )
    assert spine_path in inputs
    assert sidecar in inputs


def test_retention_classes(dag):
    assert dag.retention('ingest', 'US_building-nsi-2026') == 'until_consumed'
    assert dag.retention('harmonize', 'US_footprint-spine-2026') == 'keep'
    assert dag.retention('ingest', 'image-googlestreetview-2026') == ('until_consumed')
    assert dag.retention('curate', TARGET) == 'keep'
    assert dag.bucket(TARGET) == 'share'


def test_target_paths(dag):
    assert dag.target_paths() == [get_output_path(TARGET, admin_id=COUNTY)]


def test_scoped_run_does_not_deliver(dag):
    """A one-county run must leave the shipped regional bundle alone."""
    assert dag.delivery_node is None
    assert all(node.stage != 'deliver' for node in dag.nodes())


def test_delivery_node_covers_the_region(shipping_dag):
    node = shipping_dag.delivery_node
    assert node is not None
    assert (node.stage, node.recipe_id, node.admin_id) == ('deliver', TARGET, 'US-NC')
    assert node in shipping_dag.nodes()


def test_delivery_outputs_match_the_writer(shipping_dag):
    bundle = delivery_paths(TARGET, region=REGION)
    job = ('deliver', TARGET, 'US-NC')
    assert shipping_dag.output_path(*job) == bundle['canonical']
    # Every role except 'canonical', which is the primary output.
    # 'terms' belongs here: `export_delivery` always writes the LICENSE
    # notice and `target_paths` asks for the whole bundle, so omitting it
    # left `rule all` demanding a file no rule declared.
    assert shipping_dag.extra_outputs(*job) == [
        bundle[role] for role in ('point', 'geo', 'evidence', 'terms')
    ]


def test_delivery_inputs_are_every_member_county(shipping_dag):
    members = delivery_members(TARGET, region=REGION)
    assert len(members) == 44
    assert shipping_dag.input_paths('deliver', TARGET, 'US-NC') == [
        get_output_path(TARGET, admin_id=member) for member in members
    ]


def test_delivery_edges_let_plan_propagate(shipping_dag):
    """Every member county must feed the bundle, or plan() misses re-ships."""
    consumer = (TARGET, 'US-NC')
    upstreams = {up for up, down in shipping_dag._edges if down == consumer}
    assert upstreams == {
        (TARGET, member) for member in delivery_members(TARGET, region=REGION)
    }


def test_target_paths_are_the_bundle_when_shipping(shipping_dag):
    assert shipping_dag.target_paths() == list(
        delivery_paths(TARGET, region=REGION).values()
    )


def test_deliver_false_suppresses_the_bundle():
    dag = RecipeDAG(TARGET, admin_ids=[COUNTY], deliver=False)
    assert dag.delivery_node is None


def test_admin_and_tile_link_nodes_and_edges(dag):
    by_id = {node.recipe_id: node for node in dag.nodes()}
    # admin2 enters the DAG through admin3's create_index recipe reference
    assert by_id['US_admin-census-2025_admin2'].stage == 'ingest'
    edge_ids = {(up[0], down[0]) for up, down in dag._edges}
    assert ('US_admin-census-2025_admin2', 'US_admin-census-2025_admin3') in edge_ids
    # The tile grid consumes the admin layers declared under entity_links,
    # and those name the harmonized reference rather than a raw census
    # vintage: these crosswalks decide which tiles a unit's download
    # pulls, so a layer keyed on superseded identifiers sends the wrong
    # tiles to the wrong unit.
    assert ('admin-openplaces-2026_admin3', 'tile-obm-2025') in edge_ids


def test_tile_entity_links_extra_outputs(dag):
    extras = dag.extra_outputs('ingest', 'tile-obm-2025')
    assert (
        get_entity_link_path('tile-obm-2025', 'admin-openplaces-2026_admin3') in extras
    )
    assert len(extras) == 5


def test_footprint_inputs_include_tile_admin_link(dag):
    inputs = dag.input_paths('ingest', 'footprint-obm-2025', COUNTY)
    assert (
        get_entity_link_path('tile-obm-2025', 'admin-openplaces-2026_admin3') in inputs
    )


def test_placeslab_present_by_default(dag):
    # Opt-in lane, but present without exclusion: both the enrich recipe
    # (declared by US_parcel-openplaces-2026's merge_enrichments) and its
    # own reference ingest recipe (reference_parcel_recipe_id).
    ids = {n.recipe_id for n in dag.nodes()}
    assert 'US_parcel_parcel-placeslab-fmv2026' in ids
    assert 'US_parcel-placeslab-fmv2026' in ids


def test_excluding_placeslab_enrich_prunes_its_ingest_recipe_too():
    excluded = RecipeDAG(
        TARGET,
        admin_ids=[COUNTY],
        exclude_recipe_ids={'US_parcel_parcel-placeslab-fmv2026'},
    )
    ids = {n.recipe_id for n in excluded.nodes()}
    assert 'US_parcel_parcel-placeslab-fmv2026' not in ids
    assert 'US_parcel-placeslab-fmv2026' not in ids


def test_excluded_lane_leaves_no_dangling_curate_input():
    excluded = RecipeDAG(
        TARGET,
        admin_ids=[COUNTY],
        exclude_recipe_ids={'US_parcel_parcel-placeslab-fmv2026'},
    )
    inputs = excluded.input_paths('curate', 'US_parcel-openplaces-2026', COUNTY)
    assert not any('placeslab' in str(p) for p in inputs)


def test_excluding_image_enrich_recipe_prunes_its_image_ingest_too():
    excluded = RecipeDAG(
        TARGET,
        admin_ids=[COUNTY],
        exclude_recipe_ids={'US_footprint_built-n-stories-brails-2026'},
    )
    ids = {n.recipe_id for n in excluded.nodes()}
    assert 'US_footprint_built-n-stories-brails-2026' not in ids
    assert 'image-googlestreetview-2026' not in ids
    # The unrelated satellite-dependent enrich recipe is unaffected
    assert 'US_footprint_built-roof-shape-brails-2026' in ids
    assert 'image-googlesatellite-z20' in ids


def test_to_mermaid_defaults_to_horizontal(dag):
    assert 'flowchart LR' in dag.to_mermaid()


def test_to_mermaid_supports_vertical_direction(dag):
    assert 'flowchart TB' in dag.to_mermaid(direction='TB')


def test_to_mermaid_rejects_invalid_direction(dag):
    with pytest.raises(ValueError, match='direction'):
        dag.to_mermaid(direction='sideways')


def test_regions_are_declared_and_disjoint():
    """The recipe ships several named regions, and no county feeds two.

    Membership comes from the shared region registry, not from the recipe,
    so this also asserts the two CHEER regions stay wired together.
    """
    from openplaces.io.readers import get_region_admin_ids

    regions = {spec['region_id']: spec for spec in delivery_regions(TARGET)}
    for region_id, spec in regions.items():
        assert spec['admin_ids'] == get_region_admin_ids(region_id)
    assert {'cheer-eastern-nc', 'cheer-coastal-tx'} <= set(regions)
    assert len(regions['cheer-eastern-nc']['admin_ids']) == 44
    assert len(regions['cheer-coastal-tx']['admin_ids']) == 42
    members = [m for spec in regions.values() for m in spec['admin_ids']]
    assert len(members) == len(set(members))
    assert not set(regions['cheer-eastern-nc']['admin_ids']) & set(
        regions['cheer-coastal-tx']['admin_ids']
    )


def test_forced_delivery_ships_only_the_region_the_run_touches(shipping_dag):
    """A Carolina-scoped run must not rebuild the Texas bundle.

    Both bundles sit at admin level 2, so a bare level comparison would put
    both in scope; only the containment test tells them apart.
    """
    assert [node.admin_id for node, _ in shipping_dag.delivery_nodes] == ['US-NC']


def test_state_scope_ships_that_state_alone():
    dag = RecipeDAG(TARGET, admin_ids=['US-TX'], deliver=True)
    assert [node.admin_id for node, _ in dag.delivery_nodes] == ['US-TX']


def test_ambiguous_region_is_refused_rather_than_guessed():
    """Asking a multi-region recipe for "the" bundle must not pick one."""
    with pytest.raises(ValueError, match='delivery regions'):
        delivery_paths(TARGET)


def test_unknown_region_is_named_in_the_error():
    """A typo'd region id must say what is registered, not fail obscurely."""
    from openplaces.io.readers import get_regions

    # The registry is long enough that the listing is elided in the
    # middle, so match on the framing and its last entry, not on the id
    # the typo was aiming for.
    with pytest.raises(KeyError, match='registered: .*western-nc'):
        get_regions('cheer-coastal-texas')
