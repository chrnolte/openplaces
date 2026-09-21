"""Tests for RecipeDAG against the committed CHEER recipe tree."""

from copy import copy
from pathlib import Path

import pytest

from openplaces.flow import RecipeDAG
from openplaces.flow.dag import (
    node_key,
    parse_deliver_config,
    rule_name,
    validation_manifest_path,
)
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


def test_property_parcel_link_is_written_after_both_of_its_sides(dag):
    # The order the entity structure depends on: property spine, then
    # the parcel geospine (which reads it), then the parcel spine, which
    # writes the property-to-parcel link because it is the first recipe
    # in which both sides exist.
    property_spine = get_output_path('US_property-spine-2026', admin_id=COUNTY)
    geospine = get_output_path('US_parcel-geospine-2026', admin_id=COUNTY)
    assert property_spine in dag.input_paths(
        'harmonize', 'US_parcel-geospine-2026', COUNTY
    )
    spine_inputs = dag.input_paths('harmonize', 'US_parcel-spine-2026', COUNTY)
    assert geospine in spine_inputs
    assert property_spine in spine_inputs

    link = get_entity_link_path(
        'US_property-spine-2026', 'US_parcel-geospine-2026', COUNTY
    )
    # Beside the finer entity's output, named after the geospine.
    assert link.parent == property_spine.parent
    assert link in dag.extra_outputs('harmonize', 'US_parcel-spine-2026', COUNTY)
    assert link not in dag.extra_outputs('harmonize', 'US_property-spine-2026', COUNTY)


def test_retention_classes(dag):
    assert dag.retention('ingest', 'US_building-nsi-2026') == 'until_consumed'
    assert dag.retention('harmonize', 'US_footprint-spine-2026') == 'keep'
    assert dag.retention('ingest', 'image-googlestreetview-2026') == ('until_consumed')
    assert dag.retention('curate', TARGET) == 'keep'
    assert dag.bucket(TARGET) == 'share'


def test_target_paths(dag):
    assert dag.target_paths() == [get_output_path(TARGET, admin_id=COUNTY)]


def test_target_paths_expand_a_scope_coarser_than_the_save_level():
    """A state-scoped run must list the county files it actually builds.

    The jobs are already expanded to the recipe's save level when the
    graph is built; asking for the state's own output raised instead,
    so `rule all` could not be constructed.
    """
    state_dag = RecipeDAG(TARGET, admin_ids=['US-DE'], deliver=False)
    counties = [node.admin_id for node in state_dag.nodes() if node.recipe_id == TARGET]
    assert len(counties) > 1
    assert state_dag.target_paths() == [
        get_output_path(TARGET, admin_id=county) for county in counties
    ]


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
    """Every in-graph member county must feed the bundle.

    Restricted to members this run actually built: a forced ship from one
    county still declares the whole region as members, and an edge to a
    job the graph does not contain is a KeyError in `to_mermaid`.
    """
    consumer = (TARGET, 'US-NC', REGION)
    upstreams = {up for up, down in shipping_dag._edges if down == consumer}
    node_keys = {node_key(node) for node in shipping_dag.nodes()}
    members = set(delivery_members(TARGET, region=REGION))
    assert upstreams == {
        (TARGET, member, None)
        for member in members
        if (TARGET, member, None) in node_keys
    }
    assert upstreams


def test_every_edge_names_a_node_in_the_graph(shipping_dag):
    """An edge to a missing job made to_mermaid raise instead of drawing."""
    node_keys = {node_key(node) for node in shipping_dag.nodes()}
    for upstream, consumer in shipping_dag._edges:
        assert upstream in node_keys, upstream
        assert consumer in node_keys, consumer
    assert shipping_dag.to_mermaid(collapse_admin=False).startswith('%%{init:')


TEAM_REGION = f'{REGION}.team'


def test_target_paths_are_the_bundle_when_shipping(shipping_dag):
    # The region's bundle, then its team twin's (declared in team_regions).
    assert shipping_dag.target_paths() == [
        *delivery_paths(TARGET, region=REGION).values(),
        *delivery_paths(TARGET, region=TEAM_REGION).values(),
        # Then the job that re-scores the public region once it ships.
        validation_manifest_path(TARGET, REGION),
    ]


def test_deliver_false_suppresses_the_bundle():
    dag = RecipeDAG(TARGET, admin_ids=[COUNTY], deliver=False)
    assert dag.delivery_node is None


def test_admin_and_tile_link_nodes_and_edges(dag):
    by_id = {node.recipe_id: node for node in dag.nodes()}
    # admin2 enters the DAG through admin3's create_index recipe reference
    assert by_id['US_admin-census-2025_admin2'].stage == 'ingest'
    edge_ids = {(up[0], down[0]) for up, down in dag._edges}
    assert ('US_admin-census-2025_admin2', 'US_admin-census-2025_admin3') in edge_ids
    # The tile grid is a plain global ingest: it consumes no admin layer.
    # Tile-to-admin links are built per admin unit when a download first
    # needs them, so a county build never waits on the world.
    assert not any(down == 'tile-obm-2025' for _up, down in edge_ids)


def test_tile_grid_declares_no_link_sidecars(dag):
    assert dag.extra_outputs('ingest', 'tile-obm-2025') == []


def test_footprint_inputs_are_its_state_admin_file_and_the_tile_grid(dag):
    inputs = {str(p) for p in dag.input_paths('ingest', 'footprint-obm-2025', COUNTY)}
    names = {Path(p).name for p in inputs}
    assert 'US-NC_admin-openplaces-2026_admin3.parquet' in names
    assert 'tile-obm-2025.parquet' in names
    assert 'admin-openplaces-2026_admin3.parquet' not in names


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

    specs = delivery_regions(TARGET)
    regions = {s['region_id']: s for s in specs if s['audience'] == 'public'}
    for region_id, spec in regions.items():
        assert spec['admin_ids'] == get_region_admin_ids(region_id)
    # A team twin ships its base region's members, not a region of its own.
    for twin in (s for s in specs if s['audience'] == 'team'):
        assert twin['admin_ids'] == regions[twin['base_region_id']]['admin_ids']
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
    public = [
        node.admin_id
        for node, spec in shipping_dag.delivery_nodes
        if spec['audience'] == 'public'
    ]
    assert public == ['US-NC']
    # Its team twin ships with it; Texas has none and stays out.
    assert {node.admin_id for node, _ in shipping_dag.delivery_nodes} == {'US-NC'}


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


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        (None, None),
        (True, True),
        (False, False),
        ('true', True),
        ('false', False),
        # Snakemake types `--config deliver=0` as an integer, which a
        # string-only test skipped, so an unscoped run shipped anyway.
        (0, False),
        (1, True),
    ],
)
def test_parse_deliver_config(raw, expected):
    assert parse_deliver_config(raw) is expected


def _spine_with_regional_reference(dag_obj, recipe_id):
    """Copy of a geospine recipe whose only link names an NC-only source."""
    from copy import deepcopy

    recipe = deepcopy(dag_obj._recipe(recipe_id))
    recipe['pipeline'] = [
        {
            'step': 'link_to_reference',
            'recipe_id': 'US-NC_parcel-nconemap-2025',
            'join': 'spatial_overlay',
        }
    ]
    return recipe


def test_sidecar_is_skipped_for_a_reference_no_job_produces():
    """A region-scoped reference has no job outside its region.

    The harmonize step soft-skips it and writes no sidecar, so declaring
    one made Snakemake reject a spine it considers complete. Testing
    exclusion alone missed this, because nothing was excluded.
    """
    fake_id = 'US_footprint-geospine-2026'
    in_region = RecipeDAG(TARGET, admin_ids=[COUNTY], deliver=False)
    in_region._recipes[fake_id] = _spine_with_regional_reference(in_region, fake_id)
    assert in_region.extra_outputs('harmonize', fake_id, COUNTY) == [
        get_entity_link_path(fake_id, 'US-NC_parcel-nconemap-2025', COUNTY)
    ]

    texas_county = 'US-TX-HAR'
    outside = RecipeDAG(TARGET, admin_ids=[texas_county], deliver=False)
    outside._recipes[fake_id] = _spine_with_regional_reference(outside, fake_id)
    assert outside.extra_outputs('harmonize', fake_id, texas_county) == []


def test_sibling_regions_on_one_admin_unit_get_their_own_nodes(dag):
    """Two declared regions can roll up to the same admin unit.

    The recipe ships an eastern and a western North Carolina bundle, and
    two Boston ones. Keyed on (recipe, admin unit) alone the two nodes
    are the same node: the rule name collided and the region resolver
    returned the first match, so the second region never shipped.
    """
    # An unscoped run covers every declared region. Built from the
    # county fixture's graph rather than a state-wide one, whose node
    # expansion is far too slow for a test.
    unscoped = copy(dag)
    unscoped.requested_admin_ids = []
    nodes = unscoped._build_delivery_nodes(None)

    by_region = {node.region: node for node, _ in nodes}
    assert {'cheer-eastern-nc', 'western-nc'} <= set(by_region)
    assert {'boston-inner-core', 'boston-outer-rings'} <= set(by_region)
    assert by_region['cheer-eastern-nc'].admin_id == 'US-NC'
    assert by_region['western-nc'].admin_id == 'US-NC'
    assert by_region['boston-inner-core'].admin_id == 'US-MA'
    assert by_region['boston-outer-rings'].admin_id == 'US-MA'

    # One rule name per region, and each node resolves to its own region.
    rule_names = [rule_name(node) for node, _ in nodes]
    assert len(set(rule_names)) == len(nodes)
    for node, spec in nodes:
        assert (
            unscoped.delivery_region(node.recipe_id, node.admin_id, node.region)
            == spec['region_id']
        )
        assert node_key(node) not in {
            node_key(other) for other, _ in nodes if other is not node
        }


def test_a_run_scoped_to_one_region_ships_only_that_region(shipping_dag):
    """A Brunswick County run must not also ship the western bundle."""
    regions = [node.region for node, _ in shipping_dag.delivery_nodes]
    assert regions == [REGION, TEAM_REGION]
    assert shipping_dag.target_paths() == [
        *delivery_paths(TARGET, region=REGION).values(),
        *delivery_paths(TARGET, region=TEAM_REGION).values(),
        # Then the job that re-scores the public region once it ships.
        validation_manifest_path(TARGET, REGION),
    ]


def test_the_region_names_the_bundle_a_deliver_node_writes(shipping_dag):
    """Resolution goes through the node's own region, not its admin unit."""
    node = shipping_dag.delivery_nodes[0][0]
    assert (
        shipping_dag.output_path(node.stage, node.recipe_id, node.admin_id, node.region)
        == delivery_paths(TARGET, region=REGION)['canonical']
    )
    assert 'western-nc' not in rule_name(node)
