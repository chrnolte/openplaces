"""Tests for RecipeDAG against the committed CHEER recipe tree."""

import pytest

from openplaces.flow import RecipeDAG
from openplaces.geo.link import get_entity_link_path
from openplaces.recipe import get_output_path

TARGET = 'US_footprint-cheer-2026'
COUNTY = 'US-NC-BS'


@pytest.fixture(scope='module')
def dag():
    return RecipeDAG(TARGET, admin_ids=[COUNTY])


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
    extras = dag.extra_outputs('harmonize', 'US_footprint-spine-2026', COUNTY)
    assert extras == [
        get_entity_link_path(
            'US_footprint-spine-2026', 'US-NC_parcel-nconemap-2025', COUNTY
        )
    ]


def test_input_paths_of_curate_include_spine_and_sidecar(dag):
    inputs = dag.input_paths('curate', TARGET, COUNTY)
    spine_path = get_output_path('US_footprint-spine-2026', admin_id=COUNTY)
    sidecar = get_entity_link_path(
        'US_footprint-spine-2026', 'US-NC_parcel-nconemap-2025', COUNTY
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


def test_admin_and_tile_link_nodes_and_edges(dag):
    by_id = {node.recipe_id: node for node in dag.nodes()}
    # admin2 enters the DAG through admin3's create_index recipe reference
    assert by_id['US_admin-census-2021_admin2'].stage == 'ingest'
    edge_ids = {(up[0], down[0]) for up, down in dag._edges}
    assert ('US_admin-census-2021_admin2', 'US_admin-census-2021_admin3') in edge_ids
    # the tile grid consumes the admin layers declared under entity_links
    assert ('US_admin-census-2021_admin3', 'tile-obm-2025') in edge_ids


def test_tile_entity_links_extra_outputs(dag):
    extras = dag.extra_outputs('ingest', 'tile-obm-2025')
    assert (
        get_entity_link_path('tile-obm-2025', 'US_admin-census-2021_admin3') in extras
    )
    assert len(extras) == 5


def test_footprint_inputs_include_tile_admin_link(dag):
    inputs = dag.input_paths('ingest', 'footprint-obm-2025', COUNTY)
    assert (
        get_entity_link_path('tile-obm-2025', 'US_admin-census-2021_admin3') in inputs
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
