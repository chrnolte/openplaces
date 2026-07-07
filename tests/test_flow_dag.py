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
        'US_building-nsi-2022',
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
    assert dag.retention('ingest', 'US_building-nsi-2022') == 'until_consumed'
    assert dag.retention('harmonize', 'US_footprint-spine-2026') == 'keep'
    assert dag.retention('ingest', 'image-googlestreetview-2026') == ('until_consumed')
    assert dag.retention('curate', TARGET) == 'keep'
    assert dag.bucket(TARGET) == 'share'


def test_target_paths(dag):
    assert dag.target_paths() == [get_output_path(TARGET, admin_id=COUNTY)]
