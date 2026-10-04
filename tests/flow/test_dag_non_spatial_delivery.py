"""A delivery with no geometry must plan, not raise.

`extra_outputs` is called for every node when the Snakefile builds its
plan, so indexing a bundle role that a non-spatial recipe never writes
kills the whole plan rather than one job. The transaction recipe is
the first such delivery.
"""

from __future__ import annotations

from openplaces.flow import RecipeDAG
from openplaces.io.delivery import delivery_paths, share_is_spatial
from openplaces.recipe import get_recipe_by_id

TARGET = 'US_transaction-openplaces-2026'
REGION = 'wisconsin-statewide'


def test_the_transaction_delivery_is_the_non_spatial_case():
    """Guards the premise, so this file fails loudly if that changes."""
    assert share_is_spatial(get_recipe_by_id(TARGET)) is False


def test_extra_outputs_declares_only_the_files_that_get_written():
    """No point or boundary file, so neither may be declared.

    Declaring one makes `rule all` demand a file no rule produces, the
    mirror of the bug that put 'terms' in this list originally.
    """
    dag = RecipeDAG(TARGET, admin_ids=['US-WI'], deliver=True)

    outputs = dag.extra_outputs('deliver', TARGET, 'US-WI', region=REGION)
    names = [p.name for p in outputs]

    assert any(n.endswith('_evidence.parquet') for n in names), names
    assert any(n.endswith('_LICENSE.txt') for n in names), names
    assert not any('_point' in n for n in names), names
    assert not any(n.endswith('_geo.parquet') for n in names), names


def test_every_declared_output_is_a_path_the_bundle_knows():
    """The declared set and the written set are the same set."""
    dag = RecipeDAG(TARGET, admin_ids=['US-WI'], deliver=True)

    declared = set(dag.extra_outputs('deliver', TARGET, 'US-WI', region=REGION))
    bundle = delivery_paths(TARGET, region=REGION)

    assert declared == set(bundle.values()) - {bundle['canonical']}


def test_the_transaction_spine_declares_its_link_tables():
    """`link_transactions` writes one link table per side it names, beside
    the spine's output, and the DAG declares both as the job's outputs."""
    from openplaces.geo.link import get_entity_link_path
    from openplaces.recipe import get_output_path

    dag = RecipeDAG(TARGET, admin_ids=['US-WI-VI'])
    spine = 'US_transaction-spine-2026'
    outputs = set(dag.extra_outputs('harmonize', spine, 'US-WI-VI'))
    parcel_link = get_entity_link_path(spine, 'US_parcel-spine-2026', 'US-WI-VI')
    property_link = get_entity_link_path(spine, 'US_property-spine-2026', 'US-WI-VI')
    assert parcel_link in outputs
    assert property_link in outputs
    # Beside the transaction spine, the finer entity.
    assert parcel_link.parent == get_output_path(spine, admin_id='US-WI-VI').parent
    # And the curate job reads them.
    inputs = set(dag.input_paths('curate', TARGET, 'US-WI-VI'))
    assert parcel_link in inputs


def test_planning_the_whole_graph_does_not_raise():
    """`extra_outputs` runs for every node, which is where it bit."""
    dag = RecipeDAG(TARGET, admin_ids=['US-WI'], deliver=True)

    for node in dag.nodes():
        dag.extra_outputs(
            node.stage,
            node.recipe_id,
            getattr(node, 'admin_id', None),
            region=getattr(node, 'region', None),
        )
