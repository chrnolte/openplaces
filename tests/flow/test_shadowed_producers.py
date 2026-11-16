"""One job per output file when a state fork shadows a national recipe.

`US-MA_footprint-fema-2023` writes the same entity-keyed file as the
national `US_footprint-fema-2023`, so the DAG keeps the fork and drops
the national job (transition review 4.4). Consumers that name the
national id must still show their edge to the kept job, and two equally
specific producers of one file must fail rather than be picked by walk
order.
"""

from __future__ import annotations

import warnings

import pytest

from openplaces.flow import RecipeDAG
from openplaces.flow.dag import StageNode, node_key

UNIT = 'US-MA-SOM'
NATIONAL = ('US_footprint-fema-2023', UNIT, None)
FORK = ('US-MA_footprint-fema-2023', UNIT, None)


@pytest.fixture(scope='module')
def dag():
    with warnings.catch_warnings():
        # Image recipes cannot expand to level 4 where admin4 is not built.
        warnings.simplefilter('ignore', UserWarning)
        return RecipeDAG('US_footprint-cheer-2026', admin_ids=[UNIT])


def test_the_state_fork_is_kept_and_the_national_job_dropped(dag):
    keys = {node_key(n) for n in dag.nodes()}
    assert FORK in keys
    assert NATIONAL not in keys
    assert dag._shadowed[NATIONAL] == FORK


def test_consumers_naming_the_national_id_keep_their_edge(dag):
    # Before 2026-11-14 only the footprint geospine, which names the
    # fork through auto-discovery, had this edge; the three consumers
    # naming the national id lost theirs in plan() and to_mermaid().
    consumers = {down[0] for up, down in dag._edges if up == FORK}
    assert {
        'US_footprint-geospine-2026',
        'US_parcel-geospine-2026',
        'US_parcel-spine-2026',
        'US_footprint_footprint-fema-2023',
    } <= consumers
    assert not any(up == NATIONAL for up, _ in dag._edges)


class _Stub(RecipeDAG):
    """A DAG over given nodes, recipes and output paths, without a walk."""

    def __init__(self, nodes, scopes, paths):
        self._nodes = list(nodes)
        self._scopes = scopes
        self._paths = paths
        self._shadowed = {}

    def _recipe(self, recipe_id):
        return {'admin_id': self._scopes[recipe_id]}

    def output_path(self, stage, recipe_id, admin_id):
        return self._paths[recipe_id]


def test_the_most_specific_producer_wins():
    nodes = [
        StageNode('ingest', 'US_x-src-2026', 'US-MA-SOM'),
        StageNode('ingest', 'US-MA_x-src-2026', 'US-MA-SOM'),
    ]
    stub = _Stub(
        nodes,
        {'US_x-src-2026': 'US', 'US-MA_x-src-2026': 'US-MA'},
        {'US_x-src-2026': 'same.parquet', 'US-MA_x-src-2026': 'same.parquet'},
    )
    seen = {node_key(n) for n in nodes}
    stub._drop_shadowed_producers(seen)
    assert [n.recipe_id for n in stub._nodes] == ['US-MA_x-src-2026']
    assert seen == {('US-MA_x-src-2026', 'US-MA-SOM', None)}


def test_an_equally_specific_tie_raises():
    nodes = [
        StageNode('ingest', 'US-MA_x-src-2026', 'US-MA-SOM'),
        StageNode('ingest', 'US-MA_x-other-2026', 'US-MA-SOM'),
    ]
    stub = _Stub(
        nodes,
        {'US-MA_x-src-2026': 'US-MA', 'US-MA_x-other-2026': 'US-MA'},
        {'US-MA_x-src-2026': 'same.parquet', 'US-MA_x-other-2026': 'same.parquet'},
    )
    with pytest.raises(ValueError, match='equally admin-specific'):
        stub._drop_shadowed_producers({node_key(n) for n in nodes})
