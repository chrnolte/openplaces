"""A global consumer of a per-unit-saving recipe depends on that recipe's
per-unit jobs, never on a single global job.

The world admin layers (`admin-openplaces-2026_admin2`, `_admin3`) are
processed one parent unit at a time and save one file per unit. The
global OBM tile-grid ingest consumes them, and the graph used to resolve
that edge to one `_global` job whose output, `_all/.../admin2.parquet`,
no recipe writes. Snakemake then failed the job for a missing output and
blocked every county chain behind the tile grid.
"""

from pathlib import Path

import openplaces.flow.dag as dag_module
from openplaces.core.schema import AdminId
from openplaces.flow.dag import RecipeDAG

TARGET = 'US_footprint-openplaces-2026'


def _admin_nodes(dag, recipe_id):
    return [n for n in dag._nodes if n.recipe_id == recipe_id]


def test_world_admin_layers_have_no_global_job(monkeypatch):
    # Expansion asks the reader for the units at the save level; give it
    # a small world so the test does not depend on the spine on disk.
    def small_world(level, admin_id=None, **kwargs):
        scope = admin_id or None
        if level == 1:
            return ['AQ', 'US'] if scope is None else [scope]
        if level == 2:
            return ['US-DE', 'US-TX'] if scope in (None, 'US') else []
        return ['US-TX-KEY']

    import openplaces.io.readers as readers

    monkeypatch.setattr(readers, 'get_admin_ids', small_world)
    dag = RecipeDAG(TARGET, admin_ids=['US-TX-KEY'], deliver=False)

    admin2 = _admin_nodes(dag, 'admin-openplaces-2026_admin2')
    admin3 = _admin_nodes(dag, 'admin-openplaces-2026_admin3')
    assert all(n.admin_id is not None for n in admin2 + admin3)
    # A global consumer of either layer would get one job per unit at
    # the layer's save level; since the tile grid stopped consuming
    # them, a county graph holds only the county's own state.
    assert [str(a) for a in dag._node_admins('admin-openplaces-2026_admin2', None)] == [
        'AQ',
        'US',
    ]
    assert {n.admin_id for n in admin3} == {'US-TX'}


def test_a_global_consumer_expands_to_the_units_of_its_scope(monkeypatch):
    import openplaces.io.readers as readers

    monkeypatch.setattr(
        readers,
        'get_admin_ids',
        lambda level, admin_id=None, **kwargs: ['US-DE', 'US-TX'] if level == 2 else [],
    )
    dag = RecipeDAG(TARGET, admin_ids=['US-TX-KEY'], deliver=False)
    expanded = dag._node_admins('admin-openplaces-2026_admin3', None)
    assert [str(a) for a in expanded] == ['US-DE', 'US-TX']
    # The tile grid itself consumes no admin layer any more.
    inputs = [Path(p).name for p in dag.input_paths('ingest', 'tile-obm-2025', None)]
    assert not any('admin' in name for name in inputs)


def test_a_truly_global_recipe_still_resolves_to_one_job():
    dag = RecipeDAG(TARGET, admin_ids=['US-TX-KEY'], deliver=False)
    assert dag._node_admins('admin-openplaces-2026_admin1', None) == [None]
    assert isinstance(dag_module.AdminId('US'), AdminId)
