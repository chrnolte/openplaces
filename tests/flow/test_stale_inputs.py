"""An input a built job lacks, which another job of the graph produces,
is a target.

Under mtime rerun triggers Snakemake does not schedule the producer of a
missing input when the consumer's output exists, so a new source recipe
for units whose spines were built planned nothing (2026-10-19, an Andes
run that added Cundinamarca's cadastre). The graph below is synthetic:
one spine that reads an old source and a new one.
"""

from collections import namedtuple

from openplaces.flow.dag import RecipeDAG
from openplaces.io.cleanup import write_receipt

Node = namedtuple('Node', 'stage recipe_id admin_id region')


def _dag(tmp_path, monkeypatch, files):
    spine = Node('harmonize', 'spine', 'XX-YY-ZZ', None)
    old = Node('ingest', 'old_source', 'XX-YY-ZZ', None)
    new = Node('ingest', 'new_source', 'XX-YY-ZZ', None)
    out = {n.recipe_id: tmp_path / f'{n.recipe_id}.parquet' for n in (spine, old, new)}
    for name in files:
        out[name].write_bytes(b'')
    dag = RecipeDAG.__new__(RecipeDAG)
    dag._nodes = [spine, old, new]
    monkeypatch.setattr(dag, 'output_path', lambda stage, rid, *a: out[rid])
    sources = [out['old_source'], out['new_source']]
    monkeypatch.setattr(
        dag, 'input_paths', lambda stage, rid, *a: sources if rid == 'spine' else []
    )
    return dag, out


def test_a_new_source_of_a_built_spine_is_a_target(tmp_path, monkeypatch):
    dag, out = _dag(tmp_path, monkeypatch, ['spine', 'old_source'])

    assert dag.stale_inputs() == [out['new_source']]


def test_nothing_is_added_when_the_consumer_is_not_built(tmp_path, monkeypatch):
    # Snakemake already schedules everything a missing output needs.
    dag, _ = _dag(tmp_path, monkeypatch, ['old_source'])

    assert dag.stale_inputs() == []


def test_nothing_is_added_when_every_input_exists(tmp_path, monkeypatch):
    dag, _ = _dag(tmp_path, monkeypatch, ['spine', 'old_source', 'new_source'])

    assert dag.stale_inputs() == []


def test_a_consumed_intermediate_is_not_a_target(tmp_path, monkeypatch):
    # input_paths drops an absent, receipted input of a built job; the
    # real method does, and stale_inputs relies on it.
    dag, out = _dag(tmp_path, monkeypatch, ['spine', 'old_source'])
    write_receipt(out['new_source'], {'consumers_verified': []})
    monkeypatch.setattr(
        dag,
        'input_paths',
        lambda stage, rid, *a: (
            RecipeDAG._without_consumed_intermediates(
                dag, [out['old_source'], out['new_source']], stage, rid, *a
            )
            if rid == 'spine'
            else []
        ),
    )

    assert dag.stale_inputs() == []
