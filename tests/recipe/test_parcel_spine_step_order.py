"""The parcel spine's steps that read another recipe's pipeline output
sit after its checkpointed step.

A restored attribute checkpoint skips every step before it, and its
validity covers the geospine only. A step placed before the checkpoint
that reads a rebuilt property or transaction spine therefore keeps the
old join on every attribute rerun, unnoticed (seen on Victoria County TX
for the property link, 2026-09-20; the transaction link had the same
weakness until 2026-09-23). This pins the order so it cannot drift back.
"""

from openplaces.recipe import get_recipe_by_id

RECIPE = 'US_parcel-spine-2026'


def _pipeline():
    return get_recipe_by_id(RECIPE)['pipeline']


def _index_of(steps, predicate):
    return next(i for i, step in enumerate(steps) if predicate(step))


def test_exactly_one_step_is_checkpointed():
    steps = _pipeline()
    assert sum(1 for s in steps if s.get('checkpoint')) == 1


def test_the_property_link_table_is_written_after_the_checkpoint():
    steps = _pipeline()
    checkpoint = _index_of(steps, lambda s: s.get('checkpoint'))
    link = _index_of(steps, lambda s: s.get('step') == 'link_entities_by_id')
    assert link > checkpoint


def test_the_transaction_join_runs_after_the_checkpoint():
    steps = _pipeline()
    checkpoint = _index_of(steps, lambda s: s.get('checkpoint'))
    transactions = _index_of(
        steps,
        lambda s: (
            s.get('step') == 'link_by_id' and s.get('entity_type') == 'transaction'
        ),
    )
    assert transactions > checkpoint


def _strings(obj):
    """Every string inside a nested recipe value, keys included."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _strings(key)
            yield from _strings(value)
    elif isinstance(obj, list | tuple):
        for value in obj:
            yield from _strings(value)


def test_no_step_before_the_checkpoint_reads_the_transaction_columns():
    # The move is safe only while nothing earlier consumes what the
    # join writes; a step that starts to would have to move with it.
    steps = _pipeline()
    checkpoint = _index_of(steps, lambda s: s.get('checkpoint'))
    written = {'last_sale_price', 'last_sale_date', 'n_transactions'}
    for step in steps[:checkpoint]:
        assert not written & set(_strings(step)), step.get('step')
