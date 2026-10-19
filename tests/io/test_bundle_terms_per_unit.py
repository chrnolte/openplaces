"""A bundle's notice names the sources its own units read, and no others.

Walked without its units, the transaction bundle's notice saw only the
national recipe's declared inputs: Wisconsin's notice missed RETR, its
own sales feed, and listed Florida's DOR roll, which the national recipe
named for a Florida-only temporal join (2026-10-19). The join is now a
Florida patch, and the export walks the bundle's units.
"""

from openplaces.io.delivery.terms import bundle_terms
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id

RECIPE = 'US_transaction-openplaces-2026'


def _source_ids(admin_ids):
    terms = bundle_terms(RECIPE, admin_ids=admin_ids)
    return {entry['source_id'] for entry in terms['sources']}


def test_a_wisconsin_notice_lists_retr_and_not_floridas_roll():
    ids = _source_ids(['US-WI-VI'])
    assert 'widor' in ids
    assert 'fldor' not in ids


def test_a_florida_notice_lists_its_roll():
    assert 'fldor' in _source_ids(['US-FL-LA'])


def test_the_temporal_join_runs_in_florida_only():
    recipe = get_recipe_by_id(RECIPE)

    def joins(admin_id):
        patched, _ = apply_recipe_patches(recipe, admin_id)
        return [
            s['direction']
            for s in patched['pipeline']
            if s['step'] == 'join_temporal_snapshot'
        ]

    assert joins('US-FL-LA') == ['backward', 'forward']
    assert joins('US-WI-VI') == []
    steps = [s['step'] for s in apply_recipe_patches(recipe, 'US-FL-LA')[0]['pipeline']]
    assert steps.index('join_temporal_snapshot') < steps.index(
        'keep_registered_columns'
    )
