import pytest

from openplaces.recipe import (
    find_entity_recipe_id,
    get_recipe_by_id,
    get_recipe_dict,
)


def test_find_entity_recipe_follows_pipeline_stage_order():
    admin_id = 'US-NC-BRU'

    assert (
        find_entity_recipe_id(admin_id, 'footprint') == 'US_footprint-openplaces-2026'
    )
    assert (
        find_entity_recipe_id(
            admin_id,
            'footprint',
            stage='harmonize',
            source_id='spine',
        )
        == 'US_footprint-spine-2026'
    )

    enrichment_id = find_entity_recipe_id(
        admin_id,
        'footprint',
        stage='enrich',
    )
    assert get_recipe_by_id(enrichment_id)['stage'] == 'enrich'


def test_curation_recipe_declares_predecessors():
    recipe = get_recipe_by_id('US_footprint-openplaces-2026')

    assert recipe['stage'] == 'curate'
    assert recipe['entity_recipe'] == 'US_footprint-spine-2026'

    merge_step = next(
        step for step in recipe['pipeline'] if step['step'] == 'merge_enrichments'
    )
    assert {spec['recipe_id'] for spec in merge_step['recipes']} == {
        'US_footprint_built-roof-shape-brails-2026',
        'US_footprint_built-n-stories-brails-2026',
        # Region-scoped: a precomputed statewide inventory standing
        # in for the per-image classifications above wherever it has
        # coverage. merge_enrichments skips a spec with no evidence
        # for the admin unit, so a run outside NC is unaffected.
        'US-NC_footprint_building-cheer-v0',
        # Also NC-only, and skipped the same way: the NCDPS per-building
        # construction years the year_built reconcile ranks above NSI.
        'US-NC_footprint_footprint-ncdps-2023',
    }


def test_unknown_stage_is_rejected_at_load(tmp_path):
    """A misspelled stage must not load as a stage-less recipe.

    'harmonise' used to load without complaint, rank below every ingest
    recipe in `find_entity_recipe_id`, and surface only when the job it
    belongs to finally ran.
    """
    recipe_file = tmp_path / 'US-XX_parcel-fabricated-2026.yaml'
    recipe_file.write_text(
        'admin_id: US-XX\n'
        'stage: harmonise\n'
        'entity:\n'
        '  entity_type: parcel\n'
        '  source:\n'
        '    source_id: fabricated\n'
        '  version: "2026"\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='not a known openplaces pipeline stage'):
        get_recipe_dict(recipe_file, 'US-XX')


def test_unmatched_source_id_returns_none_rather_than_another_source():
    """source_id filters; it does not merely rank.

    As a preference it returned the next-best recipe of a different
    source, and every caller tests only for None, so an enrich run whose
    spine had been renamed would have silently read the geospine.
    """
    assert (
        find_entity_recipe_id(
            'US-NC-BRU',
            'footprint',
            stage='harmonize',
            source_id='nonexistent',
            silent=True,
        )
        is None
    )
    assert (
        find_entity_recipe_id(
            'US-NC-BRU',
            'footprint',
            stage='harmonize',
            source_id='spine',
            silent=True,
        )
        == 'US_footprint-spine-2026'
    )
