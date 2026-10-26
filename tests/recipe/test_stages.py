import pytest

from openplaces.recipe import (
    find_entity_recipe_id,
    get_recipe_by_id,
    get_recipe_dict,
)


def test_find_entity_recipe_follows_pipeline_stage_order():
    admin_id = 'US-NC-BRU'

    assert find_entity_recipe_id(admin_id, 'footprint') == 'US_footprint-cheer-2026'
    assert find_entity_recipe_id(admin_id, 'building') == 'US_building-openplaces-2026'
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
    recipe = get_recipe_by_id('US_building-openplaces-2026')

    assert recipe['stage'] == 'curate'
    assert recipe['entity_recipe'] == 'US_building-spine-2026'

    merge_step = next(
        step for step in recipe['pipeline'] if step['step'] == 'merge_enrichments'
    )
    # Every entry is an outline enrichment, read under the footprint
    # spine and joined through the building's footprint_id; without it
    # the step looks under the building spine and skips the entry.
    assert all(
        spec.get('entity_recipe') == 'US_footprint-spine-2026'
        and spec.get('key') == 'footprint_id'
        for spec in merge_step['recipes']
    )
    assert {spec['recipe_id'] for spec in merge_step['recipes']} == {
        'US_footprint_built-roof-shape-brails-2026',
        'US_footprint_built-n-stories-brails-2026',
        # Region-scoped: a precomputed statewide inventory standing
        # in for the per-image classifications above wherever it has
        # coverage (NC statewide; Harris and Jefferson TX).
        # merge_enrichments skips a spec with no evidence for the
        # admin unit, so a run outside those units is unaffected.
        'US-NC_footprint_building-cheer-v0',
        'US-TX_footprint_building-cheer-v0',
        # Also NC-only, and skipped the same way: the NCDPS per-building
        # construction years the year_built reconcile ranks above NSI.
        'US-NC_footprint_footprint-ncdps-2023',
        # National: Overture heights, which the floor-weighted value
        # split and the unconfirmed-height flag read.
        'US_footprint_footprint-overture-2026',
        # National: FEMA's LiDAR heights, from which the n_stories
        # reconcile takes a measured story count.
        'US_footprint_footprint-fema-2023',
    }


def test_the_cheer_footprint_curate_folds_the_building_curate():
    """b3 step 3: the shipped footprint curate decides nothing; it carries
    the building curate onto the outlines and keeps delivery and
    validation, whose rulesets it reads from the building curate."""
    cheer = get_recipe_by_id('US_footprint-cheer-2026')
    assert cheer['entity_recipe'] == 'US_footprint-spine-2026'
    fold = cheer['pipeline'][0]
    assert fold['step'] == 'aggregate_from_entities'
    assert fold['recipe_id'] == 'US_building-openplaces-2026'
    assert fold['columns'] == 'all'
    assert fold['count_as'] == 'n_buildings_per_footprint'
    assert 'impute' in cheer['share']['delivery']
    assert cheer['validation']['rules_recipe'] == 'US_building-openplaces-2026'
    building = get_recipe_by_id('US_building-openplaces-2026')
    assert 'share' not in building and 'validation' not in building
    assert 'occupancy' in building


def test_validation_finds_the_rulesets_beside_the_building_curate():
    """A ruleset validation cannot locate returns None, and the notebooks
    then leave that evidence lane out of the scores without an error, so
    the footprint curate's context must reach the building curate's."""
    import pandas as pd

    from openplaces.io.curator.validation import validation_context

    ctx = validation_context('US_footprint-cheer-2026')
    assert 'Single-Family' in ctx.residential_classes
    terms = pd.Series(['SINGLE FAMILY RESIDENCE', 'MOBILE HOME'])
    for ruleset in (ctx.class_map, ctx.keyword_ruleset):
        assert ctx.class_from_ruleset(terms, ruleset) is not None, ruleset


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
