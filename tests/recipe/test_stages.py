from openplaces.recipe import find_entity_recipe_id, get_recipe_by_id


def test_find_entity_recipe_follows_pipeline_stage_order():
    admin_id = 'US-NC-BR'

    assert find_entity_recipe_id(admin_id, 'footprint') == 'US_footprint-cheer-2026'
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


def test_cheer_curation_recipe_declares_predecessors():
    recipe = get_recipe_by_id('US_footprint-cheer-2026')

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
    }
