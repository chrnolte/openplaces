"""A patch recipe amends a national pipeline for the units in its scope.

plans/core-schema-and-stage-contracts-review.md, section 4: a county's
rules live in a county file that names the national recipe in
`patches:` and lists `pipeline_patch` operations; the national recipe
stays the generic pipeline, one recipe id stays in the graph, and the
output records which patches applied. Fabricated recipes throughout.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.core.schema import AdminId, Entity
from openplaces.recipe import (
    apply_recipe_patches,
    find_entity_recipe_id,
    find_recipe_patches,
)


def _national(pipeline=None):
    return {
        'recipe_id': 'US_parcel-fabricated-2026',
        'admin_id': AdminId('US'),
        'stage': 'curate',
        'entity': Entity('parcel', 'fabricated', '2026'),
        'pipeline': pipeline
        or [
            {'step': 'declare_columns', 'columns': ['a']},
            {'step': 'exclude_by_value', 'column': 'id', 'values': ['WATER']},
            {'step': 'derive_indicators', 'indicators': []},
            {'step': 'resolve_by_vote', 'target': 't', 'decisions': []},
            {'step': 'order_columns'},
        ],
    }


def _patch(admin_id, operations, **extra):
    return {
        'recipe_id': f'{admin_id}_parcel-fabricated-2026',
        'admin_id': AdminId(admin_id),
        'stage': 'curate',
        'entity': Entity('parcel', 'fabricated', '2026'),
        'patches': 'US_parcel-fabricated-2026',
        'pipeline_patch': operations,
        **extra,
    }


def _steps(recipe):
    return [s['step'] for s in recipe['pipeline']]


def test_no_patch_returns_the_recipe_itself():
    recipe = _national()
    same, applied = apply_recipe_patches(recipe, 'US-XX-YY', patches=[])
    assert same is recipe
    assert applied == []


def test_extend_appends_to_a_list_parameter_and_leaves_the_input_untouched():
    recipe = _national()
    patch = _patch(
        'US-XX-YY',
        [{'extend': 'exclude_by_value', 'key': 'values', 'values': ['ROW', 'CALO']}],
    )
    patched, applied = apply_recipe_patches(recipe, 'US-XX-YY', patches=[patch])
    assert applied == ['US-XX-YY_parcel-fabricated-2026']
    assert patched['pipeline'][1]['values'] == ['WATER', 'ROW', 'CALO']
    assert recipe['pipeline'][1]['values'] == ['WATER']


def test_set_insert_replace_and_remove():
    recipe = _national()
    patch = _patch(
        'US-XX-YY',
        [
            {'set': 'resolve_by_vote', 'params': {'min_score': 2}},
            {
                'insert_after': 'declare_columns',
                'step': {'step': 'merge_enrichments', 'recipes': []},
            },
            {'insert_before': 'order_columns', 'step': {'step': 'cast_integers'}},
            {
                'replace': 'derive_indicators',
                'step': {'step': 'derive_indicators', 'indicators': [{'output': 'x'}]},
            },
            {'remove': 'exclude_by_value'},
        ],
    )
    patched, _ = apply_recipe_patches(recipe, 'US-XX-YY', patches=[patch])
    assert _steps(patched) == [
        'declare_columns',
        'merge_enrichments',
        'derive_indicators',
        'resolve_by_vote',
        'cast_integers',
        'order_columns',
    ]
    vote = next(s for s in patched['pipeline'] if s['step'] == 'resolve_by_vote')
    assert vote['min_score'] == 2 and vote['target'] == 't'
    indicators = next(
        s for s in patched['pipeline'] if s['step'] == 'derive_indicators'
    )
    assert indicators['indicators'] == [{'output': 'x'}]


def test_a_repeated_step_name_needs_an_occurrence():
    pipeline = [
        {'step': 'derive_indicators', 'indicators': [1]},
        {'step': 'derive_indicators', 'indicators': [2]},
    ]
    recipe = _national(pipeline)
    with pytest.raises(ValueError, match='occurrence'):
        apply_recipe_patches(
            recipe,
            'US-XX-YY',
            patches=[_patch('US-XX-YY', [{'remove': 'derive_indicators'}])],
        )
    patched, _ = apply_recipe_patches(
        recipe,
        'US-XX-YY',
        patches=[
            _patch('US-XX-YY', [{'remove': 'derive_indicators', 'occurrence': 2}])
        ],
    )
    assert patched['pipeline'] == [{'step': 'derive_indicators', 'indicators': [1]}]


def test_a_missing_step_an_unknown_operation_and_a_wrong_stage_are_refused():
    recipe = _national()
    with pytest.raises(ValueError, match='no step'):
        apply_recipe_patches(
            recipe, 'US-XX-YY', patches=[_patch('US-XX-YY', [{'remove': 'nope'}])]
        )
    with pytest.raises(ValueError, match='exactly one of'):
        apply_recipe_patches(
            recipe,
            'US-XX-YY',
            patches=[_patch('US-XX-YY', [{'delete': 'order_columns'}])],
        )
    wrong_stage = _patch('US-XX-YY', [{'remove': 'order_columns'}])
    wrong_stage['stage'] = 'harmonize'
    with pytest.raises(ValueError, match='cannot patch'):
        apply_recipe_patches(recipe, 'US-XX-YY', patches=[wrong_stage])
    wrong_entity = _patch('US-XX-YY', [{'remove': 'order_columns'}])
    wrong_entity['entity'] = Entity('footprint', 'fabricated', '2026')
    with pytest.raises(ValueError, match='entity types differ'):
        apply_recipe_patches(recipe, 'US-XX-YY', patches=[wrong_entity])


def test_patches_apply_coarse_to_fine_so_the_county_has_the_last_word(monkeypatch):
    index = pd.DataFrame(
        [
            {
                'admin_id': 'US-XX-YY',
                'recipe_id': 'US-XX-YY_parcel-fabricated-2026',
                'patches': 'US_parcel-fabricated-2026',
            },
            {
                'admin_id': 'US-XX',
                'recipe_id': 'US-XX_parcel-fabricated-2026',
                'patches': 'US_parcel-fabricated-2026',
            },
            {
                'admin_id': 'US-ZZ',
                'recipe_id': 'US-ZZ_parcel-fabricated-2026',
                'patches': 'US_parcel-fabricated-2026',
            },
            {
                'admin_id': 'US-XX-YY',
                'recipe_id': 'US-XX-YY_footprint-fabricated-2026',
                'patches': 'US_footprint-fabricated-2026',
            },
        ]
    )
    monkeypatch.setattr('openplaces.diagnostics._recipe_index', lambda: index)
    assert find_recipe_patches('US_parcel-fabricated-2026', 'US-XX-YY') == [
        'US-XX_parcel-fabricated-2026',
        'US-XX-YY_parcel-fabricated-2026',
    ]
    assert find_recipe_patches('US_parcel-fabricated-2026', 'US-XX-QQ') == [
        'US-XX_parcel-fabricated-2026'
    ]
    assert find_recipe_patches('US_parcel-fabricated-2026', 'US-ZZ-AA') == [
        'US-ZZ_parcel-fabricated-2026'
    ]


def test_a_patch_recipe_is_never_the_entity_recipe_of_its_county(tmp_path, monkeypatch):
    """A county's patch file ranks finer than the national recipe it
    patches; auto-discovery must skip it, or the county curates with a
    two-line pipeline."""
    from openplaces import recipe as recipe_module

    root = tmp_path / 'recipes'
    national = root / 'US' / '_all' / 'parcel' / 'fabricated' / '2026'
    county = root / 'US' / 'XX' / 'YY' / '_all' / 'parcel' / 'fabricated' / '2026'
    national.mkdir(parents=True)
    county.mkdir(parents=True)
    (national / 'US_parcel-fabricated-2026.yaml').write_text(
        'admin_id: US\nstage: curate\nentity:\n  entity_type: parcel\n'
        '  source:\n    source_id: fabricated\n  version: "2026"\n'
        'entity_recipe: US_parcel-spine-2026\npipeline:\n  - step: order_columns\n',
        encoding='utf-8',
    )
    (county / 'US-XX-YY_parcel-fabricated-2026.yaml').write_text(
        'admin_id: US-XX-YY\nstage: curate\nentity:\n  entity_type: parcel\n'
        '  source:\n    source_id: fabricated\n  version: "2026"\n'
        'patches: US_parcel-fabricated-2026\npipeline_patch:\n'
        '  - remove: order_columns\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(recipe_module, 'recipe_roots', lambda: [root])
    monkeypatch.setattr('openplaces.path.recipe_roots', lambda: [root])
    found = find_entity_recipe_id(
        'US-XX-YY', 'parcel', stage='curate', source_id='fabricated', silent=True
    )
    assert found == 'US_parcel-fabricated-2026'
