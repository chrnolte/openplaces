"""A source's own foundation and construction words become one class.

The curate step `translate_descriptions` reads a descriptive column,
consults the row's own source override first and the shared keyword
table second, and writes the class the first matching row states. Only
reviewed rows write; a stop row or an unreviewed row leaves the column
silent without falling through. Every value, pattern and code below is
fabricated.
"""

from pathlib import Path

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState, translation
from openplaces.path import recipe_roots

SHARED = """\
# A header comment, as the shipped tables carry.
pattern,match_type,foundation_type,reviewed,note
NO BASEMENT,contains,,true,stop row
CRAWL,contains,C,true,crawl space
SLAB,contains,S,true,slab
PIER,contains,P,false,unreviewed
BLOCK,regex,C,true,block wall
"""

OVERRIDE_A = """\
column,pattern,match_type,foundation_type,reviewed,note
,CS,exact,S,true,a concrete slab in source a
,CRAWL SPACE,exact,P,true,source a's odd reading
,XX,exact,,true,stop row
,^Q\\d$,regex,B,true,a coded basement
"""

OVERRIDE_B = """\
column,pattern,match_type,foundation_type,reviewed,note
,CS,exact,C,true,a crawl space in source b
,PB,exact,P,false,unreviewed
,7,exact,S,true,a numeric code
"""


@pytest.fixture
def tables(tmp_path, monkeypatch):
    shared = tmp_path / 'shared.csv'
    shared.write_text(SHARED, encoding='utf-8')
    overrides = {}
    for source, text in {'srca': OVERRIDE_A, 'srcb': OVERRIDE_B}.items():
        path = tmp_path / f'{source}.csv'
        path.write_text(text, encoding='utf-8')
        overrides[f'recipe-{source}'] = path

    monkeypatch.setattr(translation, '_keywords_path', lambda state, name: shared)
    monkeypatch.setattr(
        translation,
        'resolve_source_recipe',
        lambda label, admin_id: {'srca': 'recipe-srca', 'srcb': 'recipe-srcb'}.get(
            label.split(':')[0]
        ),
    )
    monkeypatch.setattr(
        translation,
        'source_override_path',
        lambda recipe_id, suffix: overrides.get(recipe_id),
    )


def _run(curated, **kwargs):
    state = CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US', 'XX', 'YY'),
        verbose=False,
        timer=None,
        curated=curated,
    )
    params = {
        'output': 'foundation_type',
        'columns': ['foundation_description'],
        'keywords': 'foundation-type-keywords',
        'overrides': 'foundation-type-override',
    }
    params.update(kwargs)
    return translation.translate_descriptions(state, **params).curated


def test_shared_words_translate_and_name_the_source(tables):
    out = _run(
        pd.DataFrame(
            {
                'foundation_description': ['Conc Slab', 'FULL CRAWL', 'Dirt'],
                'source': ['srcc', 'srcc', 'srcc'],
            }
        )
    )
    assert out['foundation_type'].tolist()[:2] == ['S', 'C']
    assert pd.isna(out['foundation_type'].iloc[2])
    assert out['foundation_type_source'].tolist()[:2] == [
        'srcc+keywords',
        'srcc+keywords',
    ]
    assert pd.isna(out['foundation_type_source'].iloc[2])


def test_a_code_means_what_its_own_source_says(tables):
    out = _run(
        pd.DataFrame(
            {
                'foundation_description': ['CS', 'CS', 'CS'],
                'source': ['srca', 'srcb', 'x'],
            }
        )
    )
    assert out['foundation_type'].tolist()[:2] == ['S', 'C']
    # No source table names the code and the shared table has no row
    # for bare codes, so a third source's CS stays untranslated.
    assert pd.isna(out['foundation_type'].iloc[2])
    assert out['foundation_type_source'].tolist()[:2] == [
        'srca+override',
        'srcb+override',
    ]


def test_an_override_row_beats_the_shared_table(tables):
    out = _run(
        pd.DataFrame(
            {
                'foundation_description': ['Crawl Space', 'crawl space'],
                'source': ['srca', 'srcb'],
            }
        )
    )
    assert out['foundation_type'].tolist() == ['P', 'C']


def test_an_override_regex_row_matches(tables):
    out = _run(pd.DataFrame({'foundation_description': ['Q4'], 'source': ['srca']}))
    assert out['foundation_type'].tolist() == ['B']


def test_an_unreviewed_first_match_does_not_fall_through(tables):
    # PIER matches an unreviewed row; BLOCK, a later reviewed row, also
    # matches the same value and must not claim it.
    out = _run(
        pd.DataFrame({'foundation_description': ['PIER BLOCK'], 'source': ['srcc']})
    )
    assert pd.isna(out['foundation_type'].iloc[0])


def test_unreviewed_rows_write_when_asked(tables):
    out = _run(
        pd.DataFrame({'foundation_description': ['PIER BLOCK'], 'source': ['srcc']}),
        reviewed_only=False,
    )
    assert out['foundation_type'].tolist() == ['P']


def test_a_stop_row_silences_its_column_but_not_the_next(tables):
    out = _run(
        pd.DataFrame(
            {
                'foundation_description': ['NO BASEMENT', 'XX', 'NO BASEMENT'],
                'basement_note': ['SLAB', 'SLAB', None],
                'source': ['srcc', 'srca', 'srcc'],
            }
        ),
        columns=['foundation_description', 'basement_note'],
    )
    assert out['foundation_type'].tolist()[:2] == ['S', 'S']
    assert pd.isna(out['foundation_type'].iloc[2])


def test_merged_rows_try_each_source_in_label_order(tables):
    out = _run(
        pd.DataFrame(
            {
                'foundation_description': ['CS', 'CS', 'PB'],
                'source': ['srca+srcb:units', 'srcb+srca', 'srca+srcb:units'],
            }
        )
    )
    assert out['foundation_type'].tolist()[:2] == ['S', 'C']
    # srca has no row for PB; srcb's is unreviewed, which ends the search.
    assert pd.isna(out['foundation_type'].iloc[2])
    # The sidecar names bare source ids, so a delivery that withholds
    # srcb recognizes a value that may have come from its units.
    assert out['foundation_type_source'].iloc[0] == 'srca+srcb+override'


def test_a_value_already_stated_is_kept(tables):
    out = _run(
        pd.DataFrame(
            {
                'foundation_description': ['SLAB', 'SLAB'],
                'foundation_type': ['B', None],
                'source': ['srcc', 'srcc'],
            }
        )
    )
    assert out['foundation_type'].tolist() == ['B', 'S']
    assert pd.isna(out['foundation_type_source'].iloc[0])


def test_a_float_typed_code_matches_the_code_as_written(tables):
    # A code column read without a string cast arrives as 7.0; the
    # override says 7.
    out = _run(
        pd.DataFrame(
            {'foundation_description': [7.0, None], 'source': ['srcb', 'srcb']}
        )
    )
    assert out['foundation_type'].iloc[0] == 'S'
    assert pd.isna(out['foundation_type'].iloc[1])


def test_no_input_column_is_a_no_op(tables):
    curated = pd.DataFrame({'source': ['srca']})
    out = _run(curated)
    assert 'foundation_type' not in out.columns


def test_a_translation_is_never_marked_imputed(tables):
    from openplaces.io.curator.provenance import is_imputed

    out = _run(pd.DataFrame({'foundation_description': ['SLAB'], 'source': ['srca']}))
    assert not is_imputed(out['foundation_type_source']).any()


def _shipped_tables():
    targets = {
        'foundation-type': 'foundation_type',
        'construction-type': 'construction_type',
        'quality-class': 'quality_class',
        'condition-class': 'condition_class',
        'dwelling-type': 'dwelling_class',
    }
    found = []
    for root in recipe_roots():
        for path in sorted(Path(root).rglob('*.csv')):
            for variable, target in targets.items():
                if path.name.endswith(
                    (f'_{variable}-keywords.csv', f'_{variable}-override.csv')
                ):
                    found.append((path, target))
    return found


@pytest.mark.parametrize(
    ('path', 'target'),
    _shipped_tables(),
    ids=lambda value: value.name if isinstance(value, Path) else value,
)
def test_every_shipped_table_parses(path, target):
    rules = translation.load_translation_table(str(path), target)
    assert rules, f'{path.name} has no rows'
    if path.name.endswith('-keywords.csv'):
        # Shared tables name words; exact matching and a per-column
        # restriction belong to a source's own override.
        assert all(r.match_type in ('contains', 'regex') for r in rules)
        assert all(r.column is None for r in rules)
    else:
        assert all(r.match_type in ('exact', 'regex') for r in rules)
