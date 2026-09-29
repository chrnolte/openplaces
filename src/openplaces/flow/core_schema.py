"""Write the core schema page from the code, so the schema is read from
one place and cannot drift from what the pipeline writes.

The schema is the entities (core.schema.ENTITY_DEFINITIONS), the spine
recipe that establishes each, the attribute registry rows tagged to it
by entity and by the stage that first writes them, and the link tables
that relate entities (ENTITY_LINK_ORDER, the link-table columns and the
methods that fill them). None of that is declared here: each part is
imported from the module that owns it and rendered.

Two things the page reports that a hand-written page could not. The
registry rows that carry no entity or stage tag, which is the list
deliverable 1 of plans/core-schema-and-stage-contracts-review.md works
down. And, for one admin unit that has been built on this machine, the
columns each spine actually wrote, split into registered and not, so a
column a step invents outside the registry is visible the day it lands.

Run as ``python -m openplaces.flow.core_schema [--admin-id ID] [--out PATH]``.
It exits 0 whatever it finds: an untagged row or an unregistered column
is schema debt, not a broken pipeline; tests/recipe/test_core_schema.py
is where a limit is enforced.
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import pandas as pd

from openplaces.core.attribute_registry import load_registry
from openplaces.core.schema import ENTITY_DEFINITIONS, ENTITY_LINK_ORDER, ENTITY_TYPES
from openplaces.diagnostics import find_recipes

STAGES = ('ingest', 'harmonize', 'enrich', 'curate')

#: Labels link_entities_by_id writes into link_method, each naming the
#: exact-key rule that found the pair (io/harmonizer/entity_links.py).
#: Listed here rather than imported because that module defines them
#: inline where each pass is written; a test checks every label below
#: appears in its source, so the two cannot drift silently.
LINK_METHODS = {
    'parcel_id_local': 'the property row names the parcel by its local id',
    'stacked_units': 'a unit split off a parcel layer names the lot it stands on',
    'stacked_units_crosswalk': (
        'a roll row keyed on a unit reaches the lot the split saw that unit on'
    ),
}

DEFAULT_OUT = Path('docs/1_overview/concepts/_generated/core-schema.rst')


def entity_table() -> pd.DataFrame:
    """One row per entity type: definition, spines, recipe counts."""
    rows = []
    for entity_type in ENTITY_TYPES:
        recipes = find_recipes(entity_type)
        spines = sorted(
            r
            for r in recipes.loc[recipes['stage'] == 'harmonize', 'recipe_id']
            if 'spine' in r
        )
        counts = recipes['stage'].value_counts()
        rows.append(
            {
                'entity_type': entity_type,
                'one_row_is': ENTITY_DEFINITIONS[entity_type],
                'id_column': f'{entity_type}_id',
                'spines': ', '.join(spines) or '(none)',
                **{f'n_{s}': int(counts.get(s, 0)) for s in STAGES},
            }
        )
    return pd.DataFrame(rows)


def registry_table() -> pd.DataFrame:
    """The registry with blank tags made explicit."""
    registry = load_registry().reset_index()
    for tag in ('entity_type', 'stage'):
        registry[tag] = registry[tag].fillna('').astype(str)
    return registry


def untagged(registry: pd.DataFrame) -> pd.DataFrame:
    """Rows missing a stage tag.

    A blank entity tag is not a gap: it means the name is shared across
    entity types (the registry's own convention), so only the stage,
    which every column has exactly one of, is reported as missing.
    """
    return registry[registry['stage'] == '']


def observed_columns(admin_id: str, spines: list[str]) -> list[dict]:
    """What each spine wrote for *admin_id*, registered or not.

    A spine with no output for the unit reports ``built`` False rather
    than raising; the page says which spines it could check.
    """
    from openplaces.io.readers import describe_recipe

    out = []
    for recipe_id in spines:
        try:
            described = describe_recipe(recipe_id, admin_id)
        except (FileNotFoundError, ValueError) as error:
            out.append({'recipe_id': recipe_id, 'built': False, 'why': str(error)})
            continue
        kinds = described.index.to_series().map(
            lambda c: _column_kind(c, described.loc[c, 'attribute'])
        )
        out.append(
            {
                'recipe_id': recipe_id,
                'built': True,
                'n_rows': described.attrs.get('n_rows'),
                'n_columns': len(described),
                'n_keys': int((kinds == 'key').sum()),
                'n_sidecars': int((kinds == 'sidecar').sum()),
                'unregistered': sorted(kinds.index[kinds == 'unregistered']),
            }
        )
    return out


def _column_kind(column: str, attribute) -> str:
    """'registered', 'key' (an id or the join index), 'sidecar' (a
    provenance or conflict column of a registered attribute) or
    'unregistered'."""
    if attribute is not None and not pd.isna(attribute):
        return 'registered'
    # An id column with or without a provenance suffix (footprint_id,
    # building_id_nsi, parcel_id_local_all), the join index, the geometry.
    if column in ('_join_id', 'geometry') or re.search(r'(^|_)id(_|$)', column):
        return 'key'
    registry = load_registry()
    for suffix in ('_source', '_conflict', '_original'):
        if column.endswith(suffix):
            base = column[: -len(suffix)]
            from openplaces.recipe import resolve_attribute_name

            if base in registry.index or resolve_attribute_name(base) in registry.index:
                return 'sidecar'
    return 'unregistered'


def _rst_table(frame: pd.DataFrame, title: str) -> list[str]:
    lines = [f'.. list-table:: {title}', '   :header-rows: 1', '']
    lines.append('   * - ' + '\n     - '.join(str(c) for c in frame.columns))
    for _, row in frame.iterrows():
        cells = ['' if pd.isna(v) else str(v) for v in row.tolist()]
        lines.append('   * - ' + '\n     - '.join(cells))
    lines.append('')
    return lines


def render(admin_id: str | None) -> str:
    """The whole page as reStructuredText."""
    entities = entity_table()
    registry = registry_table()
    missing = untagged(registry)
    lines = [
        f'.. Generated by python -m openplaces.flow.core_schema on {date.today()};',
        '   do not edit. The sources are core/schema.py, core/attribute_registry.csv,',
        '   the recipe tree and io/harmonizer/entity_links.py.',
        '',
        'Core schema',
        '===========',
        '',
        'One row of each entity type means one thing, and harmonize establishes',
        'each entity in its spine recipe. Columns are the attribute registry rows,',
        'grouped by the stage that first writes them; a blank entity tag means the',
        'name is shared across entity types. Relationships are link tables.',
        '',
        'Entities',
        '--------',
        '',
    ]
    lines += _rst_table(entities, 'Entity types, their spines and recipe counts')
    lines += ['Columns by stage', '----------------', '']
    for stage in STAGES:
        rows = registry[registry['stage'] == stage]
        lines += [stage, '^' * len(stage), '']
        if rows.empty:
            lines += ['No registry row is tagged to this stage.', '']
            continue
        shown = rows[['name', 'entity_type', 'data_type', 'unit', 'aggregation']]
        lines += _rst_table(shown, f'{len(rows)} attributes first written by {stage}')
    lines += [
        'Links',
        '-----',
        '',
        'A link table holds one row per pair, stored beside the finer entity in',
        f'this order: {" < ".join(ENTITY_LINK_ORDER)}. Its columns are the two ids,',
        'link_method, link_source, share and share_basis, and nothing else, so a',
        'link can never carry a person. link_method is a fixed label naming the',
        'rule that found the pair, never a score, and no link is removed once',
        'written. A key on more rows than the placeholder cutoff keeps its pairs',
        'under the method with the suffix _shared_key; a reader that sums values',
        'leaves that method out.',
        '',
    ]
    for method, meaning in LINK_METHODS.items():
        lines.append(f'- ``{method}``: {meaning}.')
    lines.append('')
    lines += ['Untagged registry rows', '----------------------', '']
    shared = int((registry['entity_type'] == '').sum())
    lines.append(
        f'{len(missing)} of {len(registry)} rows lack a stage tag; {shared} carry'
        ' no entity tag, which means the name is shared across entity types.'
        ' A stage tag is the stage that first writes the column, and'
        ' tests/recipe/test_core_schema.py lets the untagged set only shrink.'
    )
    lines.append('')
    if not missing.empty:
        lines += _rst_table(missing[['name', 'entity_type']], 'Rows without a stage')
    if admin_id:
        spines = [
            s for row in entities['spines'] if row != '(none)' for s in row.split(', ')
        ]
        lines += [f'Observed columns, {admin_id}', '-' * (18 + len(admin_id)), '']
        for item in observed_columns(admin_id, spines):
            if not item['built']:
                lines.append(f'- ``{item["recipe_id"]}``: no output for this unit.')
                continue
            extra = item['unregistered']
            lines.append(
                f'- ``{item["recipe_id"]}``: {item["n_columns"]} columns on'
                f' {item["n_rows"]:,} rows ({item["n_keys"]} ids,'
                f' {item["n_sidecars"]} provenance or conflict sidecars);'
                f' {len(extra)} not in the registry'
                + (': ' + ', '.join(f'``{c}``' for c in extra) if extra else '')
                + '.'
            )
        lines.append('')
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        '--admin-id',
        default=None,
        help='a built unit whose spine outputs to describe (omit to skip)',
    )
    args = parser.parse_args(argv)
    text = render(args.admin_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding='utf8')
    print(f'wrote {args.out} ({len(text.splitlines())} lines)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
