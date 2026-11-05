"""Write the column reference of a delivery bundle from the code, so the
documented columns cannot drift from what a bundle ships.

A bundle's canonical columns are the recipe's own statement (its
`share: columns:` and `share: point_columns:`, read through
io.delivery.share_columns), plus the few columns the delivery writes by
itself: a `{column}_source` sidecar for each canonical column, the
geometry source, and the adjusted value and appraisal year where it
imputes (io.building_imputation.ADJUSTED_COLUMNS). What each column
means, its unit and its type come
from the attribute registry, resolved through the provenance suffix the
way every stage resolves them. Nothing is described here: a column the
registry describes badly is fixed in the registry.

Given a delivered bundle, the page also lists the evidence file's
columns, read from the parquet footer alone (no rows), and the
canonical columns follow the bundle's own order. The page is dated by
the bundle's modification time, never by the run, so regenerating from
unchanged inputs reproduces the page byte for byte.

Run as ``python -m openplaces.flow.delivery_columns RECIPE_ID
[--bundle CANONICAL_PARQUET] [--out PATH] [--described LABEL=COL,...]``.
A column the surrounding page explains (--described) links to that
explanation instead of repeating the registry's text. It exits 1 when a
canonical column has no registry description
(tests/flow/test_delivery_columns.py holds the same rule).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from openplaces.core.attribute_registry import load_registry
from openplaces.core.schema import ENTITY_TYPES
from openplaces.io.building_imputation import ADJUSTED_COLUMNS
from openplaces.io.delivery import share_columns
from openplaces.io.delivery.slicing import PARENT_SUFFIX
from openplaces.recipe import get_recipe_by_id, resolve_attribute_name

SOURCE_SUFFIX = '_source'
PER_AREA_SUFFIX = '_per_area'
ALL_SUFFIX = '_all'
EVIDENCE_SUFFIX = '_evidence'
POINT_SUFFIX = '_point'

DEFAULT_DIR = Path('docs/3_examples/_generated')

#: Admin levels as core.schema numbers them (AGENTS.md, Identifiers).
ADMIN_LEVELS = {
    1: 'country',
    2: 'state',
    3: 'county (in New England, the town)',
    4: 'municipality or county subdivision',
}


@dataclass(frozen=True)
class Column:
    """One documented column of a bundle."""

    name: str
    description: str
    unit: str
    data_type: str
    files: str
    has_source: bool | None
    derived_from: str


def describe(column: str) -> tuple[str, str, str]:
    """Description, unit and data type of *column*, or blanks.

    Registered names (after the provenance suffix is resolved) read
    their registry row. Six kinds of name are keys or derived columns
    the registry does not list by design, and are described by rule: an
    admin unit id, an entity id, a linked record's id, a
    `{column}_per_area` ratio, a `{column}_all` list, and a source's own
    `{column}_source_{source}` field.
    """
    registry = load_registry()
    base = resolve_attribute_name(column)
    if base in registry.index:
        row = registry.loc[base]
        text = '' if pd.isna(row['description']) else str(row['description'])
        if base != column:
            text = f'{text} ({_suffix_note(column, base)})' if text else ''
        unit = '' if pd.isna(row['unit']) else str(row['unit'])
        return text, unit, str(row['data_type'])
    admin = re.fullmatch(r'admin(\d)_id', column)
    if admin and int(admin.group(1)) in ADMIN_LEVELS:
        level = int(admin.group(1))
        return (
            f'openplaces id of the admin unit at level {level} '
            f'({ADMIN_LEVELS[level]}) the row lies in',
            '',
            'string',
        )
    # The id a piece was cut from where the delivery sliced an entity
    # two units delivered (io.delivery.slicing.parent_column).
    parent = re.fullmatch(rf'([a-z]+)_id{PARENT_SUFFIX}', column)
    if parent and parent.group(1) in ENTITY_TYPES:
        return (
            f'Set only on the pieces of a {parent.group(1)} that two '
            'neighboring units both delivered and the delivery cut at the '
            f'line between them: the id of the {parent.group(1)} the piece '
            "was cut from; each piece keeps its own unit's values",
            '',
            'string',
        )
    entity = re.fullmatch(r'([a-z]+)_id', column)
    if entity and entity.group(1) in ENTITY_TYPES:
        return (
            f'Unique id of the {entity.group(1)}; every file of the bundle '
            'carries it, in the same row order, so any two files join on it',
            '',
            'string',
        )
    # The id a source gives the record linked to this row
    # (building_id_nsi): an id, so not a registry attribute.
    linked = re.fullmatch(r'([a-z]+)_id_([a-z_]+)', column)
    if linked and linked.group(1) in ENTITY_TYPES:
        source = linked.group(2).replace('_', ' ')
        return f'Id of the linked record in the {source} source', '', 'string'
    if column.endswith(PER_AREA_SUFFIX):
        numerator = column[: -len(PER_AREA_SUFFIX)]
        text, unit, _ = describe(numerator)
        if text:
            per = f'{unit} per m2' if unit else 'per m2'
            return f'{numerator} divided by area_m2', per, 'float'
    # Every distinct value where a row links to several records
    # (evidence.collect_link_ids and the harmonizer's join convention):
    # missing where it would repeat the base column.
    if column.endswith(ALL_SUFFIX):
        base_column = column[: -len(ALL_SUFFIX)]
        if describe(base_column)[0]:
            return (
                f'Every distinct value of {base_column} where the row links to '
                'more than one record, joined by a vertical bar; missing where '
                f'it would repeat {base_column}',
                '',
                'string',
            )
    # A source's own statement of where one of its values came from
    # (year_built_source_footprint_ncdps), not an openplaces sidecar.
    own = re.fullmatch(r'(.+)_source_([a-z_]+)', column)
    if own and own.group(1) in load_registry().index:
        source = own.group(2).replace('_', ' ')
        return (
            f'Where the {source} source says its {own.group(1)} came from, in '
            "the source's own words",
            '',
            'string',
        )
    return '', '', ''


#: Evidence groups, in page order: the last token of a column's
#: provenance suffix names the source that supplied it. Columns
#: without one are the footprint's own measures, its places, and what
#: openplaces derived from several sources.
EVIDENCE_GROUPS = {
    'parcel': 'From the county tax roll and parcel map',
    'nsi': 'From the National Structure Inventory (NSI)',
    'overture': 'From Overture (address points, building heights)',
    'fema': "From FEMA's USA Structures",
    'cheer': 'From the earlier CHEER inventory',
    'ncdps': "From North Carolina's statewide building footprints (NC DPS)",
    '': "The footprint's own measures, places and derived columns",
}


def evidence_source(column: str) -> str:
    """The source token that supplied evidence *column*, or ''.

    Read from the provenance suffix the way every stage resolves names
    (``n_dwellings_nsi`` is NSI's, ``year_built_parcel`` the roll's). A
    linked record's id names its source the same way
    (``building_id_nsi``), a source's own field names it after
    ``_source_`` (``year_built_source_footprint_ncdps``), and the
    parcel link ids (``parcel_id``, ``parcel_id_local``) are the roll's.
    """
    for suffix in (PER_AREA_SUFFIX, ALL_SUFFIX):
        if column.endswith(suffix):
            column = column[: -len(suffix)]
    own = re.fullmatch(r'.+_source_([a-z_]+)', column)
    if own and own.group(1).split('_')[-1] in EVIDENCE_GROUPS:
        return own.group(1).split('_')[-1]
    # The roll's ids, and its values for the whole lot
    # (improvement_value_parcel_whole), are the roll's; a count per
    # parcel (n_footprints_per_parcel) is openplaces' own.
    if column.startswith('parcel_id') or '_parcel_' in f'{column}_'.replace(
        '_per_parcel_', '_'
    ):
        return 'parcel'
    linked = re.fullmatch(r'([a-z]+)_id_([a-z_]+)', column)
    if linked and linked.group(1) in ENTITY_TYPES:
        token = linked.group(2).split('_')[-1]
        return token if token in EVIDENCE_GROUPS else ''
    base = resolve_attribute_name(column)
    if base == column or not column.startswith(base):
        return ''
    token = column[len(base) :].strip('_').split('_')[-1]
    # Only a source's token names a group; a trailing word of the name
    # itself (geometry_source_id) does not.
    return token if token in EVIDENCE_GROUPS else ''


def _suffix_note(column: str, base: str) -> str:
    suffix = column[len(base) :].lstrip('_')
    return f'as recorded by the {suffix.replace("_", " ")} source'


def canonical_columns(
    recipe: dict, bundle_columns: list[str] | None = None
) -> list[str]:
    """The canonical file's columns in delivered order, sidecars left out.

    Declared columns, then the columns the delivery writes by itself.
    With *bundle_columns*, the bundle's own order wins and any column it
    carries beyond those is kept, so the page shows what shipped.
    """
    columns, point_columns = share_columns(recipe)
    delivery = (recipe.get('share') or {}).get('delivery') or {}
    added = []
    if delivery.get('impute'):
        added += list(ADJUSTED_COLUMNS)
    expected = [*columns, *added, *point_columns]
    if bundle_columns is None:
        return expected
    shipped = [
        c
        for c in bundle_columns
        if not c.endswith(SOURCE_SUFFIX) and c not in ('geometry', 'bbox')
    ]
    return shipped + [c for c in expected if c not in shipped]


def column_rows(recipe: dict, bundle_columns: list[str] | None = None) -> list[Column]:
    """One Column per canonical column of *recipe*'s bundle."""
    _, point_columns = share_columns(recipe)
    derived = (recipe.get('share') or {}).get('derived_from') or {}
    sidecars = {
        c[: -len(SOURCE_SUFFIX)]
        for c in bundle_columns or []
        if c.endswith(SOURCE_SUFFIX)
    }
    rows = []
    for name in canonical_columns(recipe, bundle_columns):
        text, unit, data_type = describe(name)
        if name in point_columns:
            files = 'point'
        elif name in ('lat', 'long'):
            files = 'canonical (point: as geometry)'
        else:
            files = 'canonical, point'
        # Without a bundle it is unknown: a sidecar ships only where the
        # curated column carries one.
        has_source = name in sidecars if bundle_columns is not None else None
        rows.append(
            Column(
                name=name,
                description=text,
                unit=unit,
                data_type=data_type,
                files=files,
                has_source=has_source,
                derived_from=', '.join(derived.get(name) or []),
            )
        )
    return rows


def undescribed(rows: list[Column]) -> list[str]:
    """Canonical columns the registry leaves without a description."""
    return [r.name for r in rows if not r.description]


def _rst_table(frame: pd.DataFrame, title: str, widths: str | None = None) -> list[str]:
    lines = [f'.. list-table:: {title}', '   :header-rows: 1']
    if widths:
        lines.append(f'   :widths: {widths}')
    lines.append('')
    lines.append('   * - ' + '\n     - '.join(str(c) for c in frame.columns))
    for _, row in frame.iterrows():
        cells = ['' if pd.isna(v) else str(v) for v in row.tolist()]
        lines.append('   * - ' + '\n     - '.join(cells))
    lines.append('')
    return lines


def _rst_heading(title: str, char: str) -> list[str]:
    return [title, char * len(title), '']


def _parse_described(values: list[str]) -> dict[str, str]:
    """Map columns to labels from ``LABEL=col1,col2`` arguments."""
    out = {}
    for value in values:
        label, _, columns = value.partition('=')
        if not label or not columns:
            raise SystemExit(f'--described expects LABEL=col1,col2: {value!r}')
        for column in columns.split(','):
            out[column.strip()] = label.strip()
    return out


def _read_columns(path: Path) -> tuple[list[str], int]:
    import pyarrow.parquet as pq

    return pq.read_schema(path).names, pq.ParquetFile(path).metadata.num_rows


def render(
    recipe_id: str,
    bundle: Path | None = None,
    described: dict[str, str] | None = None,
    heading_char: str = '-',
) -> str:
    """The column reference of *recipe_id*'s bundle as reStructuredText.

    Parameters
    ----------
    recipe_id : str
        The curate recipe whose bundle is documented.
    bundle : Path, optional
        A delivered bundle's canonical parquet, to follow its column
        order and list its evidence file's columns.
    described : dict, optional
        Columns the page around the fragment already explains, mapped
        to the reST label of that explanation. Their row links there
        instead of repeating the registry's description.
    heading_char : str
        Underline character of the two file headings (canonical file,
        evidence file), one level below the heading the fragment is
        included under.
    """
    described = described or {}
    recipe = get_recipe_by_id(recipe_id)
    bundle_columns = evidence_columns = None
    if bundle is not None:
        bundle_columns, _ = _read_columns(bundle)
        evidence = bundle.with_name(bundle.stem + EVIDENCE_SUFFIX + bundle.suffix)
        if evidence.exists():
            evidence_columns, _ = _read_columns(evidence)
    rows = column_rows(recipe, bundle_columns)
    sources = f'the recipe {recipe_id} and core/attribute_registry.csv'
    if bundle is not None:
        written = datetime.fromtimestamp(bundle.stat().st_mtime).date()
        sources += f', and the bundle {bundle.name} written {written}'
    lines = [
        '.. Generated by python -m openplaces.flow.delivery_columns; do not edit.',
        f'   The sources are {sources}.',
        '',
    ]
    lines += _rst_heading('Canonical file', heading_char)
    canonical = pd.DataFrame(
        {
            'column': [f'``{r.name}``' for r in rows],
            'meaning': [
                f'See :ref:`{described[r.name]}`.'
                if r.name in described
                else r.description
                for r in rows
            ],
            'unit': [r.unit for r in rows],
            'type': [r.data_type for r in rows],
            '_source': ['yes' if r.has_source else '' for r in rows],
            'in files': [r.files for r in rows],
        }
    )
    lines += _rst_table(
        canonical,
        f'The {len(rows)} canonical columns',
        widths='18 52 6 8 6 10',
    )
    derived = [r for r in rows if r.derived_from]
    if derived:
        lines.append(
            'Computed from other columns, without a ``_source`` of their own, and'
            ' withheld wherever one of their inputs is:'
        )
        lines.append('')
        for r in derived:
            inputs = ', '.join(f'``{c}``' for c in r.derived_from.split(', '))
            lines.append(f'- ``{r.name}``: {inputs}.')
        lines.append('')
    if evidence_columns is not None:
        canonical_names = {r.name for r in rows}
        shown = [
            c
            for c in evidence_columns
            if not c.endswith(SOURCE_SUFFIX)
            and c not in ('geometry', 'bbox')
            and c not in canonical_names
        ]
        lines += _rst_heading('Evidence file', heading_char)
        lines.append(
            f'The evidence file holds {len(shown)} columns (and their '
            '``_source`` sidecars), grouped here by the source that '
            'supplied them.'
        )
        lines.append('')
        groups: dict[str, list[str]] = {}
        for column in shown:
            groups.setdefault(evidence_source(column), []).append(column)
        order = [g for g in EVIDENCE_GROUPS if g in groups]
        order += sorted(g for g in groups if g not in EVIDENCE_GROUPS)
        for group in order:
            members = groups[group]
            described = [describe(c) for c in members]
            evidence = pd.DataFrame(
                {
                    'column': [f'``{c}``' for c in members],
                    'meaning': [d[0] for d in described],
                    'unit': [d[1] for d in described],
                }
            )
            label = EVIDENCE_GROUPS.get(group, f'From {group}')
            lines += _rst_table(
                evidence, f'{label} ({len(members)})', widths='30 60 10'
            )
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('recipe_id')
    parser.add_argument(
        '--bundle',
        type=Path,
        default=None,
        help="a delivered bundle's canonical parquet, to follow its order and "
        'list its evidence columns',
    )
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument(
        '--described',
        action='append',
        default=[],
        metavar='LABEL=COL1,COL2',
        help='columns the surrounding page explains under the reST label '
        'LABEL; their rows link there instead of repeating the registry',
    )
    parser.add_argument(
        '--heading-char',
        default='-',
        help='underline character of the canonical and evidence file headings',
    )
    args = parser.parse_args(argv)
    out = args.out or DEFAULT_DIR / f'{args.recipe_id}_columns.rst'
    text = render(
        args.recipe_id,
        args.bundle,
        described=_parse_described(args.described),
        heading_char=args.heading_char,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf8')
    print(f'wrote {out} ({len(text.splitlines())} lines)')
    if args.bundle is not None:
        # Evidence gaps are reported, not enforced: the evidence file
        # carries whatever the sources supplied.
        evidence = args.bundle.with_name(
            args.bundle.stem + EVIDENCE_SUFFIX + args.bundle.suffix
        )
        if evidence.exists():
            names, _ = _read_columns(evidence)
            gaps = [
                c
                for c in names
                if not c.endswith(SOURCE_SUFFIX)
                and c not in ('geometry', 'bbox')
                and not describe(c)[0]
            ]
            if gaps:
                print('evidence without description: ' + ', '.join(gaps))
    missing = undescribed(column_rows(get_recipe_by_id(args.recipe_id)))
    if missing:
        print('no registry description: ' + ', '.join(missing))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
