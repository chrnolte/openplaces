"""Reader and summary statistics for the committed recipe catalog.

Two Sphinx extensions build on this module: 'recipe_catalog' renders one
page per geography plus the catalog landing page's coverage overview and
clickable geography tables, 'recipe_state' renders the package-wide
stage/label breakdown. Both share this module's rst-fragment helpers
(`headline_lines`, `table_lines`) so the two pages render the same kind
of figure the same way. The docs build does not have openplaces
installed (it is not, e.g., on ReadTheDocs), so recipes are read as
plain YAML and schema constants are parsed statically with ast. That
also makes the module runnable on its own, printing the same overview
the docs page shows:

    python docs/_ext/catalog_data.py
"""

import ast
import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import yaml

try:
    # Routes warnings into the build log when running under Sphinx; the
    # module must still import when it is run as a script.
    from sphinx.util import logging

    logger = logging.getLogger(__name__)
except ImportError:  # pragma: no cover - script use
    import logging

    logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES_DIR = REPO_ROOT / 'src' / 'openplaces' / 'recipes'

#: Pipeline stages, in the order a dataset moves through them. A recipe
#: without a `stage` field pre-dates it and is an ingest recipe.
STAGES = ('ingest', 'harmonize', 'enrich', 'curate')

#: Label categories, in reading order: what the data is about first,
#: what it measures second.
CATEGORIES = ('Entities', 'Datasets', 'Other')

#: Admin level labels used when describing coverage in prose.
LEVEL_LABELS = {2: 'states / regions', 3: 'counties / districts', 4: 'municipalities'}

#: Mini-summary columns for the clickable geography tables (landing page
#: and every generated country/state page): a display label paired with
#: the entity type(s) it counts. 'Buildings' covers both the polygon
#: footprint entity and the point building entity, since either answers
#: "is there building coverage here".
SUMMARY_COLUMNS = (
    ('Admin', frozenset({'admin'})),
    ('Parcels', frozenset({'parcel'})),
    ('Properties', frozenset({'property'})),
    ('Buildings', frozenset({'footprint', 'building'})),
    ('Sales', frozenset({'transaction'})),
)


@cache
def _schema_constants() -> tuple[frozenset[str], frozenset[str]]:
    """Extract ENTITY_TYPES and TOP_LEVEL_THEMES from core/schema.py.

    Parsed statically with ast so the docs build does not need the
    openplaces package installed (it is not, e.g., on ReadTheDocs).

    Returns
    -------
    tuple of frozenset
        The entity types and the top-level dataset themes.
    """
    schema_path = REPO_ROOT / 'src' / 'openplaces' / 'core' / 'schema.py'
    constants: dict[str, frozenset[str]] = {}
    for node in ast.parse(schema_path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in (
                'ENTITY_TYPES',
                'TOP_LEVEL_THEMES',
            ):
                constants[target.id] = frozenset(ast.literal_eval(node.value))
    for name in ('ENTITY_TYPES', 'TOP_LEVEL_THEMES'):
        if name not in constants:
            logger.warning('catalog: %s not found in %s', name, schema_path)
            constants[name] = frozenset()
    return constants['ENTITY_TYPES'], constants['TOP_LEVEL_THEMES']


def category(label: str) -> str:
    """Classify a group label as Entities, Datasets, or Other.

    Parameters
    ----------
    label : str
        Entity type or dataset theme.

    Returns
    -------
    str
        'Entities', 'Datasets', or 'Other'.
    """
    entity_types, top_level_themes = _schema_constants()
    if label in entity_types:
        return 'Entities'
    if label.split('-')[0] in top_level_themes:
        return 'Datasets'
    return 'Other'


@cache
def admin_names(level: int) -> dict[str, str]:
    """Map admin IDs of one level to names from the latest admin spine CSV.

    Parameters
    ----------
    level : int
        Admin level (1 = country, 2 = state/region, ...).

    Returns
    -------
    dict
        Admin ID to name; empty when no spine CSV is committed.
    """
    pattern = f'_all/admin/spine/*/admin-spine-*_admin{level}.csv'
    spines = sorted(RECIPES_DIR.glob(pattern))
    if not spines:
        logger.warning('catalog: no admin%d spine CSV found', level)
        return {}
    with spines[-1].open(encoding='utf-8-sig', newline='') as f:
        return {row[f'admin{level}_id']: row['name'] for row in csv.DictReader(f)}


def display_name(admin_id: str) -> str:
    """Return the spine name of a geography ('Global' for global recipes).

    Parameters
    ----------
    admin_id : str
        Admin ID, or 'Global'.

    Returns
    -------
    str
        Human-readable name, falling back to the ID itself.
    """
    if admin_id == 'Global':
        return 'Global'
    level = admin_id.count('-') + 1
    return admin_names(level).get(admin_id, admin_id)


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def doc_path(admin_id: str) -> Path:
    """Relative document path of a geography page (without extension).

    'Global' -> global; 'US' -> united-states; 'US-MA' ->
    united-states/massachusetts. Shared by the page generator
    ('recipe_catalog') and every directive that links to a generated
    page, so the two can never compute a different path for the same id.

    Parameters
    ----------
    admin_id : str
        Admin ID, or 'Global'.

    Returns
    -------
    Path
        Document path relative to the ``recipes/`` directory.
    """
    if admin_id == 'Global':
        return Path('global')
    parts = admin_id.split('-')
    slugs = [_slug(display_name('-'.join(parts[: i + 1]))) for i in range(len(parts))]
    return Path(*slugs)


def load_recipes() -> list[tuple[str, str, Path, dict]]:
    """Return (admin_id, recipe_id, path, recipe) for every recipe YAML.

    The admin id is derived from the path components before the first
    '_all' escape directory ('Global' for global recipes, else e.g. 'US'
    or 'US-MA').

    Returns
    -------
    list of tuple
        One entry per parsable recipe, sorted by path.
    """
    entries = []
    for path in sorted(RECIPES_DIR.rglob('*.yaml')):
        try:
            recipe = yaml.safe_load(path.read_text(encoding='utf-8'))
        except yaml.YAMLError as exc:
            logger.warning('catalog: cannot parse %s (%s)', path, exc)
            continue
        if not isinstance(recipe, dict):
            continue
        rel = path.relative_to(RECIPES_DIR)
        admin_parts = []
        for part in rel.parts[:-1]:
            if part == '_all':
                break
            admin_parts.append(part)
        admin_id = '-'.join(admin_parts) or 'Global'
        entries.append((admin_id, path.stem, path, recipe))
    return entries


def entity_label(recipe: dict) -> str:
    """Return a recipe's entity type, or its dataset theme, or ''.

    Parameters
    ----------
    recipe : dict
        Raw recipe mapping.

    Returns
    -------
    str
        Entity type, theme, or an empty string when neither is declared.
    """
    entity = recipe.get('entity')
    if isinstance(entity, dict) and entity.get('entity_type'):
        return str(entity['entity_type'])
    dataset = recipe.get('dataset')
    if isinstance(dataset, dict) and dataset.get('theme'):
        return str(dataset['theme'])
    return ''


def source_block(recipe: dict) -> dict:
    """Return a recipe's `source:` mapping, or an empty dict.

    Covers both the entity and dataset slots, and the compact string form
    ('massgis' rather than a mapping), which is normalized to a mapping
    holding only the source ID.

    Parameters
    ----------
    recipe : dict
        Raw recipe mapping.

    Returns
    -------
    dict
        The source mapping; empty when the recipe declares no source.
    """
    container = recipe.get('entity') or recipe.get('dataset')
    if not isinstance(container, dict):
        return {}
    source = container.get('source')
    if isinstance(source, dict):
        return source
    if isinstance(source, str):
        return {'source_id': source}
    return {}


def stage_of(recipe: dict) -> str:
    """Return a recipe's stage, defaulting to 'ingest'.

    Parameters
    ----------
    recipe : dict
        Raw recipe mapping.

    Returns
    -------
    str
        One of `STAGES`, or the raw value for an unrecognized stage.
    """
    stage = recipe.get('stage')
    return str(stage) if stage else 'ingest'


@dataclass(frozen=True)
class Coverage:
    """Recipe coverage of one country (or of the global scope).

    Attributes
    ----------
    admin_id : str
        Country admin ID, or 'Global'.
    name : str
        Display name from the admin spine.
    n_recipes : int
        Recipes scoped to this country, including its subdivisions.
    n_by_level : dict
        Count of distinct admin units with their own recipes, keyed by
        admin level (2 = states/regions, 3 = counties, ...). Level 1 is
        omitted: it is the country itself.
    """

    admin_id: str
    name: str
    n_recipes: int
    n_by_level: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CatalogState:
    """Catalog-wide counts backing the package state overview.

    Attributes
    ----------
    n_recipes : int
        Total recipes committed to the catalog.
    n_sources : int
        Distinct source IDs across all recipes.
    n_geographies : int
        Distinct admin units that have at least one recipe of their own
        ('Global' excluded).
    n_countries : int
        Countries with at least one recipe, at any admin level.
    by_stage : dict
        Recipe count per pipeline stage.
    by_label : dict
        Recipe count per entity type / dataset theme, nested under the
        category ('Entities', 'Datasets', 'Other').
    coverage : list of Coverage
        Per-country coverage, most recipes first; 'Global' last.
    n_terms_recorded : int
        Recipes whose source records a `license` value.
    n_restricted : int
        Recipes whose source is marked `redistribution_restricted: true`.
    """

    n_recipes: int
    n_sources: int
    n_geographies: int
    n_countries: int
    by_stage: dict[str, int]
    by_label: dict[str, dict[str, int]]
    coverage: list[Coverage]
    n_terms_recorded: int
    n_restricted: int


def summarize(entries: list[tuple[str, str, Path, dict]] | None = None) -> CatalogState:
    """Compute the catalog-wide counts shown on the package state page.

    Parameters
    ----------
    entries : list of tuple, optional
        Output of `load_recipes`. Loaded when not supplied.

    Returns
    -------
    CatalogState
        Counts derived from the committed recipe tree.
    """
    entries = load_recipes() if entries is None else entries

    by_stage: dict[str, int] = {}
    by_label: dict[str, dict[str, int]] = {}
    sources: set[str] = set()
    geographies: set[str] = set()
    # Country admin ID to the recipe count of that country and
    # everything under it, and to its sub-units with their own recipes.
    country_recipes: dict[str, int] = {}
    country_units: dict[str, dict[int, set[str]]] = {}
    n_terms_recorded = 0
    n_restricted = 0

    for admin_id, _recipe_id, _path, recipe in entries:
        by_stage[stage_of(recipe)] = by_stage.get(stage_of(recipe), 0) + 1

        label = entity_label(recipe) or 'other'
        by_label.setdefault(category(label), {})
        by_label[category(label)][label] = by_label[category(label)].get(label, 0) + 1

        source = source_block(recipe)
        if source.get('source_id'):
            sources.add(str(source['source_id']))
        if source.get('license'):
            n_terms_recorded += 1
        if source.get('redistribution_restricted') is True:
            n_restricted += 1

        country = 'Global' if admin_id == 'Global' else admin_id.split('-')[0]
        country_recipes[country] = country_recipes.get(country, 0) + 1
        country_units.setdefault(country, {})
        if admin_id != 'Global':
            geographies.add(admin_id)
            level = admin_id.count('-') + 1
            if level > 1:
                country_units[country].setdefault(level, set()).add(admin_id)

    coverage = [
        Coverage(
            admin_id=country,
            name=display_name(country),
            n_recipes=n,
            n_by_level={
                level: len(units)
                for level, units in sorted(country_units[country].items())
            },
        )
        for country, n in country_recipes.items()
    ]
    # Global recipes apply everywhere, so they sort last rather than
    # competing with countries for the top of the table.
    coverage.sort(key=lambda c: (c.admin_id == 'Global', -c.n_recipes, c.name))

    return CatalogState(
        n_recipes=len(entries),
        n_sources=len(sources),
        n_geographies=len(geographies),
        n_countries=sum(1 for c in coverage if c.admin_id != 'Global'),
        by_stage={s: by_stage[s] for s in STAGES if s in by_stage}
        | {s: n for s, n in sorted(by_stage.items()) if s not in STAGES},
        by_label={
            k: dict(sorted(by_label[k].items())) for k in CATEGORIES if k in by_label
        },
        coverage=coverage,
        n_terms_recorded=n_terms_recorded,
        n_restricted=n_restricted,
    )


def _ancestors_inclusive(admin_id: str) -> list[str]:
    """Return an admin id and every one of its ancestors, coarsest first.

    Parameters
    ----------
    admin_id : str
        Admin ID, or 'Global'.

    Returns
    -------
    list of str
        E.g. ``'US-NC-WI'`` -> ``['US', 'US-NC', 'US-NC-WI']``;
        ``'Global'`` -> ``['Global']``.
    """
    if admin_id == 'Global':
        return ['Global']
    parts = admin_id.split('-')
    return ['-'.join(parts[: i + 1]) for i in range(len(parts))]


@cache
def all_admin_ids() -> frozenset[str]:
    """Every admin id with its own generated catalog page.

    This is every admin id carrying a recipe of its own, plus every
    ancestor implied by it (e.g. a recipe scoped to 'US-NC-WI' also
    implies pages for 'US' and 'US-NC'). 'Global' is excluded -- it is
    handled as a page of its own everywhere this is used, the way
    `summarize` treats it as a `Coverage` row outside the country list.

    Returns
    -------
    frozenset of str
        Every non-Global admin id that has (or will have) a page under
        ``recipes/``.
    """
    ids: set[str] = set()
    for admin_id, *_ in load_recipes():
        if admin_id != 'Global':
            ids.update(_ancestors_inclusive(admin_id))
    return frozenset(ids)


def children_of(admin_id: str | None) -> list[str]:
    """Direct children of a geography, or the top-level countries.

    Parameters
    ----------
    admin_id : str or None
        Parent admin id, or ``None`` for the top-level (country) list.

    Returns
    -------
    list of str
        Sorted admin ids exactly one level below `admin_id`.
    """
    ids = all_admin_ids()
    if admin_id is None:
        return sorted(i for i in ids if '-' not in i)
    depth = admin_id.count('-') + 2
    prefix = f'{admin_id}-'
    return sorted(i for i in ids if i.startswith(prefix) and i.count('-') + 1 == depth)


@dataclass(frozen=True)
class ChildSummary:
    """One row of a clickable geography table: a child and its subtree.

    Attributes
    ----------
    admin_id : str
        The child's admin id, or 'Global'.
    name : str
        Display name from the admin spine.
    n_recipes : int
        Recipes scoped to this geography or nested under it.
    by_column : dict
        Recipe count per `SUMMARY_COLUMNS` label, same rollup as
        `n_recipes`.
    has_children : bool
        Whether this geography itself has a further level to drill into.
    """

    admin_id: str
    name: str
    n_recipes: int
    by_column: dict[str, int]
    has_children: bool


def children_summary(
    admin_id: str | None, entries: list[tuple[str, str, Path, dict]] | None = None
) -> list[ChildSummary]:
    """Summarize the direct children of a geography for a catalog table.

    Each child's counts are rolled up over its whole subtree (a state's
    row counts its counties too), computed in one pass over every recipe
    rather than one filter pass per child, so this stays cheap even with
    thousands of leaf geographies.

    Parameters
    ----------
    admin_id : str or None
        Parent admin id, or ``None`` for the top-level (country + Global)
        list shown on the catalog landing page.
    entries : list of tuple, optional
        Output of `load_recipes`. Loaded when not supplied.

    Returns
    -------
    list of ChildSummary
        Most recipes first; 'Global' last when it appears (top level
        only), matching `summarize`'s `coverage` ordering.
    """
    entries = load_recipes() if entries is None else entries

    total: Counter[str] = Counter()
    by_col: dict[str, Counter[str]] = {label: Counter() for label, _ in SUMMARY_COLUMNS}
    for entry_admin_id, _recipe_id, _path, recipe in entries:
        label = entity_label(recipe)
        for ancestor in _ancestors_inclusive(entry_admin_id):
            total[ancestor] += 1
            for col_label, types in SUMMARY_COLUMNS:
                if label in types:
                    by_col[col_label][ancestor] += 1

    results = [
        ChildSummary(
            admin_id=child,
            name=display_name(child),
            n_recipes=total[child],
            by_column={label: by_col[label][child] for label, _ in SUMMARY_COLUMNS},
            has_children=bool(children_of(child)),
        )
        for child in children_of(admin_id)
    ]
    if admin_id is None and total['Global']:
        results.append(
            ChildSummary(
                admin_id='Global',
                name='Global',
                n_recipes=total['Global'],
                by_column={
                    label: by_col[label]['Global'] for label, _ in SUMMARY_COLUMNS
                },
                has_children=False,
            )
        )
    results.sort(key=lambda c: (c.admin_id == 'Global', -c.n_recipes, c.name))
    return results


def coverage_detail(item: Coverage) -> str:
    """Describe a country's sub-unit coverage as one short phrase.

    Parameters
    ----------
    item : Coverage
        One country's coverage.

    Returns
    -------
    str
        E.g. '5 states / regions, 12 counties / districts', or an em dash
        when only country-wide recipes exist.
    """
    parts = [
        f'{n} {LEVEL_LABELS.get(level, f"level-{level} units")}'
        for level, n in item.n_by_level.items()
    ]
    return ', '.join(parts) if parts else '—'


#: Card headline / caption for each figure in a headline row.
_HEADLINE_CAPTIONS = {
    'n_recipes': 'recipes',
    'n_sources': 'data sources',
    'n_countries': 'countries',
    'n_geographies': 'geographies',
}


def headline_lines(state: CatalogState) -> list[str]:
    """Rst lines for a row of catalog-wide headline figures.

    Shared by the 'recipe_state' and 'recipe_catalog' Sphinx extensions,
    so the package-state page and the catalog landing page show the same
    figures without duplicating the sphinx-design markup.

    Parameters
    ----------
    state : CatalogState
        Computed catalog counts.

    Returns
    -------
    list of str
        A sphinx-design grid of one card per figure.
    """
    lines = ['.. grid:: 2 2 4 4', '   :gutter: 2', '']
    for attribute, caption in _HEADLINE_CAPTIONS.items():
        lines += [
            '   .. grid-item-card::',
            '      :text-align: center',
            '',
            f'      **{getattr(state, attribute)}**',
            '',
            f'      {caption}',
            '',
        ]
    return lines


def table_lines(title: str, headers: list[str], rows: list[list[str]]) -> list[str]:
    """Rst lines for one list-table.

    Parameters
    ----------
    title : str
        Table caption.
    headers : list of str
        Column headers.
    rows : list of list of str
        Cell values, already rendered as strings.

    Returns
    -------
    list of str
        Lines of a list-table directive, empty when there are no rows.
    """
    if not rows:
        return []
    lines = [
        f'.. list-table:: {title}',
        '   :header-rows: 1',
        '   :widths: auto',
        '',
    ]
    for row in [headers, *rows]:
        lines.append(f'   * - {row[0]}')
        lines += [f'     - {cell}' for cell in row[1:]]
    lines.append('')
    return lines


def format_overview(state: CatalogState | None = None) -> str:
    """Render the package state overview as plain text.

    Parameters
    ----------
    state : CatalogState, optional
        Computed state. Calculated when not supplied.

    Returns
    -------
    str
        Multi-line report, as printed by this module's CLI.
    """
    state = summarize() if state is None else state
    width = max((len(c.name) for c in state.coverage), default=10)

    lines = [
        'openplaces recipe catalog',
        '=' * 25,
        '',
        f'{state.n_recipes} recipes  |  {state.n_sources} sources  |  '
        f'{state.n_countries} countries  |  {state.n_geographies} geographies',
        '',
        'Stages',
        '-' * 6,
    ]
    lines += [f'  {stage:<12}{n:>5}' for stage, n in state.by_stage.items()]

    lines += ['', 'Coverage', '-' * 8]
    for item in state.coverage:
        lines.append(
            f'  {item.name:<{width}}  {item.n_recipes:>4}  {coverage_detail(item)}'
        )

    for label_category, labels in state.by_label.items():
        lines += ['', label_category, '-' * len(label_category)]
        lines += [f'  {label:<24}{n:>5}' for label, n in labels.items()]

    recorded = state.n_terms_recorded
    share = 100 * recorded / state.n_recipes if state.n_recipes else 0
    lines += [
        '',
        'Source terms',
        '-' * 12,
        f'  license recorded          {recorded:>5} of {state.n_recipes}'
        f' ({share:.0f}%)',
        f'  redistribution restricted {state.n_restricted:>5}',
        '',
    ]
    return '\n'.join(lines)


def main() -> None:
    """Print the package state overview."""
    print(format_overview())


if __name__ == '__main__':
    main()
