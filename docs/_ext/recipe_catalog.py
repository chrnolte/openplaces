"""Sphinx extension rendering a catalog of openplaces recipes.

At build start, one document per geography is generated under ``recipes/``
(countries at the top, subdivisions nested, e.g.
``united-states/massachusetts.rst``), picked up by glob toctrees so the
geographies appear in the theme's section navigation. Each page runs the
``recipe-catalog`` directive, which walks
``src/openplaces/recipes/**/*.yaml`` and renders one entry per recipe of
that geography: its id, scope, stage, source portal link, and the prose
``description:`` field (parsed as reStructuredText). Recipes without a
description get a one-line stub entry.

The landing page (``docs/recipes.rst``) additionally uses two directives
defined here: ``recipe-coverage`` (headline figures plus the three
coverage maps) and ``recipe-children`` (a compact per-geography summary
table). The latter also replaces the plain title list every generated
geography page used to show for its own subdivisions -- a table with a
handful of counts per row stays readable at any scale, where a bare list
of titles does not once a country has more than a few dozen of them.

Reading the recipe tree lives in ``catalog_data``, shared with the
``recipe_state`` extension.
"""

from pathlib import Path

from catalog_data import (
    CATEGORIES,
    REPO_ROOT,
    SUMMARY_COLUMNS,
    all_admin_ids,
    category,
    children_summary,
    display_name,
    doc_path,
    entity_label,
    headline_lines,
    load_recipes,
    source_block,
    summarize,
    table_lines,
)
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from gh_file import GITHUB_BASE
from sphinx.application import Sphinx
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.typing import ExtensionMetadata


def _portal_url(recipe: dict) -> str | None:
    return source_block(recipe).get('portal_url')


def _entry_lines(recipe_id: str, path: Path, recipe: dict) -> list[str]:
    """Definition-list rst lines for one recipe.

    The geography and entity type are not repeated here: both are implicit
    in the recipe id and in the page and section the entry appears under.
    """
    rel = path.relative_to(REPO_ROOT).as_posix()
    term = f'`{recipe_id} <{GITHUB_BASE}{rel}>`__'
    # Badge only the stages beyond the implicit default ('ingest').
    if recipe.get('stage') and recipe['stage'] != 'ingest':
        term += f' :bdg-secondary-line:`{recipe["stage"]}`'

    lines = [term]
    description = recipe.get('description')
    if isinstance(description, str) and description.strip():
        for desc_line in description.strip().splitlines():
            lines.append(f'   {desc_line}' if desc_line else '')
    else:
        lines.append('   *No description yet.*')
    portal = _portal_url(recipe)
    if portal:
        lines.append('')
        lines.append(f'   `Original source <{portal}>`__')
    lines.append('')
    return lines


class RecipeCatalog(Directive):
    """Render one geography's recipes as Entities/Datasets/type sections.

    Takes the admin id as argument (e.g. 'US' or 'US-MA'; 'Global' for
    global recipes). Only recipes scoped exactly to that geography are
    rendered; finer-grained recipes appear on their own subdivision pages.
    """

    has_content = False
    required_arguments = 1

    def run(self) -> list[nodes.Node]:
        code = self.arguments[0]
        grouped: dict[str, dict[str, list]] = {}
        for entry_admin_id, recipe_id, path, recipe in load_recipes():
            if entry_admin_id != code:
                continue
            label = entity_label(recipe) or 'other'
            grouped.setdefault(category(label), {}).setdefault(label, []).append(
                (recipe_id, path, recipe)
            )

        lines: list[str] = [
            '.. contents::',
            '   :local:',
            '   :depth: 2',
            '',
        ]
        for label_category in CATEGORIES:
            labels = grouped.get(label_category)
            if not labels:
                continue
            lines.append(label_category)
            lines.append('~' * len(label_category))
            lines.append('')
            for label in sorted(labels):
                lines.append(label)
                lines.append('^' * len(label))
                lines.append('')
                for recipe_id, path, recipe in sorted(labels[label]):
                    lines.extend(_entry_lines(recipe_id, path, recipe))

        node = nodes.section()
        node.document = self.state.document
        content = StringList(lines, source='recipe-catalog')
        nested_parse_with_titles(self.state, content, node)
        return node.children


#: (image stem, alt text, caption), stacked in this order. Kept beside
#: the directive that renders them, not in `generate_coverage_maps.py`,
#: since that script's own `MAPS` list already carries the matching
#: title baked into each PNG -- duplicating it here would drift.
_INGEST_MAPS = (
    ('recipe_coverage_admin', 'Map of admin boundary recipe coverage', 'Admin'),
    (
        'recipe_coverage_footprint',
        'Map of building footprint recipe coverage',
        'Buildings',
    ),
    ('recipe_coverage_parcel', 'Map of parcel recipe coverage', 'Parcels'),
)


class RecipeCoverage(Directive):
    """Render the catalog-wide headline figures and coverage maps.

    Takes no arguments: the whole committed recipe tree is summarized.
    The headline figures are computed at build time from the recipe
    tree, same as `RecipeChildren` below. The maps are not: the docs
    build has neither openplaces nor ingested boundary data available
    (`catalog_data`'s own module docstring explains why), so they're
    pre-rendered snapshots committed as static images, refreshed by
    rerunning ``docs/_ext/generate_coverage_maps.py`` in a real
    environment -- a standalone script, not tied to any notebook, so the
    docs don't depend on notebook conventions or a notebook staying in
    sync with what the catalog needs shown.
    """

    has_content = False
    required_arguments = 0

    def run(self) -> list[nodes.Node]:
        lines = headline_lines(summarize())
        lines += [
            'The three maps below cover **data ingestion** recipes -- the',
            "recipes that download and structure each entity type's raw",
            'source data, one entity type at a time. Maps are stacked full',
            'width, cropped to the covered area, so coverage stays legible',
            'even where it is a single county.',
            '',
        ]
        for stem, alt, caption in _INGEST_MAPS:
            lines += [
                f'.. figure:: /_static/images/{stem}.png',
                '   :width: 100%',
                f'   :alt: {alt}',
                '',
                f'   {caption}',
                '',
            ]

        lines += [
            'Ingested footprints and parcels are then linked together in',
            '``harmonize`` and ``curate``. The map below is that linked',
            "entity's recipe coverage -- US only for now -- rather than",
            'ingestion; note that a country-scoped recipe colors the whole',
            'country, which is coarser than the counties actually curated.',
            '',
            '.. figure:: /_static/images/recipe_coverage_footprint_parcel_linked.png',
            '   :width: 100%',
            '   :alt: Map of footprints linked to parcels, harmonize/curate',
            '',
            '   Footprints linked to parcels (harmonize / curate, US only)',
            '',
        ]

        node = nodes.section()
        node.document = self.state.document
        content = StringList(lines, source='recipe-coverage')
        nested_parse_with_titles(self.state, content, node)
        return node.children


class RecipeChildren(Directive):
    """Render one geography's children as a compact summary table.

    Takes an optional admin id argument; omitted renders the top-level
    countries (plus 'Global', when it carries recipes of its own). Each
    row's geography name links to that child's own generated page, so
    the tree is browsed one level at a time -- the only way a table this
    compact stays readable once a country's subdivisions run into the
    thousands, which is already true of admin4 municipalities and will
    become true of admin3 counties as coverage grows.
    """

    has_content = False
    required_arguments = 0
    optional_arguments = 1

    def run(self) -> list[nodes.Node]:
        parent = self.arguments[0] if self.arguments else None
        headers = ['Geography', 'Recipes', *(label for label, _ in SUMMARY_COLUMNS)]
        rows = [
            [
                f':doc:`{child.name} </recipes/{doc_path(child.admin_id).as_posix()}>`',
                str(child.n_recipes),
                *(
                    str(child.by_column[label]) if child.by_column[label] else ''
                    for label, _ in SUMMARY_COLUMNS
                ),
            ]
            for child in children_summary(parent)
        ]
        lines = table_lines('Recipes by geography', headers, rows)

        node = nodes.section()
        node.document = self.state.document
        content = StringList(lines, source='recipe-children')
        nested_parse_with_titles(self.state, content, node)
        return node.children


def _generate_geography_pages(app: Sphinx) -> None:
    """Write one rst page per geography into {srcdir}/recipes/.

    Real documents (rather than sections of one page) are required for the
    geographies to appear in the theme's section navigation, whose toctree
    shows document titles only. Subdivisions nest below their parent
    (united-states/massachusetts.rst); a page with subdivisions shows them
    as a ``recipe-children`` summary table, backed by a hidden glob toctree
    that still builds the page tree and populates the sidebar. Files are
    only rewritten when their content changes, and stale pages of removed
    geographies are pruned.
    """
    recipe_admin_ids = {admin_id for admin_id, *_ in load_recipes()}
    admin_ids = set(all_admin_ids())
    if 'Global' in recipe_admin_ids:
        admin_ids.add('Global')

    has_children = {
        '-'.join(admin_id.split('-')[:-1]) for admin_id in admin_ids if '-' in admin_id
    }

    out_dir = Path(app.srcdir) / 'recipes'
    out_dir.mkdir(exist_ok=True)

    keep = set()
    for admin_id in sorted(admin_ids):
        name = display_name(admin_id)
        rel = doc_path(admin_id).with_suffix('.rst')
        keep.add(rel.as_posix())
        lines = [
            '.. generated by recipe_catalog.py - do not edit',
            '',
            name,
            '=' * len(name),
            '',
            f'.. recipe-catalog:: {admin_id}',
            '',
        ]
        if admin_id in has_children:
            lines += [
                'Subdivisions',
                '------------',
                '',
                f'.. recipe-children:: {admin_id}',
                '',
                '.. toctree::',
                '   :hidden:',
                '   :titlesonly:',
                '   :glob:',
                '',
                f'   {rel.stem}/*',
                '',
            ]
        content = '\n'.join(lines)
        page = out_dir / rel
        page.parent.mkdir(parents=True, exist_ok=True)
        if not page.exists() or page.read_text(encoding='utf-8') != content:
            page.write_text(content, encoding='utf-8')

    for stale in out_dir.rglob('*.rst'):
        if stale.relative_to(out_dir).as_posix() not in keep:
            stale.unlink()
    subdirs = [p for p in out_dir.rglob('*') if p.is_dir()]
    for subdir in sorted(subdirs, reverse=True):
        if not any(subdir.iterdir()):
            subdir.rmdir()


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive('recipe-catalog', RecipeCatalog)
    app.add_directive('recipe-coverage', RecipeCoverage)
    app.add_directive('recipe-children', RecipeChildren)
    app.connect('builder-inited', _generate_geography_pages)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
