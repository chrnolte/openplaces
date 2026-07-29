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
"""

import ast
import csv
import re
from functools import cache
from pathlib import Path

import yaml
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from gh_file import GITHUB_BASE
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.typing import ExtensionMetadata

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES_DIR = REPO_ROOT / 'src' / 'openplaces' / 'recipes'


@cache
def _schema_constants() -> tuple[frozenset[str], frozenset[str]]:
    """Extract ENTITY_TYPES and TOP_LEVEL_THEMES from core/schema.py.

    Parsed statically with ast so the docs build does not need the
    openplaces package installed (it is not, e.g., on ReadTheDocs).
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
            logger.warning('recipe-catalog: %s not found in %s', name, schema_path)
            constants[name] = frozenset()
    return constants['ENTITY_TYPES'], constants['TOP_LEVEL_THEMES']


def _category(label: str) -> str:
    """Classify a group label as Entities, Datasets, or Other."""
    entity_types, top_level_themes = _schema_constants()
    if label in entity_types:
        return 'Entities'
    if label.split('-')[0] in top_level_themes:
        return 'Datasets'
    return 'Other'


@cache
def _admin_names(level: int) -> dict[str, str]:
    """Map admin IDs of one level to names from the latest admin spine CSV."""
    pattern = f'_all/admin/spine/*/admin-spine-*_admin{level}.csv'
    spines = sorted(RECIPES_DIR.glob(pattern))
    if not spines:
        logger.warning('recipe-catalog: no admin%d spine CSV found', level)
        return {}
    with spines[-1].open(encoding='utf-8-sig', newline='') as f:
        return {row[f'admin{level}_id']: row['name'] for row in csv.DictReader(f)}


def _display_name(admin_id: str) -> str:
    """Return the spine name of a geography ('Global' for global recipes)."""
    if admin_id == 'Global':
        return 'Global'
    level = admin_id.count('-') + 1
    return _admin_names(level).get(admin_id, admin_id)


def _load_recipes() -> list[tuple[str, str, Path, dict]]:
    """Return (admin_id, recipe_id, path, recipe) for every recipe YAML.

    The admin id is derived from the path components before the first
    '_all' escape directory ('Global' for global recipes, else e.g. 'US'
    or 'US-MA').
    """
    entries = []
    for path in sorted(RECIPES_DIR.rglob('*.yaml')):
        try:
            recipe = yaml.safe_load(path.read_text(encoding='utf-8'))
        except yaml.YAMLError as exc:
            logger.warning('recipe-catalog: cannot parse %s (%s)', path, exc)
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


def _entity_label(recipe: dict) -> str:
    entity = recipe.get('entity')
    if isinstance(entity, dict) and entity.get('entity_type'):
        return str(entity['entity_type'])
    dataset = recipe.get('dataset')
    if isinstance(dataset, dict) and dataset.get('theme'):
        return str(dataset['theme'])
    return ''


def _portal_url(recipe: dict) -> str | None:
    container = recipe.get('entity') or recipe.get('dataset')
    if not isinstance(container, dict):
        return None
    source = container.get('source')
    if isinstance(source, dict):
        return source.get('portal_url')
    return None


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
        for entry_admin_id, recipe_id, path, recipe in _load_recipes():
            if entry_admin_id != code:
                continue
            label = _entity_label(recipe) or 'other'
            grouped.setdefault(_category(label), {}).setdefault(label, []).append(
                (recipe_id, path, recipe)
            )

        lines: list[str] = [
            '.. contents::',
            '   :local:',
            '   :depth: 2',
            '',
        ]
        for category in ('Entities', 'Datasets', 'Other'):
            labels = grouped.get(category)
            if not labels:
                continue
            lines.append(category)
            lines.append('~' * len(category))
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


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def _doc_path(admin_id: str) -> Path:
    """Relative document path of a geography page (without extension).

    'Global' → global; 'US' → united-states; 'US-MA' →
    united-states/massachusetts.
    """
    if admin_id == 'Global':
        return Path('global')
    parts = admin_id.split('-')
    slugs = [_slug(_display_name('-'.join(parts[: i + 1]))) for i in range(len(parts))]
    return Path(*slugs)


def _generate_geography_pages(app: Sphinx) -> None:
    """Write one rst page per geography into {srcdir}/recipes/.

    Real documents (rather than sections of one page) are required for the
    geographies to appear in the theme's section navigation, whose toctree
    shows document titles only. Subdivisions nest below their parent
    (united-states/massachusetts.rst) and are referenced from a glob
    toctree on the parent page. Files are only rewritten when their
    content changes, and stale pages of removed geographies are pruned.
    """
    admin_ids = {admin_id for admin_id, *_ in _load_recipes()}
    for admin_id in list(admin_ids):
        if admin_id == 'Global':
            continue
        parts = admin_id.split('-')
        admin_ids.update('-'.join(parts[:i]) for i in range(1, len(parts)))

    has_children = {
        '-'.join(admin_id.split('-')[:-1])
        for admin_id in admin_ids
        if admin_id != 'Global' and '-' in admin_id
    }

    out_dir = Path(app.srcdir) / 'recipes'
    out_dir.mkdir(exist_ok=True)

    keep = set()
    for admin_id in sorted(admin_ids):
        name = _display_name(admin_id)
        rel = _doc_path(admin_id).with_suffix('.rst')
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
                '.. toctree::',
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
    app.connect('builder-inited', _generate_geography_pages)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
