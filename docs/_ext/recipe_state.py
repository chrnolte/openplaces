"""Sphinx extension rendering the package state overview.

The ``recipe-state`` directive turns the committed recipe tree into the
headline numbers of the catalog: how many recipes exist, how many sources
they draw on, which geographies they cover, how far each has moved along
the pipeline, and how many have had their source's terms of use recorded.
Every number is counted at build time by ``catalog_data``, so the page
cannot drift from the catalog it describes -- adding a recipe updates it.

The same counts print from the command line::

    python docs/_ext/catalog_data.py
"""

from catalog_data import CatalogState, coverage_detail, summarize
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.typing import ExtensionMetadata

#: Card headline / caption for each figure in the summary row.
_HEADLINE_CAPTIONS = {
    'n_recipes': 'recipes',
    'n_sources': 'data sources',
    'n_countries': 'countries',
    'n_geographies': 'geographies',
}


def _headline_lines(state: CatalogState) -> list[str]:
    """Rst lines for the row of headline figures.

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


def _table_lines(title: str, headers: list[str], rows: list[list[str]]) -> list[str]:
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


class RecipeState(Directive):
    """Render the catalog-wide state of the package.

    Takes no arguments: the whole committed recipe tree is summarized.
    """

    has_content = False
    required_arguments = 0

    def run(self) -> list[nodes.Node]:
        state = summarize()
        lines = _headline_lines(state)

        lines += ['Coverage', '~' * 8, '']
        lines += _table_lines(
            'Recipes by geography',
            ['Geography', 'Recipes', 'Covered in detail'],
            [
                [
                    f':doc:`{item.name} </recipes/{_doc_slug(item)}>`'
                    if item.admin_id != 'Global'
                    else ':doc:`Global </recipes/global>`',
                    str(item.n_recipes),
                    coverage_detail(item),
                ]
                for item in state.coverage
            ],
        )
        lines += [
            'A country row counts every recipe written for it, including those',
            'scoped to one of its subdivisions. Global recipes apply anywhere.',
            '',
        ]

        lines += ['Pipeline stages', '~' * 15, '']
        lines += _table_lines(
            'Recipes by stage',
            ['Stage', 'Recipes'],
            [[stage, str(n)] for stage, n in state.by_stage.items()],
        )

        lines += ['What the recipes describe', '~' * 25, '']
        for label_category, labels in state.by_label.items():
            lines += _table_lines(
                label_category,
                ['Type' if label_category == 'Entities' else 'Theme', 'Recipes'],
                [[label, str(n)] for label, n in labels.items()],
            )

        lines += ['Source terms', '~' * 12, '']
        recorded = state.n_terms_recorded
        share = 100 * recorded / state.n_recipes if state.n_recipes else 0
        lines += [
            f'{recorded} of {state.n_recipes} recipes ({share:.0f}%) record the terms',
            f'their source publishes under, and {state.n_restricted} are marked as',
            'restricting redistribution. An unrecorded source is one nobody has',
            'checked yet, which is not the same as an unrestricted one -- see the',
            ':ref:`recipe writing guide <writing_recipes>`.',
            '',
        ]

        node = nodes.section()
        node.document = self.state.document
        content = StringList(lines, source='recipe-state')
        nested_parse_with_titles(self.state, content, node)
        return node.children


def _doc_slug(item) -> str:
    """Return the catalog page slug of a country.

    Imported lazily from `recipe_catalog` so the page paths stay defined
    in the extension that writes them.

    Parameters
    ----------
    item : catalog_data.Coverage
        One country's coverage.

    Returns
    -------
    str
        Document path relative to `recipes/`, without extension.
    """
    from recipe_catalog import _doc_path

    return _doc_path(item.admin_id).as_posix()


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive('recipe-state', RecipeState)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
