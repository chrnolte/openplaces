"""Sphinx extension rendering the package state overview.

The ``recipe-state`` directive turns the committed recipe tree into the
headline numbers of the catalog: how many recipes exist, how many sources
they draw on, which geographies they cover, how far each has moved along
the pipeline, and how many have had their source's terms of use recorded.
Every number is counted at build time by ``catalog_data``, so the page
cannot drift from the catalog it describes -- adding a recipe updates it.
The headline figures and table rendering are shared with the
``recipe_catalog`` extension's ``recipe-coverage``/``recipe-children``
directives, via ``catalog_data.headline_lines``/``table_lines``.

The same counts print from the command line::

    python docs/_ext/catalog_data.py
"""

from catalog_data import (
    coverage_detail,
    doc_path,
    headline_lines,
    summarize,
    table_lines,
)
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.typing import ExtensionMetadata


class RecipeState(Directive):
    """Render the catalog-wide state of the package.

    Takes no arguments: the whole committed recipe tree is summarized.
    """

    has_content = False
    required_arguments = 0

    def run(self) -> list[nodes.Node]:
        state = summarize()
        lines = headline_lines(state)

        coverage_rows = []
        for item in state.coverage:
            path = doc_path(item.admin_id).as_posix()
            coverage_rows.append(
                [
                    f':doc:`{item.name} </recipes/{path}>`',
                    str(item.n_recipes),
                    coverage_detail(item),
                ]
            )

        lines += ['Coverage', '~' * 8, '']
        lines += table_lines(
            'Recipes by geography',
            ['Geography', 'Recipes', 'Covered in detail'],
            coverage_rows,
        )
        lines += [
            'A country row counts every recipe written for it, including those',
            'scoped to one of its subdivisions. Global recipes apply anywhere.',
            '',
        ]

        lines += ['Pipeline stages', '~' * 15, '']
        lines += table_lines(
            'Recipes by stage',
            ['Stage', 'Recipes'],
            [[stage, str(n)] for stage, n in state.by_stage.items()],
        )

        lines += ['What the recipes describe', '~' * 25, '']
        for label_category, labels in state.by_label.items():
            lines += table_lines(
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


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive('recipe-state', RecipeState)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
