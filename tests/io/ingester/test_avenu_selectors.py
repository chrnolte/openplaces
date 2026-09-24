"""The results-row selector must survive a row being clicked.

Clicking a row to load its details re-classes it to
`DataGridSelectedRow`. A selector that does not match that class loses
the clicked row from its own match set, and since `parse_results`
addresses rows by index, the set changing under the loop makes `i`
stop denoting a fixed row: rows get skipped, re-read, or paired with a
neighbour's book and page.
"""

from __future__ import annotations

from openplaces.io.scrapers.avenu_selectors import RESULTS_ROW


def test_results_row_matches_the_selected_row_class():
    """Measured on Cambridge 2024-03: 20 rows matched, 19 after a click.

    The missing class cost real records rather than only mispairing
    them: with it absent, book/page 82529/537 was never read at all,
    the walk going straight from 82529/247 to 82529/583.
    """
    for css_class in (
        'DataGridRow',
        'DataGridAlternatingRow',
        'DataGridSelectedRow',
    ):
        assert f'tr.{css_class}' in RESULTS_ROW, (
            f'RESULTS_ROW does not match tr.{css_class}; a row carrying '
            'that class leaves the match set and the index-addressed '
            'walk in parse_results desyncs'
        )


def test_results_row_stays_inside_the_results_container():
    """Every alternative is scoped, so no stray table row can match.

    The grid is one table among several on the page; an unscoped
    `tr.DataGridRow` would pick up whatever else the registry renders.
    """
    alternatives = [part.strip() for part in RESULTS_ROW.split(',')]
    assert len(alternatives) >= 3
    for alternative in alternatives:
        assert alternative.startswith('#DocList1_ContentContainer1 '), (
            f'{alternative!r} is not scoped to the results container'
        )
