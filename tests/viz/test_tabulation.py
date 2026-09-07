"""Tests for the cross-tabulation behind the stacked bar chart."""

import pandas as pd
import pytest

from openplaces.viz import tabulation
from openplaces.viz.tabulation import _NASTR, plot_tabulation, tabulate


@pytest.fixture
def half_missing():
    """Four rows, half of them missing their y label."""
    return pd.DataFrame(
        {
            'occupancy_type': ['Single-Family', 'Manufactured Home', None, None],
            'group': ['a', 'b', 'a', 'b'],
        }
    )


def test_show_empty_category_survives_a_categorical_dtype(half_missing):
    """Curate casts categoricals, so this is the common case.

    A categorical column cannot be filled before the category exists, so
    the missing rows were dropped by the groupby and the result was then
    renormalized over the survivors: half the rows vanished and the table
    still summed to 1.
    """
    as_object = tabulate(half_missing, y_cat='occupancy_type', x_cat='group')
    frame = half_missing.astype({'occupancy_type': 'category', 'group': 'category'})
    as_category = tabulate(frame, y_cat='occupancy_type', x_cat='group')

    assert _NASTR in as_object.index
    assert _NASTR in as_category.index
    assert as_category.loc[_NASTR].sum() == pytest.approx(0.5)
    assert as_category.sum().sum() == pytest.approx(1.0)
    assert as_category.loc['Single-Family'].sum() == pytest.approx(0.25)


def test_show_empty_category_false_still_drops_missing(half_missing):
    frame = half_missing.astype({'occupancy_type': 'category'})
    table = tabulate(
        frame, y_cat='occupancy_type', x_cat='group', show_empty_category=False
    )
    assert _NASTR not in table.index


def test_palette_match_uses_per_column_weights(monkeypatch):
    """Weights are per stacked (x) category, not per bar.

    The call passed one weight per y row and the zip inside
    `match_palette` truncated to the shorter side, so a chart whose two
    rare labels hold a few percent of the rows computed coverage as 0
    and lost the curated palette.
    """
    captured = {}
    original = tabulation.match_palette

    def _spy(values, col_name=None, weights=None, **kwargs):
        captured['values'] = [str(v) for v in values]
        captured['weights'] = None if weights is None else list(weights)
        return original(values, col_name=col_name, weights=weights, **kwargs)

    monkeypatch.setattr(tabulation, 'match_palette', _spy)

    frame = pd.DataFrame(
        {
            'occupancy_type': ['Single-Family'] * 98 + ['rare-1', 'rare-2'],
            # Many more y groups than x categories, which is what let the
            # zip pair weights with the wrong labels.
            'county': [f'c{i % 20}' for i in range(100)],
        }
    )
    plot_tabulation(frame, y_cat='county', x_cat='occupancy_type')

    assert captured['weights'] is not None
    assert len(captured['weights']) == len(captured['values'])
    weight_of = dict(zip(captured['values'], captured['weights'], strict=True))
    assert weight_of['Single-Family'] == pytest.approx(0.98, abs=0.01)
