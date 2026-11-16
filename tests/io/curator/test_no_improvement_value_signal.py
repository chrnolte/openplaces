"""The no_improvement_value signal reads the figure structure_value carries.

A county patch that derives improvement from the total (Northampton
NC) leaves the roll's improvement value at 0 while structure_value is
filled; the signal must not fire there. Where structure_value is
missing (a roll with no values) the roll's 0 still fires. All values
are fabricated.
"""

from __future__ import annotations

import pandas as pd

from openplaces.io.curator.indicators import evaluate_indicator
from openplaces.recipe import get_recipe_by_id


def _signal():
    def walk(node):
        if isinstance(node, dict):
            if node.get('label') == 'no_improvement_value':
                yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    found = list(walk(get_recipe_by_id('US_building-openplaces-2026')['pipeline']))
    assert found, 'no_improvement_value is gone from the building curate'
    # Every vote that lists it uses one definition.
    assert all(f == found[0] for f in found)
    return found[0]


def _rows():
    return pd.DataFrame(
        {
            'parcel_id': ['p1', 'p2', 'p3', 'p4', 'p5'],
            # p1: patched county, roll 0, structure value derived.
            # p2: unvalued roll, structure value missing, roll 0.
            # p3: a valued house.
            # p4: a home holding a token share of its lot's value.
            # p5: a structure value of 0 the roll states.
            'improvement_value_parcel': [0.0, 0.0, 90_000.0, 500.0, 0.0],
            'improvement_value_parcel_whole': [0.0, 0.0, 90_000.0, 500.0, 0.0],
            'land_value_parcel_whole': [20_000.0, 0.0, 30_000.0, 40_000.0, 25_000.0],
            'structure_value': [64_000.0, None, 90_000.0, 500.0, 0.0],
            'improvement_value_imputed_parcel_whole': [
                64_000.0,
                None,
                90_000.0,
                500.0,
                0.0,
            ],
            'land_value_imputed_parcel_whole': [
                20_000.0,
                None,
                30_000.0,
                40_000.0,
                25_000.0,
            ],
        }
    )


def test_the_signal_follows_structure_value_and_falls_back_to_the_roll():
    fired = evaluate_indicator(_rows(), _signal())
    assert fired.tolist() == [False, True, False, True, True]


def test_without_a_structure_value_column_nothing_fires():
    rows = _rows().drop(
        columns=[
            'structure_value',
            'improvement_value_imputed_parcel_whole',
            'land_value_imputed_parcel_whole',
        ]
    )
    # An absent column is all-False for is_null too, so nothing fires;
    # the building curate always has structure_value at this step.
    assert not evaluate_indicator(rows, _signal()).any()
