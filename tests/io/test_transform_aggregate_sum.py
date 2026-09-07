"""The aggregate `sum` op's treatment of an all-missing row.

A parcel with no assessed value in any input column has an unknown total,
not a total of zero, so `sum` returns missing there, matching `min`, `max`
and `mean`. A recipe that wants zeros says so with `fill_na`, which fills
before the sum and is therefore unaffected.
"""

import numpy as np
import pandas as pd

from openplaces.io.transform import apply_transformations
from openplaces.recipe import get_recipe_by_id

RECIPE = {
    'transformations': [
        {
            'type': 'aggregate',
            'operation': 'sum',
            'inputs': ['land_value', 'improvement_value'],
            'output': 'total_value',
        }
    ]
}

FILLED_RECIPE = {
    'transformations': [
        {
            'type': 'aggregate',
            'operation': 'sum',
            'inputs': ['land_value', 'improvement_value'],
            'args': {'fill_na': 0},
            'output': 'total_value',
        }
    ]
}


def _frame():
    return pd.DataFrame(
        {
            'land_value': [100_000.0, 100_000.0, np.nan],
            'improvement_value': [250_000.0, np.nan, np.nan],
        }
    )


def test_a_row_with_no_value_in_any_input_stays_missing():
    result = apply_transformations(_frame(), RECIPE, silent=True)

    assert result['total_value'][0] == 350_000.0
    assert result['total_value'][1] == 100_000.0
    assert pd.isna(result['total_value'][2])


def test_fill_na_still_produces_a_number_for_an_all_missing_row():
    result = apply_transformations(_frame(), FILLED_RECIPE, silent=True)

    assert result['total_value'].tolist() == [350_000.0, 100_000.0, 0.0]


def test_the_north_dakota_recipe_path_is_unchanged():
    # The only shipping recipe using `sum` fills with zero first, so its
    # all-missing rows still total 0.0. Exercised through the recipe's own
    # transformation list rather than a copy of it.
    recipe = get_recipe_by_id('US-ND_property-ndgishub-2026')
    df = pd.DataFrame(
        {
            'AgriculturalLandValue': [40_000.0, np.nan],
            'ResidentialLandValue': [np.nan, np.nan],
            'CommercialLandValue': [10_000.0, np.nan],
            'ResidentialStructureValue': [np.nan, np.nan],
            'CommercialStructureValue': [70_000.0, np.nan],
        }
    )

    result = apply_transformations(df, recipe, silent=True)

    assert result['land_value_taxable'].tolist() == [50_000.0, 0.0]
    assert result['improvement_value_taxable'].tolist() == [70_000.0, 0.0]
