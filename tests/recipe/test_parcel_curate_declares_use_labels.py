"""The curated parcel always carries the land-use labels the footprint reads.

The parcel spine's derive_use_classes writes no label where no source
brought a use code (Virginia's VGIN-only units), and the footprint
recipe's link_curated_entity raises on a missing column. The parcel
curate declares the labels so that the link finds them, all-null.
"""

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.formatters import declare_columns
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id

PARCEL = 'US_parcel-openplaces-2026'
FOOTPRINT = 'US_footprint-openplaces-2026'


def _steps(recipe_id, admin):
    recipe, _ = apply_recipe_patches(get_recipe_by_id(recipe_id), AdminId(admin))
    return recipe['pipeline']


def _linked_use_labels(admin):
    """Use-label columns the footprint links from the curated parcel."""
    return {
        ref_col
        for step in _steps(FOOTPRINT, admin)
        if step['step'] == 'link_curated_entity'
        and step.get('recipe_id') == PARCEL
        and not step.get('allow_missing')
        for ref_col in step['columns']
        if ref_col.startswith('use_group')
    }


def test_footprint_use_labels_are_declared_on_the_parcel():
    for admin in ('US-VA-FC', 'US-VA-AL', 'US-DC-DC'):
        linked = _linked_use_labels(admin)
        assert linked >= {'use_group_combined', 'use_group_combined_labeled'}
        declared = {
            col
            for step in _steps(PARCEL, admin)
            if step['step'] == 'declare_columns'
            for col in step['columns']
        }
        assert linked <= declared, sorted(linked - declared)


def test_declaring_keeps_a_label_that_exists():
    curated = pd.DataFrame(
        {'use_group_code': ['Single Family'], 'use_group_combined': ['Townhouse']}
    )
    state = CurateState(
        recipe={},
        entity_recipe=None,
        admin_id=AdminId('US-VA-FC'),
        verbose=False,
        timer=None,
        curated=curated,
    )
    out = declare_columns(
        state, columns=['use_group_combined', 'use_group_combined_labeled']
    ).curated
    assert out['use_group_combined'].tolist() == ['Townhouse']
    assert out['use_group_combined_labeled'].isna().all()
