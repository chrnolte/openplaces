"""Outlines drawn tall that no measured height explains."""

from __future__ import annotations

import pandas as pd

from openplaces.io.curator import CurateState
from openplaces.io.curator.evidence import flag_unconfirmed_height


def _state(frame):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=None,
        verbose=False,
        timer=None,
        curated=frame,
    )


def test_a_tall_building_is_confirmed_and_a_low_one_is_not():
    sqft = 10.7639
    frame = pd.DataFrame(
        {
            # $10,000 per sq ft of outline each
            'structure_value': [
                1e4 * 100 * sqft,
                1e4 * 100 * sqft,
                1e4 * 100 * sqft,
                100 * 100 * sqft,
                0.0,
            ],
            'area_m2': [100.0, 100.0, 100.0, 100.0, 100.0],
            'n_stories_overture': [10.0, 1.0, None, None, None],
            'height_overture': [None, None, None, None, None],
        },
        index=['TOWER', 'ANNEX', 'UNKNOWN', 'CHEAP', 'NO_VALUE'],
    )
    out = flag_unconfirmed_height(
        _state(frame),
        stories_column='n_stories_overture',
        height_column='height_overture',
    ).curated['height_unconfirmed']
    assert not out['TOWER']  # $1,000 per sq ft of floor
    assert out['ANNEX']
    assert out['UNKNOWN']  # no measured height: one floor
    assert not out['CHEAP']
    assert pd.isna(out['NO_VALUE'])


def test_a_reconciled_story_count_confirms_an_unmeasured_building():
    sqft = 10.7639
    frame = pd.DataFrame(
        {
            # $5,000 per sq ft of outline each, no measured height
            'structure_value': [5e3 * 1000 * sqft, 5e3 * 1000 * sqft],
            'area_m2': [1000.0, 1000.0],
            'n_stories_overture': [None, None],
            'n_stories': [8.0, None],
        },
        index=['CONDO', 'BOX'],
    )
    out = flag_unconfirmed_height(
        _state(frame),
        stories_column='n_stories_overture',
        reconciled_stories_column='n_stories',
    ).curated['height_unconfirmed']
    assert not out['CONDO']  # $625 per sq ft of floor
    assert out['BOX']


def test_a_height_alone_gives_the_floors():
    sqft = 10.7639
    frame = pd.DataFrame(
        {
            'structure_value': [1e4 * 100 * sqft],
            'area_m2': [100.0],
            'height_overture': [35.0],  # 10 floors at 3.5 m
        },
        index=['TOWER'],
    )
    out = flag_unconfirmed_height(
        _state(frame), height_column='height_overture'
    ).curated['height_unconfirmed']
    assert not out['TOWER']


def test_a_lot_whose_value_fits_its_floors_releases_its_storied_buildings():
    sqft = 10.7639
    # One lot, value split by area: $6,500 per sq ft of outline each.
    # Over the lot's floor area (2 + 3 + 1 floors) it comes to $2,676.
    v = 6_500 * sqft
    frame = pd.DataFrame(
        {
            'structure_value': [v * 500, v * 500, v * 50],
            'area_m2': [500.0, 500.0, 50.0],
            'n_stories': [2.0, 3.0, 1.0],
            'lot': ['L', 'L', 'L'],
        },
        index=['TWO', 'THREE', 'KIOSK'],
    )
    out = flag_unconfirmed_height(
        _state(frame), reconciled_stories_column='n_stories', lot_column='lot'
    ).curated['height_unconfirmed']
    assert not out['TWO']  # $3,250 per floor alone, but the lot fits
    assert not out['THREE']
    assert out['KIOSK']  # under 100 m2: still judged alone
    alone = flag_unconfirmed_height(
        _state(frame.copy()), reconciled_stories_column='n_stories'
    ).curated['height_unconfirmed']
    assert alone['TWO']
