"""One step decides a building's dwelling count, after its class.

plans/occupancy-and-dwellings-one-decider.md, O2 to O4 (rulings of
2026-11-21 and 2026-11-22). Every id, class, area and count below is
fabricated.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import _STEP_REGISTRY, CurateState, _load_steps
from openplaces.io.curator.estimators.dwellings import decide_dwellings
from openplaces.io.curator.indicators import indicator_columns
from openplaces.io.curator.provenance import is_imputed
from openplaces.io.steps import resolve_step_columns, undeclared_new_columns
from openplaces.recipe import get_recipe_by_id

STATED_ZERO = ['Commercial', 'Retail', 'Office', 'Industrial', 'Institutional']
BUILDING_RECIPE = 'US_building-openplaces-2026'


def _state(curated, recipe=None):
    return CurateState(
        recipe=recipe or {},
        entity_recipe={},
        admin_id=AdminId('US', 'XX', 'YY'),
        verbose=False,
        timer=None,
        curated=curated,
    )


def _buildings(rows):
    """rows: id -> (lot, area, class, lot class, lot count, prelim, token)."""
    columns = [
        'parcel_id',
        'area_m2',
        'occupancy_type',
        'land_use_class_parcel',
        'n_dwellings_parcel_whole',
        'n_dwellings_preliminary',
        'n_dwellings_preliminary_source',
    ]
    frame = pd.DataFrame.from_dict(rows, orient='index', columns=columns)
    frame.index.name = 'building_id'
    return frame


def _decide(frame, **params):
    params.setdefault('stated_zero_classes', STATED_ZERO)
    return decide_dwellings(_state(frame.copy()), **params).curated


def _imputed(token) -> bool:
    return bool(is_imputed(pd.Series([token])).iloc[0])


def test_a_retail_lot_whose_roll_states_zero_has_no_dwellings():
    out = _decide(
        _buildings({'b1': ('lot1', 300.0, 'Retail', 'Retail', 0.0, 2.0, 'overture')})
    )
    assert out.loc['b1', 'n_dwellings'] == 0
    # The roll's own value, not a count openplaces made up.
    assert out.loc['b1', 'n_dwellings_source'] == 'parcel'


def test_an_agricultural_lot_with_roll_zero_keeps_its_house():
    out = _decide(
        _buildings(
            {
                'b1': (
                    'lot1',
                    180.0,
                    'Single-Family',
                    'Agricultural',
                    0.0,
                    1.0,
                    'overture',
                )
            }
        )
    )
    assert out.loc['b1', 'n_dwellings'] == 1
    assert out.loc['b1', 'n_dwellings_source'] == 'overture'


@pytest.mark.parametrize(
    ('lot_class', 'stated'),
    [(None, 0.0), ('Retail', np.nan), ('Vacant', 0.0)],
    ids=['unknown use', 'no stated count', 'vacant'],
)
def test_a_zero_without_a_non_residential_statement_keeps_the_estimate(
    lot_class, stated
):
    out = _decide(
        _buildings(
            {'b1': ('lot1', 150.0, 'Single-Family', lot_class, stated, 1.0, 'nsi')}
        )
    )
    assert out.loc['b1', 'n_dwellings'] == 1
    assert out.loc['b1', 'n_dwellings_source'] == 'nsi'


def test_a_secondary_building_other_than_the_largest_has_no_dwellings():
    out = _decide(
        _buildings(
            {
                'house': ('lot1', 160.0, 'Single-Family', None, np.nan, 1.0, 'nsi'),
                'shed': ('lot1', 20.0, 'Secondary', None, np.nan, 1.0, 'nsi'),
                'barn': ('lot2', 400.0, 'Secondary', None, np.nan, 2.0, 'nsi'),
                'home': ('lot2', 120.0, 'Single-Family', None, np.nan, 1.0, 'overture'),
                'alone': (None, 15.0, 'Secondary', None, np.nan, 1.0, 'nsi'),
            }
        )
    )
    assert out.loc['shed', 'n_dwellings'] == 0
    assert _imputed(out.loc['shed', 'n_dwellings_source'])
    # The largest building keeps today's count, Secondary or not: which
    # building is the house there is open (question 5).
    assert out.loc['barn', 'n_dwellings'] == 2
    assert out.loc['barn', 'n_dwellings_source'] == 'nsi'
    # No lot, nothing to compare with.
    assert out.loc['alone', 'n_dwellings'] == 1
    assert out.loc['house', 'n_dwellings'] == 1


GUARD = [
    'Vacant',
    'Mostly Vacant',
    'Retail',
    'Commercial',
    'Office',
    'Industrial',
    'Institutional',
    'Recreation',
]


@pytest.mark.parametrize(
    ('lot_class', 'guarded'),
    [
        ('Retail', True),
        ('Vacant', True),
        ('Single-Family', False),
        ('Agricultural', False),
        (None, False),
    ],
    ids=['retail', 'vacant', 'residential', 'agricultural', 'no class'],
)
def test_the_largest_secondary_building_on_a_guarded_lot(lot_class, guarded):
    out = _decide(
        _buildings(
            {
                'big': ('lot1', 400.0, 'Secondary', lot_class, np.nan, 2.0, 'nsi'),
                'small': ('lot1', 30.0, 'Secondary', lot_class, np.nan, 1.0, 'nsi'),
            }
        ),
        guard_classes=GUARD,
    )
    # It stays Secondary in every case: the step decides the count only.
    assert out.loc['big', 'occupancy_type'] == 'Secondary'
    assert out.loc['small', 'n_dwellings'] == 0
    if guarded:
        assert out.loc['big', 'n_dwellings'] == 0
        assert out.loc['big', 'n_dwellings_source'] == (
            'secondary_nonresidential_lot+imputed'
        )
        assert _imputed(out.loc['big', 'n_dwellings_source'])
    else:
        assert out.loc['big', 'n_dwellings'] == 2
        assert out.loc['big', 'n_dwellings_source'] == 'nsi'


def test_a_guarded_lot_leaves_its_other_classes_alone():
    out = _decide(
        _buildings(
            {
                'shop': ('lot1', 400.0, 'Retail', 'Retail', np.nan, 3.0, 'overture'),
                'flat': ('lot1', 90.0, 'Single-Family', 'Retail', np.nan, 1.0, 'nsi'),
            }
        ),
        guard_classes=GUARD,
    )
    assert out.loc['shop', 'n_dwellings'] == 3
    assert out.loc['flat', 'n_dwellings'] == 1


def test_the_national_recipe_names_the_guard():
    step = next(e for e in _pipeline() if e.get('step') == 'decide_dwellings')
    assert step['guard_classes'] == GUARD


def test_a_single_family_house_with_a_roll_share_of_two_counts_one():
    out = _decide(
        _buildings(
            {
                'share': ('lot1', 150.0, 'Single-Family', None, 5.0, 2.4, 'parcel'),
                'low': ('lot2', 150.0, 'Single-Family', None, 1.0, 1.4, 'parcel'),
                'points': (
                    'lot3',
                    150.0,
                    'Single-Family',
                    None,
                    np.nan,
                    3.0,
                    'overture',
                ),
                'flats': ('lot4', 400.0, 'Multi-Family', None, 6.0, 6.0, 'parcel'),
            }
        )
    )
    assert out.loc['share', 'n_dwellings'] == 1
    assert _imputed(out.loc['share', 'n_dwellings_source'])
    # The roll's share stays where it was, as evidence.
    assert out.loc['share', 'n_dwellings_preliminary'] == 2.4
    # A share that ships as 1, a count from address points (not ruled)
    # and a multi-family building keep their counts.
    assert out.loc['low', 'n_dwellings'] == 1.4
    assert out.loc['points', 'n_dwellings'] == 3
    assert out.loc['flats', 'n_dwellings'] == 6
    assert not _imputed(out.loc['flats', 'n_dwellings_source'])


def test_rule_one_is_off_without_listed_classes():
    out = _decide(
        _buildings({'b1': ('lot1', 300.0, 'Retail', 'Retail', 0.0, 2.0, 'overture')}),
        stated_zero_classes=[],
    )
    assert out.loc['b1', 'n_dwellings'] == 2


def test_the_step_decides_only_what_it_declares():
    frame = _buildings(
        {'b1': ('lot1', 300.0, 'Retail', 'Retail', 0.0, 2.0, 'overture')}
    )
    params = {'stated_zero_classes': STATED_ZERO}
    resolved = resolve_step_columns(decide_dwellings, params)
    assert resolved.decides == {'n_dwellings'}
    out = decide_dwellings(_state(frame.copy()), **params).curated
    assert undeclared_new_columns(decide_dwellings, params, frame.columns, out) == []


def _pipeline():
    return get_recipe_by_id(BUILDING_RECIPE)['pipeline']


def test_the_national_recipe_leaves_the_stated_zero_rule_off():
    # Switched on per unit in a patch recipe once measured there: a
    # curated lot's 0 is not always the roll's statement.
    step = next(e for e in _pipeline() if e.get('step') == 'decide_dwellings')
    assert step['stated_zero_classes'] == []


def test_the_occupancy_votes_read_no_dwelling_count_column():
    # The class is decided before the count; no vote may read the count
    # this branch moved, or the q1b vote would change.
    moved = {'n_dwellings', 'n_dwellings_preliminary', 'n_dwellings_parcel_whole'}
    for entry in _pipeline():
        if entry.get('step') in ('resolve_by_vote', 'derive_indicators'):
            read = set(indicator_columns(entry.get('decisions')))
            read |= set(indicator_columns(entry.get('indicators')))
            assert not (read & moved), entry.get('target')


def _evidence_fixture():
    """Buildings at the step before the count's lanes: roll share,
    Overture and NSI evidence, the lot class and a lot count."""
    return pd.DataFrame(
        {
            'n_dwellings_parcel': [0.0, 0.4, 3.0, np.nan, 1.0, 0.0],
            'n_dwellings_overture': [2.0, np.nan, 1.0, np.nan, 1.0, 1.0],
            'n_dwellings_nsi': [1.0, 1.0, np.nan, 2.0, np.nan, np.nan],
            'land_use_class_parcel': [
                'Retail',
                'Single-Family',
                'Multi-Family',
                None,
                'Vacant',
                'Agricultural',
            ],
            'occupancy_type_building_nsi': ['RES1'] * 6,
            'structure_value_building_nsi': [1.0e5] * 6,
            'structure_value_full_building_nsi': [2.0e5] * 6,
        },
        index=pd.Index([f'b{i}' for i in range(6)], name='building_id'),
    )


def _run_lanes(pipeline, frame):
    """The recipe's steps from the roll-share gate to the Overture zero
    fill, the inputs every later vote reads."""
    _load_steps()
    state = _state(frame.copy(), get_recipe_by_id(BUILDING_RECIPE))
    names = ['suppress_where', 'reconcile_values', 'resolve_by_vote']
    started = False
    for entry in pipeline:
        name = entry.get('step')
        if name == 'suppress_where' and entry.get('column') == 'n_dwellings_overture':
            started = True
        if not started:
            continue
        params = {
            k: v
            for k, v in entry.items()
            if k not in ('step', 'enabled', 'phase_override', 'checkpoint')
        }
        if name in names or name == 'fill_missing_numeric':
            state = _STEP_REGISTRY[name](state, **params)
        if name == 'fill_missing_numeric':
            break
    return state.curated


def test_the_vote_inputs_are_the_same_as_before_the_move():
    # Today's lane step wrote n_dwellings; this branch writes
    # n_dwellings_preliminary. Every column a vote reads comes out the
    # same on a fixture either way.
    now = _pipeline()
    before = copy.deepcopy(now)
    for entry in before:
        priority = entry.get('priority') or {}
        if entry.get('step') == 'reconcile_values' and (
            'n_dwellings_preliminary' in priority
        ):
            entry['priority'] = {
                ('n_dwellings' if k == 'n_dwellings_preliminary' else k): v
                for k, v in priority.items()
            }
    out_now = _run_lanes(now, _evidence_fixture())
    out_before = _run_lanes(before, _evidence_fixture())
    read = set()
    for entry in now:
        if entry.get('step') == 'resolve_by_vote':
            read |= set(indicator_columns(entry.get('decisions')))
    shared = sorted(c for c in read if c in out_now.columns)
    assert 'n_dwellings_overture' in shared
    pd.testing.assert_frame_equal(out_now[shared], out_before[shared])
    # The moved count itself is the same number under its new name.
    pd.testing.assert_series_equal(
        out_now['n_dwellings_preliminary'],
        out_before['n_dwellings'],
        check_names=False,
    )
