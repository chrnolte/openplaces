"""The temporal join runs nationwide, so it must say what it did.

Every value here is fabricated.
"""

import pandas as pd
import pytest

from openplaces.io.curator import transactions as tx


def _state(curated):
    state = tx.CurateState.__new__(tx.CurateState)
    state.curated = curated
    state.admin_id = 'XX-YY-ZZ'
    state.verbose = False
    return state


def _sales():
    return pd.DataFrame(
        {
            'parcel_id_assessor': ['a1', 'a2', 'a3'],
            'sale_year': [2011, 2015, 2019],
            'sale_vacant': ['Improved property'] * 3,
        }
    )


PANEL = pd.DataFrame(
    {
        'parcel_id_assessor': ['a1', 'a1', 'a2'],
        'tax_year': [2011, 2019, 2019],
        'land_value': [10, 30, 20],
    }
)


def _run(monkeypatch, ref, **kwargs):
    if isinstance(ref, Exception):

        def loader(*a, **k):
            raise ref
    else:

        def loader(*a, **k):
            return ref.copy()

    monkeypatch.setattr(tx, 'get_entities', loader)
    opts = dict(
        recipe_id='roll',
        join_key='parcel_id_assessor',
        date_column='sale_year',
        vintage_column='tax_year',
        columns=['tax_year', 'land_value'],
        match_type_column='property_match_type',
        require_panel=True,
    )
    opts.update(kwargs)
    return tx.join_temporal_snapshot(_state(_sales()), **opts).curated


def test_a_county_with_no_roll_says_so_instead_of_raising(monkeypatch):
    out = _run(monkeypatch, FileNotFoundError('no file'))
    assert out['property_match_type'].tolist() == ['no_panel'] * 3
    assert 'land_value' not in out.columns


def test_a_single_vintage_roll_is_not_a_panel(monkeypatch):
    one = PANEL[PANEL['tax_year'] == 2019]
    out = _run(monkeypatch, one)
    assert out['property_match_type'].tolist() == ['single_vintage'] * 3


def test_an_exact_year_and_a_fallback_year_are_told_apart(monkeypatch):
    out = _run(monkeypatch, PANEL, direction='backward')
    # a1 sold 2011 and the panel holds 2011: exact.
    assert out['property_match_type'][0] == 'exact'
    assert out['land_value'][0] == 10
    # a2 sold 2015, panel has only 2019: no earlier year to take.
    assert pd.isna(out['property_match_type'][1])


def test_the_last_pass_closes_the_column_so_blank_never_means_two_things(
    monkeypatch,
):
    out = _run(monkeypatch, PANEL, direction='backward', mark_unmatched='not_in_panel')
    assert out['property_match_type'].notna().all()
    assert set(out['property_match_type']) <= {
        'exact',
        'backward_fallback',
        'not_in_panel',
    }


def test_a_restricted_last_pass_closes_the_whole_column(monkeypatch):
    # The real last pass is restricted (a bare lot must not inherit a
    # later vintage's house). Marking only its own active rows left the
    # rows it declined to touch blank: 0.4% of Volusia.
    sales = _sales()
    sales.loc[2, 'sale_vacant'] = 'Vacant'
    monkeypatch.setattr(tx, 'get_entities', lambda *a, **k: PANEL.copy())
    out = tx.join_temporal_snapshot(
        _state(sales),
        recipe_id='roll',
        join_key='parcel_id_assessor',
        date_column='sale_year',
        vintage_column='tax_year',
        columns=['tax_year', 'land_value'],
        direction='forward',
        restrict_to={'column': 'sale_vacant', 'equals': 'Improved property'},
        match_type_column='property_match_type',
        require_panel=True,
        mark_unmatched='not_in_panel',
    ).curated
    assert out['property_match_type'].notna().all()
    # The vacant sale the pass declined to touch is closed too, and the
    # rows it did match keep the label for the direction it used.
    assert set(out['property_match_type']) <= {
        'forward_fallback',
        'exact',
        'not_in_panel',
    }
    assert out['property_match_type'][2] == 'not_in_panel'


def test_an_imperfect_sale_year_does_not_break_the_merge(monkeypatch):
    # A sale year with any gap parses to float64 while a complete
    # tax_year parses to int64, and merge_asof refuses keys of two
    # types: the join failed on exactly the counties whose sale dates
    # are imperfect, which is most of them.
    sales = _sales()
    sales.loc[1, 'sale_year'] = None
    monkeypatch.setattr(tx, 'get_entities', lambda *a, **k: PANEL.copy())
    out = tx.join_temporal_snapshot(
        _state(sales),
        recipe_id='roll',
        join_key='parcel_id_assessor',
        date_column='sale_year',
        vintage_column='tax_year',
        columns=['tax_year', 'land_value'],
        match_type_column='property_match_type',
        require_panel=True,
    ).curated
    assert out['property_match_type'][0] == 'exact'
    # The undated sale is named for what is missing, not blamed on the
    # panel.
    assert out['property_match_type'][1] == 'no_sale_date'


def test_a_restriction_on_an_absent_column_selects_nothing(monkeypatch):
    # `sale_vacant` is Florida's word; the national pipeline runs this
    # step over every state, and Wisconsin has neither the column nor a
    # panel. Raising there would fail the whole curate.
    monkeypatch.setattr(tx, 'get_entities', lambda *a, **k: PANEL.copy())
    out = _run(monkeypatch, PANEL, restrict_to={'column': 'not_here', 'equals': 'x'})
    assert 'land_value' not in out.columns


def test_a_column_this_roll_lacks_is_skipped_not_fatal(monkeypatch):
    # One nationwide recipe names one column list; Lake County FL has no
    # land_area_sqft and asking for it failed the whole curate.
    out = _run(monkeypatch, PANEL, columns=['tax_year', 'land_value', 'land_area_sqft'])
    assert out['land_value'][0] == 10
    assert 'land_area_sqft' not in out.columns


def test_without_require_panel_a_missing_roll_still_raises(monkeypatch):
    # A state-specific recipe naming its own roll should fail loudly if
    # that roll is absent; only a nationwide pipeline opts out.
    with pytest.raises(FileNotFoundError):
        _run(monkeypatch, FileNotFoundError('no file'), require_panel=False)
