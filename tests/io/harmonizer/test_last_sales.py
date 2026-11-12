"""Assessor last-sale lane: roll fields reshaped into transaction rows.

Every value here is fabricated.
"""

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import last_sales as ls


def test_parse_sale_dates_reads_every_encoding_found_on_disk():
    expected = pd.Timestamp('2012-08-17')
    ms = pd.Series([1345161600000.0, 0.0, None])
    assert ls.parse_sale_dates(ms).tolist()[0] == expected
    assert ls.parse_sale_dates(ms).isna().tolist() == [False, True, True]

    calendar = pd.Series([20120817, 0], dtype='int32')
    assert ls.parse_sale_dates(calendar).tolist()[0] == expected
    assert pd.isna(ls.parse_sale_dates(calendar)[1])

    text = pd.Series(['2012/08/17', '2012-08-17', 'not a date', None])
    parsed = ls.parse_sale_dates(text)
    assert parsed.tolist()[:2] == [expected, expected]
    assert parsed.isna().tolist()[2:] == [True, True]

    aware = pd.Series(pd.to_datetime(['2012-08-17'], utc=True))
    parsed = ls.parse_sale_dates(aware)
    assert parsed.dt.tz is None
    assert parsed[0] == expected


def test_row_needs_a_price_and_a_real_date():
    table = pd.DataFrame(
        {
            'parcel_id_local': ['a', 'b', 'c', 'd'],
            'last_sale_price': [250000.0, 0.0, None, 180000.0],
            'last_sale_date': ['2019-05-01', '2020-01-15', '2021-03-01', None],
        }
    )
    out = ls.last_sales_to_transactions(table, 'fabricated')
    # 'b' is a nominal transfer and stays; 'c' has no price, 'd' no date.
    assert out['parcel_id_local'].tolist() == ['a', 'b']
    assert out['price'].tolist() == [250000.0, 0.0]
    assert out['sale_year'].tolist() == [2019.0, 2020.0]
    assert out['sale_month'].tolist() == [5.0, 1.0]
    assert (out['sale_record_kind'] == 'assessor_last_sale').all()
    assert (out['source'] == 'fabricated').all()


def test_zero_year_placeholder_is_not_a_sale():
    # The Florida pattern: every column filled, 0 meaning "no sale".
    table = pd.DataFrame(
        {
            'parcel_id_local': ['a', 'b'],
            'last_sale_price': [0.0, 310000.0],
            'last_sale_year': [0.0, 2023.0],
            'last_sale_month': ['0', '7'],
            'last_sale_qualification_code': ['x', 'Qualified'],
            'last_sale_vacant': ['I', 'I'],
        }
    )
    out = ls.last_sales_to_transactions(table, 'fabricated')
    assert out['parcel_id_local'].tolist() == ['b']
    assert out['sale_year'].tolist() == [2023.0]
    assert out['sale_month'].tolist() == [7.0]
    assert out['recorded_date'].isna().all()
    assert out['sale_qualification_code'].tolist() == ['Qualified']
    assert out['sale_vacant'].tolist() == ['I']


def test_month_only_roll_does_not_invent_a_day():
    months = pd.date_range('2015-01-01', periods=40, freq='MS')
    table = pd.DataFrame(
        {'last_sale_price': 1.0, 'last_sale_date': months.strftime('%Y-%m-%d')}
    )
    out = ls.last_sales_to_transactions(table, 'x')
    assert len(out) == 40
    assert out['recorded_date'].isna().all()
    assert out['sale_year'].tolist()[:2] == [2015.0, 2015.0]
    assert out['sale_month'].tolist()[:2] == [1.0, 2.0]

    # One real day is enough to read the source as daily.
    table.loc[0, 'last_sale_date'] = '2015-01-17'
    out = ls.last_sales_to_transactions(table, 'x')
    assert out['recorded_date'].notna().all()


def test_table_without_price_or_without_date_yields_nothing():
    dated = pd.DataFrame({'last_sale_date': ['2019-05-01']})
    priced = pd.DataFrame({'last_sale_price': [1.0]})
    assert ls.last_sales_to_transactions(dated, 'x').empty
    assert ls.last_sales_to_transactions(priced, 'x').empty


def test_book_and_page_split_only_when_unambiguous():
    table = pd.DataFrame(
        {
            'last_sale_price': [1.0, 1.0, 1.0, 1.0],
            'last_sale_date': ['2019-05-01'] * 4,
            'last_sale_book_page': ['0123/0456', '77 88', '1-2-3', None],
        }
    )
    out = ls.last_sales_to_transactions(table, 'x')
    assert out['sale_book'].tolist()[:2] == ['0123', '77']
    assert out['sale_page'].tolist()[:2] == ['0456', '88']
    assert out['sale_book'].isna().tolist()[2:] == [True, True]


def test_separate_book_and_page_columns_win_over_the_combined_one():
    table = pd.DataFrame(
        {
            'last_sale_price': [1.0],
            'last_sale_date': ['2019-05-01'],
            'last_sale_book': ['10'],
            'last_sale_page': ['20'],
            'last_sale_book_page': ['99/99'],
        }
    )
    out = ls.last_sales_to_transactions(table, 'x')
    assert out['sale_book'].tolist() == ['10']
    assert out['sale_page'].tolist() == ['20']


def test_blank_book_and_page_are_missing_not_shared_text():
    table = pd.DataFrame(
        {
            'last_sale_price': [1.0, 1.0],
            'last_sale_date': ['2019-05-01', '2019-05-02'],
            'last_sale_book': [' ', '10'],
            'last_sale_page': ['', '20'],
        }
    )
    out = ls.last_sales_to_transactions(table, 'x')
    assert out['sale_book'].isna().tolist() == [True, False]
    assert out['sale_page'].isna().tolist() == [True, False]


def test_multi_year_roll_states_each_last_sale_once():
    # Three roll years: the first two repeat one sale, the third shows
    # the property sold again. A use code drifting between years does
    # not make a new sale.
    table = pd.DataFrame(
        {
            'parcel_id_local': ['p1', 'p1', 'p1', 'p2'],
            'last_sale_price': [200000.0, 200000.0, 260000.0, 90000.0],
            'last_sale_year': [2019.0, 2019.0, 2023.0, 2019.0],
            'last_sale_month': [5.0, 5.0, 8.0, 5.0],
            'use_group_code': ['001', '004', '004', '001'],
        }
    )
    out = ls.last_sales_to_transactions(table, 'x')
    assert sorted(zip(out['parcel_id_local'], out['sale_year'])) == [
        ('p1', 2019.0),
        ('p1', 2023.0),
        ('p2', 2019.0),
    ]


def test_sales_without_any_identifier_are_never_merged():
    table = pd.DataFrame(
        {
            'last_sale_price': [200000.0, 200000.0],
            'last_sale_date': ['2019-05-03', '2019-05-03'],
        }
    )
    assert len(ls.last_sales_to_transactions(table, 'x')) == 2


def test_owner_columns_never_reach_a_transaction_row():
    table = pd.DataFrame(
        {
            'last_sale_price': [1.0],
            'last_sale_date': ['2019-05-01'],
            'owner_name': ['Fabricated Owner'],
            'owner_address': ['1 Fabricated Way'],
            'grantor': ['Fabricated Seller'],
            'grantee': ['Fabricated Buyer'],
        }
    )
    out = ls.last_sales_to_transactions(table, 'x')
    assert not {'owner_name', 'owner_address', 'grantor', 'grantee'} & set(out.columns)


def _run_step(monkeypatch, tables, spine=None, **params):
    """Run append_last_sales over fabricated in-memory tables."""

    def fake_discover(sources, state):
        entity_type = sources[0]['entity_type']
        return [
            {'recipe_id': rid, 'label': rid}
            for rid, (etype, _table) in tables.items()
            if etype == entity_type
        ]

    def fake_get_entities(recipe_id, admin_id, layer=None, missing=None):
        return tables[recipe_id][1]

    monkeypatch.setattr(ls, '_expand_auto_discover', fake_discover)
    monkeypatch.setattr(ls, 'get_entities', fake_get_entities)
    monkeypatch.setattr(ls, 'restrict_to_admin_by_name', lambda df, *_: df)
    state = HarmonizeState.__new__(HarmonizeState)
    state.spine = spine
    state.admin_id = 'US-ZZ-ZZZ'
    state.metadata = {}
    state.timer = None
    state.verbose = False
    return ls.append_last_sales(state, **params).spine


def test_stacked_unit_sale_comes_from_the_property_row(monkeypatch):
    # A lot with two condominium units: the units' sales sit on property
    # rows keyed to the lot, the lot's parcel row repeats one of them,
    # and a second parcel has no property row at all.
    properties = pd.DataFrame(
        {
            'parcel_id_local': ['lot1', 'lot1'],
            'property_id_local': ['lot1-u1', 'lot1-u2'],
            'last_sale_price': [200000.0, 210000.0],
            'last_sale_date': ['2020-02-01', '2021-06-01'],
        }
    )
    parcels = pd.DataFrame(
        {
            'parcel_id_local': ['lot1', 'lot2'],
            'last_sale_price': [210000.0, 150000.0],
            'last_sale_date': ['2021-06-01', '2018-09-01'],
        }
    )
    spine = _run_step(
        monkeypatch,
        {'roll': ('property', properties), 'layer': ('parcel', parcels)},
    )
    assert sorted(spine['price'].tolist()) == [150000.0, 200000.0, 210000.0]
    lot1 = spine[spine['parcel_id_local'] == 'lot1']
    assert sorted(lot1['property_id_local'].tolist()) == ['lot1-u1', 'lot1-u2']
    assert (lot1['source'] == 'roll').all()


def test_deed_rows_are_kept_apart_and_labeled(monkeypatch):
    # The roll's sale of 2012 lies outside the feed's 2018 coverage, so
    # it is added, and each row says which source it came from.
    deeds = pd.DataFrame(
        {
            'parcel_id_local': ['lot2'],
            'price': [150000.0],
            'recorded_date': pd.to_datetime(['2018-09-03']),
            'source': ['recorder'],
        }
    )
    parcels = pd.DataFrame(
        {
            'parcel_id_local': ['lot2'],
            'last_sale_price': [120000.0],
            'last_sale_date': ['2012-04-01'],
        }
    )
    spine = _run_step(monkeypatch, {'layer': ('parcel', parcels)}, spine=deeds)
    assert spine['sale_record_kind'].tolist() == ['deed', 'assessor_last_sale']
    assert spine.index.is_unique


def _deeds(months, source='recorder', partition=None):
    """Fabricated deeds, one per YYYY-MM month given."""
    frame = pd.DataFrame(
        {
            'parcel_id_local': [f'd{i}' for i in range(len(months))],
            'price': 100000.0,
            'sale_year': [float(m[:4]) for m in months],
            'sale_month': [float(m[5:]) for m in months],
            'source': source,
        }
    )
    if partition is not None:
        frame['tax_year'] = partition
    return frame


def _roll(periods):
    """Fabricated roll rows, one last sale per (year, month) given."""
    return pd.DataFrame(
        {
            'parcel_id_local': [f'r{i}' for i in range(len(periods))],
            'last_sale_price': 90000.0,
            'last_sale_year': [float(y) for y, _ in periods],
            'last_sale_month': [m for _, m in periods],
        }
    )


def test_a_sale_comes_from_the_deeds_inside_their_coverage(monkeypatch):
    # Deeds of 2019-01 to 2019-12: a roll's last sale in any month of
    # that span is not added, whether or not a deed of that month exists
    # (2019-06 has none: coverage is the span, not the density); one
    # before or after it is.
    deeds = _deeds(['2019-01', '2019-03', '2019-12'])
    roll = _roll([(2019, 6.0), (2019, 1.0), (2019, 12.0), (2018, 12.0), (2020, 1.0)])
    spine = _run_step(monkeypatch, {'layer': ('parcel', roll)}, spine=deeds)
    last = spine[spine['sale_record_kind'] == 'assessor_last_sale']
    assert sorted(zip(last['sale_year'], last['sale_month'], strict=True)) == [
        (2018.0, 12.0),
        (2020.0, 1.0),
    ]
    assert (spine['sale_record_kind'] == 'deed').sum() == 3


def test_a_unit_with_no_deed_feed_keeps_every_last_sale(monkeypatch):
    roll = _roll([(2019, 6.0), (2005, 2.0)])
    spine = _run_step(monkeypatch, {'layer': ('parcel', roll)})
    assert (spine['sale_record_kind'] == 'assessor_last_sale').sum() == 2


def test_a_missing_file_is_a_gap_the_roll_fills(monkeypatch):
    # Files 2016 and 2018 of a feed listing sales of t - 1 and early t;
    # file 2017 is missing, so 2016-10 to 2016-12 is uncovered. Read as
    # one span the feed would claim those months too.
    deeds = pd.concat(
        [
            _deeds(['2015-01', '2016-09'], partition=2016),
            _deeds(['2017-01', '2018-09'], partition=2018),
        ],
        ignore_index=True,
    )
    roll = _roll([(2016, 11.0), (2016, 9.0), (2017, 1.0)])
    spine = _run_step(
        monkeypatch,
        {'layer': ('parcel', roll)},
        spine=deeds,
        coverage_partition_column='tax_year',
    )
    last = spine[spine['sale_record_kind'] == 'assessor_last_sale']
    assert last['sale_month'].tolist() == [11.0]

    spine = _run_step(monkeypatch, {'layer': ('parcel', roll)}, spine=deeds)
    assert not (spine['sale_record_kind'] == 'assessor_last_sale').any()


def test_a_year_only_last_sale_is_covered_only_by_the_whole_year():
    covered = ls.deed_coverage(_deeds(['2019-01', '2019-12', '2020-01', '2020-06']))
    rows = pd.DataFrame(
        {
            'sale_year': [2019.0, 2020.0, 2020.0, None],
            'sale_month': [None, None, 6.0, None],
        }
    )
    assert ls.within_coverage(rows, covered).tolist() == [True, False, True, False]


def test_coverage_limits_clip_a_feed_documented_as_partial():
    deeds = _deeds(['2003-05', '2010-01', '2025-04'])
    covered = ls.deed_coverage(deeds, limits=[200801, 202412])
    rows = pd.DataFrame(
        {'sale_year': [2004.0, 2008.0, 2024.0, 2025.0], 'sale_month': 1.0}
    )
    assert ls.within_coverage(rows, covered).tolist() == [False, True, True, False]


def test_coverage_reads_a_recorded_date_and_each_feed_apart():
    # A recorder states a date, not a year and month; two feeds of one
    # unit each cover their own span, not the months between them.
    deeds = pd.DataFrame(
        {
            'recorded_date': pd.to_datetime(
                ['2014-01-10', '2014-03-02', '2021-02-01', '2021-04-30']
            ),
            'source': ['older', 'older', 'newer', 'newer'],
        }
    )
    covered = ls.deed_coverage(deeds)
    rows = pd.DataFrame(
        {
            'sale_year': [2014.0, 2014.0, 2016.0, 2021.0],
            'sale_month': [2.0, 4.0, 1.0, 3.0],
        }
    )
    assert ls.within_coverage(rows, covered).tolist() == [True, False, False, True]


def test_the_rule_reads_periods_never_names_or_values(monkeypatch):
    # Owner names and assessed values on the roll, and an index column
    # on the deeds, leave the choice unchanged (design limits L7, L29).
    deeds = _deeds(['2019-01', '2019-12'])
    roll = _roll([(2019, 6.0), (2018, 3.0)])
    base = _run_step(monkeypatch, {'layer': ('parcel', roll)}, spine=deeds.copy())
    roll = roll.assign(
        owner_name='Fabricated Owner', land_value=[1.0, 9e9], improvement_value=5.0
    )
    deeds = deeds.assign(price_index=[0.5, 2.0], grantee='Fabricated Buyer')
    other = _run_step(monkeypatch, {'layer': ('parcel', roll)}, spine=deeds)
    for frame in (base, other):
        last = frame[frame['sale_record_kind'] == 'assessor_last_sale']
        assert last['sale_year'].tolist() == [2018.0]


def test_no_roll_and_no_deeds_leaves_the_state_untouched(monkeypatch):
    assert _run_step(monkeypatch, {}) is None
