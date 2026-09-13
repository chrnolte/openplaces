"""Choosing a level's units out of a Wikidata harvest, without network."""

import pandas as pd

from openplaces.io import admin_wikidata as wd

COUNTY = 'Q269218'
PROVINCE = 'Q70252'
HILL = 'Q54050'
CITY = 'Q515'
ADMIN = {COUNTY, PROVINCE, CITY}


def _row(item, label, classes, direct=None):
    row = {
        'item': f'http://www.wikidata.org/entity/{item}',
        'itemLabel': label,
        'native': '',
        'iso': '',
        'classes': '|'.join(f'http://www.wikidata.org/entity/{c}' for c in classes),
    }
    if direct is not None:
        row['direct'] = direct
    return row


def test_non_administrative_children_are_dropped():
    harvest = pd.DataFrame(
        [_row('Q1', 'Alpha', [COUNTY]), _row('Q2', 'A hill', [HILL])]
    )
    kept, dominant = wd.select_units(harvest, ADMIN)
    assert list(kept['itemLabel']) == ['Alpha']
    assert dominant == COUNTY


def test_the_dominant_class_is_chosen_among_direct_children():
    # Provinces outnumber counties overall, but only counties are direct
    # children of the country: the level is counties, and the county
    # found through its ISO code alone is kept with them.
    harvest = pd.DataFrame(
        [
            _row('Q1', 'County A', [COUNTY], 'true'),
            _row('Q2', 'County B', [COUNTY], 'true'),
            _row('Q3', 'County C', [COUNTY], 'false'),
            _row('Q4', 'Province 1', [PROVINCE], 'false'),
            _row('Q5', 'Province 2', [PROVINCE], 'false'),
            _row('Q6', 'Province 3', [PROVINCE], 'false'),
        ]
    )
    kept, dominant = wd.select_units(harvest, ADMIN)
    assert dominant == COUNTY
    assert sorted(kept['itemLabel']) == ['County A', 'County B', 'County C']


def test_a_sidecar_class_stands_beside_the_dominant_one():
    harvest = pd.DataFrame(
        [
            _row('Q1', 'Community', [COUNTY], 'true'),
            _row('Q4', 'Community B', [COUNTY], 'true'),
            _row('Q2', 'Autonomous city', [CITY], 'true'),
            _row('Q3', 'Other city', [CITY], 'false'),
        ]
    )
    kept, _ = wd.select_units(harvest, ADMIN, keep_classes=[CITY])
    assert sorted(kept['itemLabel']) == [
        'Autonomous city',
        'Community',
        'Community B',
        'Other city',
    ]
    assert set(kept['unit_class']) == {COUNTY, CITY}


def test_an_empty_candidate_set_returns_no_class():
    harvest = pd.DataFrame([_row('Q2', 'A hill', [HILL])])
    kept, dominant = wd.select_units(harvest, ADMIN)
    assert kept.empty and dominant is None


def test_queries_name_the_properties_they_rely_on():
    assert 'wdt:P297 "KE"' in wd.country_children_query('KE')
    assert 'wdt:P300' in wd.country_children_query('KE')
    assert 'wdt:P576' in wd.parent_children_query(['Q1'])
    assert 'VALUES ?parent { wd:Q1 wd:Q2 }' in wd.parent_children_query(['Q1', 'Q2'])
    assert wd.qid('http://www.wikidata.org/entity/Q42') == 'Q42'


def test_the_type_word_is_stripped_where_safe():
    names = pd.Series(['Busia County', 'Mombasa County', 'North', 'North County'])
    types = pd.Series(['county of Kenya'] * 4)
    assert wd.strip_type_word(names, types).tolist() == [
        'Busia',
        'Mombasa',
        'North',
        'North County',
    ]


def test_a_leading_type_word_is_stripped_too():
    names = pd.Series(['Province of Cartago', 'District Alpha'])
    types = pd.Series(['province of Costa Rica', 'district of Elsewhere'])
    assert wd.strip_type_word(names, types).tolist() == ['Cartago', 'Alpha']


def test_type_noun_reads_both_label_shapes():
    assert wd.type_noun('county of Kenya') == 'county'
    assert wd.type_noun('Thai subdistrict') == 'subdistrict'
    assert wd.type_noun('') == ''


def _coded(item, label, cls, iso, parents=(), direct='true', grandparents=()):
    row = _row(item, label, [cls], direct)
    row['iso'] = iso
    row['parents'] = '|'.join(f'http://www.wikidata.org/entity/{p}' for p in parents)
    row['grandparents'] = '|'.join(
        f'http://www.wikidata.org/entity/{p}' for p in grandparents
    )
    return row


LOCALITY = 'Q3257686'
CONSTITUENCY = 'Q192611'


def test_an_electoral_class_loses_a_tie_for_the_label():
    # Every district is also an electoral unit: the district class
    # names the level, so the type reads "district".
    harvest = pd.DataFrame(
        [_row(f'Q{i}', f'District {i}', [CONSTITUENCY, COUNTY]) for i in range(5)]
    )
    kept, dominant = wd.select_units(
        harvest, ADMIN | {CONSTITUENCY}, electoral_classes={CONSTITUENCY}
    )
    assert dominant == COUNTY
    assert set(kept['unit_class']) == {COUNTY}


def test_an_item_with_its_own_country_code_is_not_a_subdivision():
    rows = [_coded(f'Q{i}', f'Region {i}', PROVINCE, f'XX-R{i}') for i in range(3)]
    overseas = _coded('Q9', 'Overseas', PROVINCE, 'XX-OV')
    overseas['country_code'] = 'OV'
    rows.append(overseas)
    kept, _ = wd.select_units(pd.DataFrame(rows), ADMIN)
    assert sorted(kept['itemLabel']) == ['Region 0', 'Region 1', 'Region 2']


def test_a_single_coded_stray_does_not_decide_the_level():
    rows = [_row(f'Q{i}', f'Municipality {i}', [COUNTY], 'true') for i in range(6)]
    rows.append(_coded('Q9', 'A park', PROVINCE, 'XX-P'))
    kept, dominant = wd.select_units(pd.DataFrame(rows), ADMIN)
    assert dominant == COUNTY and len(kept) == 6


def test_a_coded_settlement_class_is_kept_beside_the_provinces():
    rows = [_coded(f'Q{i}', f'Province {i}', PROVINCE, f'XX-{i}') for i in range(9)]
    rows += [_coded(f'Q{i}', f'Metro {i}', CITY, f'XX-{i}') for i in range(20, 26)]
    kept, dominant = wd.select_units(
        pd.DataFrame(rows), ADMIN, settlement_classes={CITY}
    )
    assert dominant == PROVINCE and len(kept) == 15


def test_a_class_holding_the_dominant_parents_is_the_level_above():
    # Former provinces still carry codes; twelve counties are located
    # in them, the rest in the country.
    rows = [_coded(f'Q{i}', f'Province {i}', PROVINCE, f'XX-P{i}') for i in range(8)]
    rows += [
        _coded(
            f'Q{i}', f'County {i}', COUNTY, f'XX-{i}', [f'Q{i % 8}'] if i < 22 else []
        )
        for i in range(10, 57)
    ]
    kept, dominant = wd.select_units(pd.DataFrame(rows), ADMIN)
    assert dominant == COUNTY and len(kept) == 47


def test_settlements_lose_to_a_territorial_class_however_many():
    harvest = pd.DataFrame(
        [_row(f'Q{i}', f'Village {i}', [LOCALITY], 'true') for i in range(40)]
        + [
            _row('Q100', 'State A', [PROVINCE], 'true'),
            _row('Q101', 'State B', [PROVINCE], 'true'),
        ]
    )
    kept, dominant = wd.select_units(
        harvest, ADMIN | {LOCALITY}, settlement_classes={LOCALITY}
    )
    assert dominant == PROVINCE
    assert len(kept) == 2


def test_settlements_are_the_level_when_nothing_else_is_on_offer():
    harvest = pd.DataFrame([_row('Q1', 'Town A', [CITY]), _row('Q2', 'Town B', [CITY])])
    kept, dominant = wd.select_units(harvest, ADMIN, settlement_classes={CITY})
    assert dominant == CITY and len(kept) == 2


def test_iso_coded_candidates_decide_level_two():
    # Districts carry the country's ISO codes; the four regions above
    # them are direct children of the country but carry none.
    rows = [_row(f'Q{i}', f'Region {i}', [PROVINCE], 'true') for i in range(4)]
    for i in range(10, 22):
        row = _row(f'Q{i}', f'District {i}', [COUNTY], 'false')
        row['iso'] = f'XX-{i}'
        rows.append(row)
    kept, dominant = wd.select_units(pd.DataFrame(rows), ADMIN)
    assert dominant == COUNTY
    assert len(kept) == 12


def test_a_coded_class_nested_in_another_is_a_lower_level():
    # Departments carry codes but sit inside the regions; the regions
    # are located in a "metropolitan" item outside the pool.
    rows = [
        _coded(f'Q{i}', f'Region {i}', PROVINCE, f'XX-R{i}', ['Q999'], 'false')
        for i in range(3)
    ]
    rows += [
        _coded(f'Q{i}', f'Department {i}', COUNTY, f'XX-{i}', [f'Q{i % 3}'])
        for i in range(10, 25)
    ]
    kept, dominant = wd.select_units(pd.DataFrame(rows), ADMIN)
    assert dominant == PROVINCE
    assert len(kept) == 3


def test_every_unnested_coded_class_is_kept_at_level_two():
    rows = [_coded(f'Q{i}', f'Province {i}', PROVINCE, f'XX-{i}') for i in range(9)]
    rows += [_coded(f'Q{i}', f'Metro {i}', CITY, f'XX-{i}') for i in range(20, 26)]
    kept, dominant = wd.select_units(pd.DataFrame(rows), ADMIN)
    assert dominant == PROVINCE
    assert len(kept) == 15


def test_lower_levels_keep_the_largest_class_only():
    # No `direct` column: a level-3 harvest under explicit parents.
    rows = [_row(f'Q{i}', f'Sub {i}', [COUNTY]) for i in range(5)]
    rows += [_row(f'Q{i}', f'Other {i}', [PROVINCE]) for i in range(10, 13)]
    kept, dominant = wd.select_units(pd.DataFrame(rows), ADMIN)
    assert dominant == COUNTY and len(kept) == 5


def test_a_class_nested_two_hops_down_is_a_lower_level():
    # Paris is located in Grand Paris, which is located in its region.
    rows = [_coded(f'Q{i}', f'Region {i}', PROVINCE, f'XX-R{i}') for i in range(3)]
    rows.append(_coded('Q9', 'Capital', CITY, 'XX-CAP', ['Q99'], grandparents=['Q0']))
    kept, dominant = wd.select_units(pd.DataFrame(rows), ADMIN)
    assert dominant == PROVINCE and len(kept) == 3


def test_a_secondary_class_keeps_only_its_coded_items():
    # A historical region shares a class with a coded autonomous city.
    rows = [_coded(f'Q{i}', f'Community {i}', PROVINCE, f'XX-{i}') for i in range(5)]
    rows.append(_coded('Q8', 'Autonomous city', CITY, 'XX-CE'))
    rows.append(_coded('Q9', 'Old region', CITY, ''))
    kept, _ = wd.select_units(pd.DataFrame(rows), ADMIN)
    assert 'Old region' not in set(kept['itemLabel'])
    assert 'Autonomous city' in set(kept['itemLabel'])


def test_an_item_holding_the_dominant_units_is_dropped_whatever_its_class():
    # A former province of a generic class, coded, holds two counties.
    rows = [_coded(f'Q{i}', f'County {i}', COUNTY, f'XX-{i}') for i in range(10, 16)]
    rows += [
        _coded(f'Q{i}', f'County {i}', COUNTY, f'XX-{i}', ['Q1']) for i in (16, 17)
    ]
    rows.append(_coded('Q1', 'Coast', HILL, 'XX-300'))
    kept, dominant = wd.select_units(pd.DataFrame(rows), ADMIN | {HILL})
    assert dominant == COUNTY and len(kept) == 8


def test_a_dominant_unit_holding_another_is_not_the_level_above():
    # Two federal subjects are located in a third, which is one too.
    rows = [_coded(f'Q{i}', f'Oblast {i}', PROVINCE, f'XX-{i}') for i in range(5)]
    rows += [_coded(f'Q{i}', f'Okrug {i}', PROVINCE, f'XX-{i}', ['Q0']) for i in (8, 9)]
    kept, _ = wd.select_units(pd.DataFrame(rows), ADMIN)
    assert len(kept) == 7


def test_a_class_with_one_item_above_the_units_is_dropped_whole():
    rows = [_coded(f'Q{i}', f'Province {i}', PROVINCE, f'XX-P{i}') for i in range(3)]
    rows += [
        _coded(f'Q{i}', f'County {i}', COUNTY, f'XX-{i}', ['Q0']) for i in (10, 11)
    ]
    rows += [_coded(f'Q{i}', f'County {i}', COUNTY, f'XX-{i}') for i in range(12, 20)]
    kept, dominant = wd.select_units(pd.DataFrame(rows), ADMIN)
    assert dominant == COUNTY and len(kept) == 10


def test_a_unit_outside_the_dominant_class_takes_its_rarest_class():
    # Every county is also a "first-level division"; so is the capital,
    # which is a municipality besides: that is its label.
    generic = 'Q10864048'
    rows = [_coded(f'Q{i}', f'County {i}', COUNTY, f'XX-{i}') for i in range(6)]
    for row in rows:
        row['classes'] += f'|http://www.wikidata.org/entity/{generic}'
    capital = _coded('Q9', 'Capital', CITY, 'XX-B')
    capital['classes'] += f'|http://www.wikidata.org/entity/{generic}'
    rows.append(capital)
    kept, _ = wd.select_units(pd.DataFrame(rows), ADMIN | {generic})
    assert kept.set_index('itemLabel')['unit_class']['Capital'] == CITY
    assert kept.set_index('itemLabel')['unit_class']['County 0'] == COUNTY


def test_a_class_named_for_the_country_labels_a_unit_outside_the_level():
    rows = [_coded(f'Q{i}', f'Province {i}', PROVINCE, f'XX-{i}') for i in range(6)]
    capital = _coded('Q9', 'Capital', CITY, 'XX-CAP')
    capital['classes'] += '|http://www.wikidata.org/entity/Q7574799'
    rows.append(capital)
    kept, _ = wd.select_units(
        pd.DataFrame(rows), ADMIN | {'Q7574799'}, country_classes={'Q7574799'}
    )
    assert kept.set_index('itemLabel')['unit_class']['Capital'] == 'Q7574799'


def test_other_members_of_a_class_above_the_units_go_with_it():
    # Two provinces hold counties; a third holds none but is also a
    # bare administrative entity, and leaves with the class.
    rows = [_coded(f'Q{i}', f'Province {i}', PROVINCE, f'XX-P{i}') for i in range(3)]
    rows[2]['classes'] += '|http://www.wikidata.org/entity/Q56061'
    rows += [
        _coded(f'Q{i}', f'County {i}', COUNTY, f'XX-{i}', [f'Q{i % 2}'])
        for i in (10, 11)
    ]
    rows += [_coded(f'Q{i}', f'County {i}', COUNTY, f'XX-{i}') for i in range(12, 20)]
    kept, dominant = wd.select_units(pd.DataFrame(rows), ADMIN | {'Q56061'})
    assert dominant == COUNTY and len(kept) == 10


def test_a_single_uncoded_child_of_a_country_is_no_level():
    # Whatever else the country holds: Aruba's one bare entity stands
    # beside dozens of settlements.
    rows = [_row('Q1', 'Only', [COUNTY], 'true')]
    rows += [_row(f'Q{i}', f'Town {i}', [CITY], 'true') for i in range(10, 14)]
    kept, dominant = wd.select_units(
        pd.DataFrame(rows), ADMIN, settlement_classes={CITY}
    )
    assert kept.empty and dominant is None


def test_a_single_child_of_a_lower_level_parent_is_still_a_unit():
    kept, dominant = wd.select_units(
        pd.DataFrame([_row('Q1', 'Only', [COUNTY])]), ADMIN
    )
    assert dominant == COUNTY and len(kept) == 1
