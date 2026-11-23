"""One step divides each roll account's values over the lots it is drawn on.

plans/account-values-one-deciding-step.md, packages V3 and V5. Every id,
value and area below is fabricated.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState, account_values
from openplaces.io.curator.provenance import is_imputed as _is_imputed


def is_imputed(token) -> bool:
    return bool(_is_imputed(pd.Series([token])).iloc[0])


COLUMNS = [
    'land_value',
    'improvement_value',
    'other_value',
    'total_value',
    'n_dwellings',
]


def _state(curated):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US', 'XX', 'YY'),
        verbose=False,
        timer=None,
        curated=curated,
    )


def _lots(spec):
    """Curated lots: {lot: (area_ha, {column: value})}; the layer's
    values are what the parcel geospine left on the lot."""
    rows = []
    for lot, (area, values) in spec.items():
        row = {'parcel_id': lot, 'area_ha': area}
        row.update(values)
        rows.append(row)
    lots = pd.DataFrame(rows).set_index('parcel_id')
    for column in COLUMNS:
        if column not in lots.columns:
            lots[column] = np.nan
    return lots


def _accounts(spec):
    frame = pd.DataFrame.from_dict(spec, orient='index')
    frame.index.name = 'property_id'
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame['source'] = 'rollsource'
    return frame


def _links(rows):
    """rows: (account, lot, share, method, source)."""
    return pd.DataFrame(
        [
            {
                'property_id': a,
                'parcel_id': lot,
                'link_method': method,
                'link_source': source,
                'share': share,
                'share_basis': 'area' if share is not None else None,
            }
            for a, lot, share, method, source in rows
        ]
    ).astype({'share': 'Float64'})


@pytest.fixture
def run(monkeypatch):
    def _run(accounts, links, lots, overlaps=None, buildings=None, **params):
        tables = {'US_property-spine-2026': accounts}
        if buildings is not None:
            tables['US_building-spine-2026'] = buildings

        def fake_get_entities(recipe_id, admin_id, columns=None, missing='raise'):
            table = tables[recipe_id]
            return table[[c for c in (columns or table.columns) if c in table.columns]]

        monkeypatch.setattr(account_values, 'get_entities', fake_get_entities)
        monkeypatch.setattr(
            account_values,
            'describe_recipe',
            lambda recipe_id, admin_id: pd.DataFrame(index=tables[recipe_id].columns),
        )
        monkeypatch.setattr(
            account_values,
            'read_curate_link',
            lambda state, recipe_id: (links, Path('link.parquet')),
        )
        monkeypatch.setattr(
            account_values, '_building_overlaps', lambda state, recipe_id: overlaps
        )
        params.setdefault('report', None)
        if buildings is not None:
            params.setdefault('building_recipe_id', 'US_building-spine-2026')
            params.setdefault(
                'stories', {'evidence': [{'column': 'n_stories_building_nsi'}]}
            )
        state = account_values.divide_account_values(
            _state(lots.copy()), 'US_property-spine-2026', **params
        )
        return state.curated

    return _run


def _overlaps(rows):
    return pd.DataFrame(rows, columns=['building', 'lot', 'overlap_m2'])


def _buildings(rows):
    """rows: (building, area_m2, stories, geometry_source)."""
    return pd.DataFrame(
        [
            {
                'building_id': b,
                'area_ha': area / 1e4,
                'n_stories_building_nsi': stories,
                'geometry_source': source,
            }
            for b, area, stories, source in rows
        ]
    ).set_index('building_id')


def test_land_follows_lot_area_renormalized_over_the_curated_lots(run):
    # The link table's shares sum to one over the geospine's lots; the
    # curate dropped the third lot, so the two kept lots divide it all.
    accounts = _accounts({'A': {'land_value': 900.0}})
    links = _links(
        [
            ('A', 'a1', 0.6, 'parcel_id_local', 'roll'),
            ('A', 'a2', 0.3, 'parcel_id_local', 'roll'),
            ('A', 'gone', 0.1, 'parcel_id_local', 'roll'),
        ]
    )
    lots = _lots(
        {'a1': (0.6, {'land_value': 900.0}), 'a2': (0.3, {'land_value': 900.0})}
    )
    out = run(accounts, links, lots)
    assert out.loc['a1', 'land_value'] == pytest.approx(600.0)
    assert out.loc['a2', 'land_value'] == pytest.approx(300.0)
    assert out['land_value'].sum() == pytest.approx(900.0)
    # A divided statement is the roll's, not an estimate.
    assert out.loc['a1', 'land_value_source'] == 'rollsource'


def test_improvement_and_dwellings_follow_floor_area(run):
    # Two stories on the first lot, one on the second, equal outlines.
    accounts = _accounts({'A': {'improvement_value': 300.0, 'n_dwellings': 3.0}})
    links = _links(
        [
            ('A', 'a1', 0.5, 'parcel_id_local', 'roll'),
            ('A', 'a2', 0.5, 'parcel_id_local', 'roll'),
        ]
    )
    lots = _lots({'a1': (0.1, {}), 'a2': (0.1, {})})
    buildings = _buildings([('b1', 100.0, 2.0, 'obm'), ('b2', 100.0, None, 'obm')])
    overlaps = _overlaps([('b1', 'a1', 100.0), ('b2', 'a2', 100.0)])
    out = run(accounts, links, lots, overlaps, buildings)
    # The building without a story count weighs as one story.
    assert out.loc['a1', 'improvement_value'] == pytest.approx(200.0)
    assert out.loc['a2', 'improvement_value'] == pytest.approx(100.0)
    assert out.loc['a1', 'n_dwellings'] == pytest.approx(2.0)
    assert out.loc['a2', 'n_dwellings'] == pytest.approx(1.0)


def test_a_piece_under_the_noise_floor_carries_no_weight(run):
    # The account's house stands on its first lot. A neighbor's house of
    # 200 m2 reaches 8 m2 into its second lot: under the 10 m2 floor,
    # overlay noise along the lot line.
    accounts = _accounts({'B': {'improvement_value': 500.0}})
    links = _links(
        [
            ('B', 'b1', 0.5, 'parcel_id_local', 'roll'),
            ('B', 'b2', 0.5, 'parcel_id_local', 'roll'),
        ]
    )
    lots = _lots({'b1': (0.1, {}), 'b2': (0.1, {}), 'n1': (0.1, {})})
    buildings = _buildings(
        [('house', 150.0, 1.0, 'obm'), ('neighbor', 200.0, 1.0, 'obm')]
    )
    overlaps = _overlaps(
        [('house', 'b1', 150.0), ('neighbor', 'n1', 192.0), ('neighbor', 'b2', 8.0)]
    )
    out = run(accounts, links, lots, overlaps, buildings)
    assert out.loc['b1', 'improvement_value'] == pytest.approx(500.0)
    assert out.loc['b2', 'improvement_value'] == pytest.approx(0.0)


def test_a_sliver_alone_leaves_the_account_to_lot_area(run):
    # With only a piece under the floor, no lot of the account has
    # weight and improvement follows lot area (ruling 2).
    accounts = _accounts({'B': {'improvement_value': 500.0}})
    links = _links(
        [
            ('B', 'b1', 0.8, 'parcel_id_local', 'roll'),
            ('B', 'b2', 0.2, 'parcel_id_local', 'roll'),
        ]
    )
    lots = _lots({'b1': (0.4, {}), 'b2': (0.1, {})})
    buildings = _buildings([('neighbor', 200.0, 1.0, 'obm')])
    overlaps = _overlaps([('neighbor', 'b2', 8.0)])
    out = run(accounts, links, lots, overlaps, buildings)
    assert out.loc['b1', 'improvement_value'] == pytest.approx(400.0)
    assert out.loc['b2', 'improvement_value'] == pytest.approx(100.0)


def test_a_duplex_split_by_a_lot_line_weighs_on_both_accounts(run):
    # One outline of 100 m2, 60 m2 on account X's lot and 40 m2 on
    # account Y's. Each part belongs to its lot's account; Y also has
    # its own 120 m2 house on a second lot.
    accounts = _accounts(
        {'X': {'improvement_value': 300.0}, 'Y': {'improvement_value': 800.0}}
    )
    links = _links(
        [
            ('X', 'x1', 0.5, 'parcel_id_local', 'roll'),
            ('X', 'x2', 0.5, 'parcel_id_local', 'roll'),
            ('Y', 'y1', 0.5, 'parcel_id_local', 'roll'),
            ('Y', 'y2', 0.5, 'parcel_id_local', 'roll'),
        ]
    )
    lots = _lots({lot: (0.1, {}) for lot in ('x1', 'x2', 'y1', 'y2')})
    buildings = _buildings(
        [('duplex', 100.0, 1.0, 'obm'), ('house', 120.0, 1.0, 'obm')]
    )
    overlaps = _overlaps(
        [('duplex', 'x1', 60.0), ('duplex', 'y1', 40.0), ('house', 'y2', 120.0)]
    )
    out = run(accounts, links, lots, overlaps, buildings)
    assert out.loc['x1', 'improvement_value'] == pytest.approx(300.0)
    assert out.loc['x2', 'improvement_value'] == pytest.approx(0.0)
    assert out.loc['y1', 'improvement_value'] == pytest.approx(800.0 * 40 / 160)
    assert out.loc['y2', 'improvement_value'] == pytest.approx(800.0 * 120 / 160)


def test_a_cluster_outline_beside_a_drawn_footprint_weighs_nothing(run):
    # The account's lots hold a drawn house and a cluster outline: the
    # house takes the account's improvement.
    accounts = _accounts({'C': {'improvement_value': 100.0}})
    links = _links(
        [
            ('C', 'u1', 0.5, 'parcel_id_local', 'roll'),
            ('C', 'u2', 0.5, 'parcel_id_local', 'roll'),
        ]
    )
    lots = _lots({'u1': (0.03, {}), 'u2': (0.03, {})})
    buildings = _buildings(
        [
            ('cluster', 2000.0, 1.0, 'condo_cluster.parcel.src'),
            ('house', 80.0, 1.0, 'obm'),
        ]
    )
    overlaps = _overlaps([('cluster', 'u1', 250.0), ('house', 'u2', 80.0)])
    out = run(accounts, links, lots, overlaps, buildings)
    assert out.loc['u1', 'improvement_value'] == pytest.approx(0.0)
    assert out.loc['u2', 'improvement_value'] == pytest.approx(100.0)


def test_a_cluster_outline_alone_weighs_by_its_pieces(run):
    # A cluster outline of 2,000 m2 covers most of one 300 m2 unit lot
    # and nothing else of the account's lots.
    accounts = _accounts({'C': {'improvement_value': 100.0}})
    links = _links(
        [
            ('C', 'u1', 0.5, 'parcel_id_local', 'roll'),
            ('C', 'u2', 0.5, 'parcel_id_local', 'roll'),
        ]
    )
    lots = _lots({'u1': (0.03, {}), 'u2': (0.03, {})})
    buildings = _buildings([('cluster', 2000.0, 1.0, 'condo_cluster.parcel.src')])
    overlaps = _overlaps([('cluster', 'u1', 250.0)])
    out = run(accounts, links, lots, overlaps, buildings)
    assert out.loc['u1', 'improvement_value'] == pytest.approx(100.0)
    assert out.loc['u2', 'improvement_value'] == pytest.approx(0.0)


def test_a_stand_in_outline_never_weighs(run):
    # A parcel-shaped stand-in on the second lot, a drawn house on the
    # first: the stand-in carries no weight.
    accounts = _accounts({'D': {'improvement_value': 1000.0}})
    links = _links(
        [
            ('D', 'd1', 0.5, 'parcel_id_local', 'roll'),
            ('D', 'd2', 0.5, 'parcel_id_local', 'roll'),
        ]
    )
    lots = _lots({'d1': (0.1, {}), 'd2': (0.1, {})})
    buildings = _buildings(
        [('house', 100.0, 1.0, 'obm'), ('stand_in', 1000.0, None, 'parcel.spine')]
    )
    overlaps = _overlaps([('house', 'd1', 100.0), ('stand_in', 'd2', 1000.0)])
    out = run(accounts, links, lots, overlaps, buildings)
    assert out.loc['d1', 'improvement_value'] == pytest.approx(1000.0)
    assert out.loc['d2', 'improvement_value'] == pytest.approx(0.0)


def test_a_shared_key_account_is_divided_as_any_other(run):
    # Ruling 4: a parent tract the layer draws as platted lots.
    accounts = _accounts({'E': {'land_value': 1000.0, 'total_value': 1000.0}})
    links = _links(
        [('E', f'e{i}', None, 'parcel_id_local_shared_key', 'roll') for i in range(4)]
    )
    lots = _lots(
        {
            f'e{i}': (0.1, {'land_value': 1000.0, 'total_value': 1000.0})
            for i in range(4)
        }
    )
    out = run(accounts, links, lots)
    # No share on the links: the lots' areas divide it.
    assert out['land_value'].tolist() == pytest.approx([250.0] * 4)
    assert out['total_value'].sum() == pytest.approx(1000.0)


def test_a_stated_total_below_its_parts_is_kept_and_divided(run):
    accounts = _accounts(
        {'F': {'land_value': 100.0, 'improvement_value': 300.0, 'total_value': 200.0}}
    )
    links = _links(
        [
            ('F', 'f1', 0.5, 'parcel_id_local', 'roll'),
            ('F', 'f2', 0.5, 'parcel_id_local', 'roll'),
        ]
    )
    lots = _lots({'f1': (0.1, {}), 'f2': (0.1, {})})
    buildings = _buildings([('house', 100.0, 1.0, 'obm')])
    overlaps = _overlaps([('house', 'f1', 100.0)])
    out = run(accounts, links, lots, overlaps, buildings)
    # Parts: f1 = 50 + 300, f2 = 50 + 0; the total 200 by those parts.
    assert out.loc['f1', 'total_value'] == pytest.approx(200.0 * 350 / 400)
    assert out.loc['f2', 'total_value'] == pytest.approx(200.0 * 50 / 400)
    assert not is_imputed(out.loc['f1', 'total_value_source'])


def test_a_total_from_parts_only_where_none_is_stated_and_marked(run):
    accounts = _accounts({'G': {'land_value': 100.0, 'improvement_value': 50.0}})
    links = _links([('G', 'g1', None, 'parcel_id_local', 'roll')])
    lots = _lots({'g1': (0.1, {})})
    out = run(accounts, links, lots)
    assert out.loc['g1', 'total_value'] == pytest.approx(150.0)
    assert is_imputed(out.loc['g1', 'total_value_source'])
    assert not is_imputed(out.loc['g1', 'land_value_source'])


def test_a_stated_zero_stays_zero_and_missing_stays_missing(run):
    accounts = _accounts({'H': {'land_value': 400.0, 'improvement_value': 0.0}})
    links = _links(
        [
            ('H', 'h1', 0.5, 'parcel_id_local', 'roll'),
            ('H', 'h2', 0.5, 'parcel_id_local', 'roll'),
        ]
    )
    lots = _lots(
        {
            'h1': (0.1, {'improvement_value': 0.0}),
            'h2': (0.1, {'improvement_value': 0.0}),
        }
    )
    out = run(accounts, links, lots)
    assert out['improvement_value'].tolist() == [0.0, 0.0]
    # No account states dwellings or other value: the lots keep none.
    assert out['n_dwellings'].isna().all()
    assert out['other_value'].isna().all()


def _dwelling_lots():
    """Three lots whose layer reads 0 dwellings, and one without accounts."""
    return _lots(
        {
            'm1': (0.1, {'n_dwellings': 0.0, 'n_dwellings_source': 'layer'}),
            'm2': (0.1, {'n_dwellings': 0.0, 'n_dwellings_source': 'layer'}),
            'm3': (0.1, {'n_dwellings': 0.0, 'n_dwellings_source': 'layer'}),
            'm4': (0.1, {'n_dwellings': 0.0, 'n_dwellings_source': 'layer'}),
            'm5': (0.1, {'n_dwellings': 3.0, 'n_dwellings_source': 'layer'}),
        }
    )


def test_missing_is_not_zero_for_a_listed_count(run):
    # m1: every account leaves the count blank; m2: an account states 0;
    # m3: one blank and one stating 2; m4: no account at all; m5: blank
    # accounts over a positive layer count.
    accounts = _accounts(
        {
            'A': {'land_value': 100.0},
            'B': {'land_value': 100.0, 'n_dwellings': 0.0},
            'C': {'land_value': 100.0},
            'D': {'land_value': 100.0, 'n_dwellings': 2.0},
            'E': {'land_value': 100.0},
        }
    )
    links = _links(
        [
            ('A', 'm1', 1.0, 'parcel_id_local', 'roll'),
            ('B', 'm2', 1.0, 'parcel_id_local', 'roll'),
            ('C', 'm3', 0.5, 'parcel_id_local', 'roll'),
            ('D', 'm3', 0.5, 'parcel_id_local', 'roll'),
            ('E', 'm5', 1.0, 'parcel_id_local', 'roll'),
        ]
    )
    out = run(accounts, links, _dwelling_lots(), unstated_zero_missing=['n_dwellings'])
    assert np.isnan(out.loc['m1', 'n_dwellings'])
    assert pd.isna(out.loc['m1', 'n_dwellings_source'])
    assert out.loc['m2', 'n_dwellings'] == 0
    assert out.loc['m3', 'n_dwellings'] == 2
    # No account on the lot: the layer's statement stands, 0 included.
    assert out.loc['m4', 'n_dwellings'] == 0
    # A positive count is never a format default.
    assert out.loc['m5', 'n_dwellings'] == 3


def test_without_the_listed_count_a_layer_zero_stays(run):
    accounts = _accounts({'A': {'land_value': 100.0}})
    links = _links([('A', 'm1', 1.0, 'parcel_id_local', 'roll')])
    out = run(accounts, links, _dwelling_lots())
    assert out.loc['m1', 'n_dwellings'] == 0


def test_the_parcel_curate_lists_the_dwelling_count():
    from openplaces.recipe import get_recipe_by_id

    step = next(
        e
        for e in get_recipe_by_id('US_parcel-openplaces-2026')['pipeline']
        if e.get('step') == 'divide_account_values'
    )
    assert step['unstated_zero_missing'] == ['n_dwellings']


def test_a_lot_whose_improvement_went_to_the_other_lot_is_marked(run):
    # V3 deviation 5: the account's improvement stands on a1, so a2's
    # 0 is where the roll put it, and the step says so.
    accounts = _accounts({'A': {'land_value': 100.0, 'improvement_value': 300.0}})
    links = _links(
        [
            ('A', 'a1', 0.5, 'parcel_id_local', 'roll'),
            ('A', 'a2', 0.5, 'parcel_id_local', 'roll'),
        ]
    )
    lots = _lots({'a1': (0.1, {}), 'a2': (0.1, {}), 'free': (0.1, {})})
    buildings = _buildings([('b1', 100.0, 1.0, 'obm')])
    overlaps = _overlaps([('b1', 'a1', 100.0)])
    out = run(
        accounts,
        links,
        lots,
        overlaps,
        buildings,
        on_other_lots={'improvement_value': 'improvement_value_on_other_lots'},
    )
    assert out.loc['a2', 'improvement_value'] == 0.0
    assert out['improvement_value_on_other_lots'].to_dict() == {
        'a1': False,
        'a2': True,
        'free': False,
    }


def test_a_total_only_account_takes_the_layers_repeated_parts_once(run):
    # Ruling 5: the roll states a total only; the statewide layer
    # repeats the account's land and improvement on each of its two
    # lots.
    accounts = _accounts({'T': {'total_value': 1000.0}})
    links = _links(
        [
            ('T', 't1', 0.75, 'parcel_id_local', 'roll'),
            ('T', 't2', 0.25, 'parcel_id_local', 'roll'),
        ]
    )
    layer = {
        'land_value': 400.0,
        'improvement_value': 800.0,
        'land_value_source': 'layersrc',
        'improvement_value_source': 'layersrc',
    }
    lots = _lots({'t1': (0.3, dict(layer)), 't2': (0.1, dict(layer))})
    buildings = _buildings([('house', 100.0, 1.0, 'obm')])
    overlaps = _overlaps([('house', 't2', 100.0)])
    out = run(
        accounts,
        links,
        lots,
        overlaps,
        buildings,
        layer_columns=['land_value', 'improvement_value'],
    )
    assert out['land_value'].tolist() == pytest.approx([300.0, 100.0])
    # Improvement follows the house, on the second lot.
    assert out['improvement_value'].tolist() == pytest.approx([0.0, 800.0])
    # The roll's total by lot area.
    assert out['total_value'].tolist() == pytest.approx([750.0, 250.0])
    assert out.loc['t1', 'land_value_source'] == 'layersrc'
    assert out.loc['t1', 'total_value_source'] == 'rollsource'


def test_the_layer_is_divided_only_over_lots_where_it_states_a_value(run):
    # L25: the third lot has no layer value and receives none.
    accounts = _accounts({'T': {'total_value': 900.0}})
    links = _links([('T', f't{i}', 1 / 3, 'parcel_id_local', 'roll') for i in range(3)])
    lots = _lots(
        {
            't0': (0.1, {'land_value': 600.0}),
            't1': (0.1, {'land_value': 600.0}),
            't2': (0.1, {}),
        }
    )
    out = run(accounts, links, lots, layer_columns=['land_value'])
    assert out['land_value'].tolist()[:2] == pytest.approx([300.0, 300.0])
    assert np.isnan(out.loc['t2', 'land_value'])


def test_layer_values_that_differ_between_lots_are_each_lots_own(run):
    accounts = _accounts({'T': {'total_value': 900.0}})
    links = _links(
        [
            ('T', 't0', 0.5, 'parcel_id_local', 'roll'),
            ('T', 't1', 0.5, 'parcel_id_local', 'roll'),
        ]
    )
    lots = _lots(
        {'t0': (0.1, {'land_value': 600.0}), 't1': (0.1, {'land_value': 100.0})}
    )
    out = run(accounts, links, lots, layer_columns=['land_value'])
    assert out['land_value'].tolist() == [600.0, 100.0]


def test_an_account_with_no_building_divides_improvement_by_area(run):
    accounts = _accounts({'K': {'improvement_value': 100.0}})
    links = _links(
        [
            ('K', 'k1', 0.9, 'parcel_id_local', 'roll'),
            ('K', 'k2', 0.1, 'parcel_id_local', 'roll'),
        ]
    )
    lots = _lots({'k1': (0.9, {}), 'k2': (0.1, {})})
    buildings = _buildings([('far', 100.0, 1.0, 'obm')])
    overlaps = _overlaps([('far', 'elsewhere', 100.0)])
    out = run(accounts, links, lots, overlaps, buildings)
    assert out['improvement_value'].tolist() == pytest.approx([90.0, 10.0])


def test_a_one_lot_account_is_unchanged(run):
    accounts = _accounts(
        {
            'L': {
                'land_value': 100.0,
                'improvement_value': 200.0,
                'total_value': 300.0,
                'n_dwellings': 1.0,
            }
        }
    )
    links = _links([('L', 'l1', None, 'parcel_id_local', 'roll')])
    lots = _lots(
        {
            'l1': (
                0.1,
                {
                    'land_value': 100.0,
                    'improvement_value': 200.0,
                    'total_value': 300.0,
                    'n_dwellings': 1.0,
                },
            )
        }
    )
    out = run(accounts, links, lots)
    for column in ('land_value', 'improvement_value', 'total_value', 'n_dwellings'):
        assert out.loc['l1', column] == pytest.approx(lots.loc['l1', column])


def test_a_lot_of_two_accounts_takes_the_sum_of_their_parts(run):
    accounts = _accounts({'M': {'land_value': 100.0}, 'N': {'land_value': 50.0}})
    links = _links(
        [
            ('M', 'm1', None, 'parcel_id_local', 'roll'),
            ('N', 'm1', 0.5, 'parcel_id_local', 'roll'),
            ('N', 'm2', 0.5, 'parcel_id_local', 'roll'),
        ]
    )
    lots = _lots(
        {'m1': (0.1, {'land_value': 150.0}), 'm2': (0.1, {'land_value': 50.0})}
    )
    out = run(accounts, links, lots)
    assert out['land_value'].tolist() == pytest.approx([125.0, 25.0])


def test_units_rows_stay_out_where_a_roll_states_the_column(run):
    # The roll account states the lot's land; the units split off the
    # parcel layer state land too, and are left out of that column.
    accounts = _accounts({'R': {'land_value': 100.0}, 'U': {'land_value': 40.0}})
    links = _links(
        [
            ('R', 'r1', None, 'parcel_id_local', 'roll'),
            ('U', 'r1', None, 'parcel_id_local', 'layer:units'),
        ]
    )
    lots = _lots({'r1': (0.1, {})})
    out = run(accounts, links, lots)
    assert out.loc['r1', 'land_value'] == pytest.approx(100.0)


def test_area_shares_fall_back_to_lot_area_where_a_share_is_missing():
    account = pd.Series(['a', 'a', 'b'])
    lot = pd.Series(['x', 'y', 'z'])
    share = pd.Series([0.5, None, None])
    area = pd.Series({'x': 3.0, 'y': 1.0, 'z': 2.0})
    shares = account_values.area_shares(account, lot, share, area)
    assert shares.tolist() == pytest.approx([0.75, 0.25, 1.0])


def test_lot_piece_floor_area_sums_pieces_above_the_floor():
    overlaps = pd.DataFrame(
        {
            'building': ['own', 'neighbor', 'big', 'tiny'],
            'lot': ['x', 'x', 'x', 'x'],
            'overlap_m2': [60.0, 10.0, 300.0, 9.9],
        }
    )
    stories = pd.Series({'own': 2.0, 'neighbor': np.nan, 'big': 3.0, 'tiny': 1.0})
    floor = account_values.lot_piece_floor_area(overlaps, stories, 10.0)
    # 60 x 2 + 10 x 1 (no count: one story) + 300 x 3; the 9.9 m2 piece
    # is under the floor.
    assert floor['x'] == pytest.approx(1030.0)


def test_conservation_reports_no_failure_on_a_divided_account():
    frame = pd.DataFrame(
        {
            'account': ['a', 'a'],
            'lot': ['x', 'y'],
            'share': [0.4, 0.6],
            'floor': [0.0, 50.0],
            'land_value': [10.0, 10.0],
            'improvement_value': [7.0, 7.0],
            'other_value': [np.nan, np.nan],
            'total_value': [20.0, 20.0],
        }
    )
    out, _ = account_values.divide(
        frame,
        by_area=['land_value', 'other_value'],
        by_floor_area=['improvement_value'],
        total='total_value',
        parts=['land_value', 'other_value', 'improvement_value'],
    )
    accounts = pd.DataFrame(
        {'land_value': [10.0], 'improvement_value': [7.0], 'total_value': [20.0]},
        index=['a'],
    )
    table = account_values.conservation(
        out,
        accounts,
        ['land_value', 'improvement_value', 'total_value'],
        pd.Series(False, index=['a']),
    )
    assert table['accounts_failing'].tolist() == [0, 0, 0]
    assert out['improvement_value'].tolist() == pytest.approx([0.0, 7.0])
