"""The footprint curate's fold (US_footprint-cheer-2026) over parcel pieces.

plans/buildings-from-parcel-pieces.md, step 3: once a townhome row's
outline is cut into one building per lot, the footprint curate folds
the buildings back onto the outline. The rows and ids are the outline
spine's as before, and an extensive value per outline adds up to what
the uncut building carried. The step runs with the recipe's own
configuration. Every id and value is fabricated
(docs/5_contribute/no-personal-data.rst).
"""

from __future__ import annotations

import pandas as pd
import pytest

import openplaces.geo.link as link_module
import openplaces.io.harmonizer.entity_links as entity_links
from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState, aggregation
from openplaces.recipe import get_recipe_by_id

RECIPE = 'US_footprint-cheer-2026'


def _fold_step():
    step = next(
        s
        for s in get_recipe_by_id(RECIPE)['pipeline']
        if isinstance(s, dict) and s.get('step') == 'aggregate_from_entities'
    )
    return {k: v for k, v in step.items() if k != 'step'}


def _uncut_buildings():
    """f1 a row of four drawn as one building over lots P0 to P3; f2 a
    house on lot P4."""
    return pd.DataFrame(
        {
            'footprint_id': ['f1', 'f2'],
            'parcel_id': ['P0', 'P4'],
            'parcel_id_local': ['L0', 'L4'],
            'occupancy_type': ['Single-Family', 'Single-Family'],
            'occupancy_type_source': ['parcel', 'parcel'],
            'structure_value': [460000.0, 90000.0],
            'structure_value_source': ['parcel', 'parcel'],
            'land_value': [90000.0, 20000.0],
            'land_value_source': ['parcel', 'parcel'],
            'n_dwellings': [4.0, 1.0],
            'year_built': [1985.0, 1990.0],
            'n_stories': [2.0, 1.0],
            'n_parcels_per_building': [4.0, 1.0],
            'n_buildings_per_parcel': [1.0, 1.0],
            'area_ha': [0.04, 0.01],
            'area_m2': [400.0, 100.0],
            'lat': [10.0, 11.0],
        },
        index=pd.Index(['f1', 'f2'], name='building_id'),
    )


def _cut_buildings():
    """f1 cut into four pieces, each valued from its own lot."""
    uncut = _uncut_buildings()
    pieces = pd.DataFrame(
        {
            'footprint_id': ['f1'] * 4,
            'parcel_id': ['P0', 'P1', 'P2', 'P3'],
            'parcel_id_local': ['L0', 'L1', 'L2', 'L3'],
            'occupancy_type': ['Single-Family'] * 4,
            'occupancy_type_source': ['parcel'] * 4,
            'structure_value': [110000.0, 110000.0, 120000.0, 120000.0],
            'structure_value_source': ['parcel'] * 4,
            'land_value': [22000.0, 22000.0, 23000.0, 23000.0],
            'land_value_source': ['parcel'] * 4,
            'n_dwellings': [1.0] * 4,
            'year_built': [1985.0] * 4,
            'n_stories': [2.0] * 4,
            'n_parcels_per_building': [1.0] * 4,
            'n_buildings_per_parcel': [1.0] * 4,
            'area_ha': [0.01] * 4,
            # The dominant piece (P0's) is the outline less the others,
            # so it is the largest.
            'area_m2': [130.0, 90.0, 90.0, 90.0],
            'lat': [9.9, 9.95, 10.05, 10.1],
        },
        index=pd.Index(['c0', 'c1', 'c2', 'c3'], name='building_id'),
    )
    return pd.concat([pieces, uncut.loc[['f2']]])


def _fold(monkeypatch, tmp_path, buildings, step=None):
    links = pd.DataFrame(
        {
            'building_id': buildings.index.to_numpy(),
            'footprint_id': buildings['footprint_id'].to_numpy(),
            'link_method': 'footprint_outline',
            'link_source': 'footprint',
            'share': pd.Series([pd.NA] * len(buildings), dtype='Float64'),
        }
    )
    monkeypatch.setattr(
        aggregation,
        'get_entities',
        lambda recipe_id, admin_id, columns=None, missing='raise': buildings[
            [c for c in columns if c in buildings.columns]
        ],
    )
    monkeypatch.setattr(
        aggregation,
        'describe_recipe',
        lambda recipe_id, admin_id: pd.DataFrame(index=buildings.columns),
    )
    monkeypatch.setattr(
        aggregation, '_linked_recipe_id', lambda recipe_id: 'US_building-spine-2026'
    )
    monkeypatch.setattr(
        link_module,
        'get_link_owner_recipe_id',
        lambda recipe: (
            'US_building-geospine-2026'
            if recipe == 'US_building-spine-2026'
            else 'US_footprint-geospine-2026'
        ),
    )
    monkeypatch.setattr(
        link_module,
        'get_entity_link_path',
        lambda a, b, admin_id=None: tmp_path / f'{a}__{b}.parquet',
    )
    (tmp_path / 'US_building-geospine-2026__US_footprint-geospine-2026.parquet').touch()
    monkeypatch.setattr(
        entity_links,
        'read_entity_link',
        lambda path: links if path.name.startswith('US_building-geospine') else None,
    )
    outlines = pd.DataFrame(
        {
            'area_ha': [0.04, 0.01],
            'lat': [10.0, 11.0],
            'geometry_source': ['obm', 'obm'],
        },
        index=pd.Index(['f1', 'f2'], name='footprint_id'),
    )
    state = CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US', 'XX', 'YY'),
        verbose=False,
        timer=None,
        curated=outlines,
    )
    return aggregation.aggregate_from_entities(state, **(step or _fold_step())).curated


@pytest.fixture
def folded(monkeypatch, tmp_path):
    uncut = _fold(monkeypatch, tmp_path, _uncut_buildings())
    cut = _fold(monkeypatch, tmp_path, _cut_buildings())
    return uncut, cut


def test_the_rows_and_ids_are_the_outlines(folded):
    uncut, cut = folded
    assert cut.index.tolist() == uncut.index.tolist() == ['f1', 'f2']


def test_extensive_values_per_outline_add_up(folded):
    uncut, cut = folded
    for column in ('structure_value', 'land_value', 'n_dwellings'):
        assert cut[column].tolist() == pytest.approx(uncut[column].tolist())
    assert cut['n_buildings_per_footprint'].tolist() == [4, 1]
    assert cut['structure_value_source'].tolist() == ['parcel', 'parcel']


def test_classes_and_the_outlines_own_columns_stay(folded):
    uncut, cut = folded
    assert cut['occupancy_type'].tolist() == uncut['occupancy_type'].tolist()
    assert cut['year_built'].tolist() == uncut['year_built'].tolist()
    assert cut['n_stories'].tolist() == uncut['n_stories'].tolist()
    # keep: the outline's centroid and area, not its buildings'.
    assert cut['lat'].tolist() == [10.0, 11.0]
    assert cut['area_ha'].tolist() == [0.04, 0.01]
    # An uncut outline is a copy of its building.
    assert cut.loc['f2'].equals(uncut.loc['f2'])


def test_the_outlines_parcel_span_is_its_own(folded):
    """Each lot is linked to one piece: the pieces' counts sum to the
    outline's, and one building per outline keeps its own."""
    uncut, cut = folded
    assert uncut['n_parcels_per_footprint'].tolist() == [4, 1]
    assert cut['n_parcels_per_footprint'].tolist() == [4, 1]


def test_the_outlines_parcel_is_one_lot(folded):
    """The dominant piece's lot, and its local id with it."""
    uncut, cut = folded
    assert uncut.loc['f1', 'parcel_id'] == 'P0'
    assert cut.loc['f1', 'parcel_id'] == 'P0'
    assert cut['parcel_id_local'].tolist() == uncut['parcel_id_local'].tolist()
    assert cut['parcel_id_local'].tolist() == ['L0', 'L4']


def test_the_dominant_piece_leads_whatever_the_row_order(monkeypatch, tmp_path):
    """The link table's row order is not fixed: the largest piece names
    the lot from any position, and an area tie goes to the lower id."""
    reversed_rows = _cut_buildings().iloc[::-1]
    cut = _fold(monkeypatch, tmp_path, reversed_rows)
    assert cut.loc['f1', 'parcel_id'] == 'P0'
    assert cut.loc['f1', 'parcel_id_local'] == 'L0'

    tied = _cut_buildings()
    tied.loc[['c0', 'c1', 'c2', 'c3'], 'area_m2'] = 100.0
    cut = _fold(monkeypatch, tmp_path, tied.iloc[::-1])
    assert cut.loc['f1', 'parcel_id'] == 'P0'


def test_a_misspelled_direction_is_refused(monkeypatch, tmp_path):
    step = {**_fold_step(), 'order_by': {'area_m2': 'largest'}}
    with pytest.raises(ValueError, match='ascending or descending'):
        _fold(monkeypatch, tmp_path, _cut_buildings(), step)


def test_every_lot_of_the_outline_is_listed_after_the_fold():
    """The pieces cannot list the outline's lots between them, so the
    recipe collects them from the outline's own overlay after the
    fold, where they overwrite what the fold wrote."""
    steps = [s for s in get_recipe_by_id(RECIPE)['pipeline'] if isinstance(s, dict)]
    names = [s.get('step') for s in steps]
    collect = next(
        i
        for i, s in enumerate(steps)
        if s.get('step') == 'collect_link_ids' and s.get('column') == 'parcel_id_all'
    )
    assert collect > names.index('aggregate_from_entities')


def test_the_pieces_assessor_numbers_merge_into_one_list(monkeypatch, tmp_path):
    """Each piece lists only its own lots, and a piece on one lot lists
    none: the fold merges the lists member by member, the largest
    piece's lots first, so the outline names every lot once."""
    uncut = _uncut_buildings()
    uncut['parcel_id_local_all'] = ['L0 + L1 + L2 + L3', pd.NA]
    cut = _cut_buildings()
    # c2 also reaches a sliver of L3, which c3 stands on.
    cut['parcel_id_local_all'] = [pd.NA, pd.NA, 'L2 + L3', pd.NA, pd.NA]
    for rows in (cut, cut.iloc[::-1]):
        folded = _fold(monkeypatch, tmp_path, rows)
        merged = folded.loc['f1', 'parcel_id_local_all']
        assert merged.split(' + ')[0] == 'L0'
        assert sorted(merged.split(' + ')) == ['L0', 'L1', 'L2', 'L3']
        # One lot: missing, as the overlay leaves it.
        assert pd.isna(folded.loc['f2', 'parcel_id_local_all'])
    folded = _fold(monkeypatch, tmp_path, uncut)
    assert folded.loc['f1', 'parcel_id_local_all'] == 'L0 + L1 + L2 + L3'
