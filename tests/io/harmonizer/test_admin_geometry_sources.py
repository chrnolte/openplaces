"""Admin geometry comes from the first source that has a polygon.

Three behaviors of the admin harmonize steps, each with the case that
motivated it:

- merge_global falls through to the next covering source for a unit
  its first source has no polygon for (FR's INSEE carries names only,
  so France's regions had no geometry although geoBoundaries has it);
- discover_sources leaves out restricted sources (GADM), so they
  cannot fill the gaps the fall-through would otherwise hand them;
- carve_lumped_territories gives a territory from a finer source the
  area a coarser source lumped into its sovereign, and nothing else.

Geometries are fabricated squares, not real boundaries.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, box

from openplaces.core.schema import AdminId
from openplaces.io.admin import ids as ids_module
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import discover as discover_module


def _state(spine=None, admin_level=1):
    return HarmonizeState(
        recipe={'admin_id': AdminId(), 'admin_level': admin_level},
        admin_id=None,
        verbose=False,
        timer=None,
        spine=spine,
    )


def _gdf(rows):
    return gpd.GeoDataFrame(
        {'name': list(rows)},
        geometry=list(rows.values()),
        index=pd.Index(list(rows), name='admin1_id'),
        crs='EPSG:4326',
    )


class TestFallThrough:
    def test_a_unit_missing_from_the_first_source_comes_from_the_next(
        self, monkeypatch
    ):
        outputs = {
            'national': _gdf({'AA': box(0, 0, 1, 1)}),
            'global': _gdf({'AA': box(5, 5, 6, 6), 'BB': box(1, 0, 2, 1)}),
        }
        monkeypatch.setattr(
            discover_module, 'get_entities', lambda rid, geom: outputs[rid]
        )
        state = _state()
        state.metadata['discovered_sources'] = [
            {'recipe_id': 'national'},
            {'recipe_id': 'global'},
        ]
        state.metadata['source_candidates'] = {
            'AA': ['national', 'global'],
            'BB': ['national', 'global'],
        }

        state = discover_module.merge_global(state)

        assert state.metadata['geometry_source'] == {
            'AA': 'national',
            'BB': 'global',
        }
        # The first source wins where it has the unit.
        assert state.spine.loc['AA'].geometry.equals(box(0, 0, 1, 1))

    def test_a_source_that_cannot_be_read_is_skipped(self, monkeypatch):
        def read(rid, geom):
            if rid == 'names_only':
                raise FileNotFoundError('no _geo file')
            return _gdf({'AA': box(0, 0, 1, 1)})

        monkeypatch.setattr(discover_module, 'get_entities', read)
        state = _state()
        state.metadata['discovered_sources'] = [
            {'recipe_id': 'names_only'},
            {'recipe_id': 'global'},
        ]
        state.metadata['source_candidates'] = {'AA': ['names_only', 'global']}

        with pytest.warns(UserWarning, match='Could not load names_only'):
            state = discover_module.merge_global(state)

        assert state.metadata['geometry_source'] == {'AA': 'global'}
        assert state.metadata['unmatched_admin_ids'] == []

    def test_an_empty_geometry_falls_through(self, monkeypatch):
        empty = _gdf({'AA': box(0, 0, 1, 1)})
        empty.loc['AA', 'geometry'] = None
        outputs = {'first': empty, 'second': _gdf({'AA': box(0, 0, 1, 1)})}
        monkeypatch.setattr(
            discover_module, 'get_entities', lambda rid, geom: outputs[rid]
        )
        state = _state()
        state.metadata['discovered_sources'] = [
            {'recipe_id': 'first'},
            {'recipe_id': 'second'},
        ]
        state.metadata['source_candidates'] = {'AA': ['first', 'second']}

        state = discover_module.merge_global(state)

        assert state.metadata['geometry_source'] == {'AA': 'second'}


class TestRestrictedSources:
    def test_a_restricted_source_is_not_a_candidate(self, monkeypatch):
        sources = [
            {'recipe_id': 'open', 'admin_id': '', 'restricted': False},
            {'recipe_id': 'gadm_like', 'admin_id': '', 'restricted': True},
        ]
        monkeypatch.setattr(
            discover_module, '_scan_admin_ingest_recipes', lambda level: sources
        )
        monkeypatch.setattr(
            discover_module,
            'get_admin_ids',
            lambda level, admin_id=None, allow_empty=False: ['AA'],
        )

        state = discover_module.discover_sources(_state())

        assert state.metadata['source_candidates'] == {'AA': ['open']}

    def test_a_recipe_can_opt_back_in(self, monkeypatch):
        sources = [
            {'recipe_id': 'gadm_like', 'admin_id': '', 'restricted': True},
        ]
        monkeypatch.setattr(
            discover_module, '_scan_admin_ingest_recipes', lambda level: sources
        )
        monkeypatch.setattr(
            discover_module,
            'get_admin_ids',
            lambda level, admin_id=None, allow_empty=False: ['AA'],
        )

        state = discover_module.discover_sources(_state(), include_restricted=True)

        assert state.metadata['source_candidates'] == {'AA': ['gadm_like']}


def _resolve(rows, origin, thresholds=None):
    state = _state(spine=_gdf(rows))
    state.metadata['discovered_sources'] = [
        {'recipe_id': 'primary'},
        {'recipe_id': 'filler'},
    ]
    state.metadata['geometry_source'] = origin
    return discover_module.carve_lumped_territories(state, thresholds=thresholds)


class TestCarveLumpedTerritories:
    def test_a_lumped_island_is_handed_over_with_the_primary_outline(self):
        mainland = box(0, 0, 10, 10)
        island = box(20, 0, 21, 1)
        # The filler draws the island a little coarser than the primary.
        coarse_island = box(19.9, -0.1, 21.1, 1.1)
        state = _resolve(
            {'SV': MultiPolygon([mainland, island]), 'TT': coarse_island},
            {'SV': 'primary', 'TT': 'filler'},
        )

        spine = state.spine
        assert spine.loc['SV'].geometry.equals(mainland)
        assert spine.loc['TT'].geometry.equals(island)

    def test_a_territory_inside_the_primary_land_is_cut_out(self):
        land = box(0, 0, 10, 10)
        base = box(1, 1, 2, 2)
        state = _resolve({'SV': land, 'TT': base}, {'SV': 'primary', 'TT': 'filler'})

        spine = state.spine
        assert spine.loc['TT'].geometry.area == pytest.approx(1.0)
        assert spine.loc['SV'].geometry.area == pytest.approx(99.0)
        assert not spine.loc['SV'].geometry.intersection(spine.loc['TT'].geometry).area

    def test_a_coarse_border_does_not_move_a_neighbors_land(self):
        neighbor = box(0, 0, 10, 10)
        # The filler's unit runs 0.2 into the neighbor along a border
        # that is 10 long: 2 of its 20 in area, under cut_share.
        territory = box(9.8, 0, 11.8, 10)
        state = _resolve(
            {'NB': neighbor, 'TT': territory},
            {'NB': 'primary', 'TT': 'filler'},
        )

        spine = state.spine
        assert spine.loc['NB'].geometry.equals(neighbor)
        assert spine.loc['TT'].geometry.area == pytest.approx(18.0)

    def test_units_of_one_source_are_left_alone(self):
        state = _resolve(
            {'AA': box(0, 0, 1, 1), 'BB': box(1, 0, 2, 1)},
            {'AA': 'primary', 'BB': 'primary'},
        )

        assert 'overlaps_resolved' not in state.metadata


class TestIndexFromCodes:
    @pytest.fixture(autouse=True)
    def _spine(self, monkeypatch):
        spine = pd.DataFrame(
            {
                'name': ['Aland', 'Bland', 'Cland'],
                'admin1_id_a3': ['AAA', 'BBB', 'CCC'],
            },
            index=pd.Index(['AA', 'BB', 'CC'], name='admin1_id'),
        )
        monkeypatch.setattr(ids_module, 'get_admin', lambda level, all_columns: spine)

    def test_codes_are_tried_in_order_remapped_and_dissolved(self):
        gdf = gpd.GeoDataFrame(
            {
                'name': ['one', 'two', 'three', 'zone'],
                'own': ['AAA', 'XBB', 'YYY', '111'],
                'sovereign': ['CCC', 'CCC', 'BBB', '111'],
            },
            geometry=[
                box(0, 0, 1, 1),
                box(2, 0, 3, 1),
                box(4, 0, 5, 1),
                box(9, 9, 10, 10),
            ],
            crs='EPSG:4326',
        )

        with pytest.warns(UserWarning, match='1 row'):
            out = ids_module.admin1_id_index_from_codes(
                gdf, ['own', 'sovereign'], remap={'XBB': 'BBB'}
            )

        assert list(out.index) == ['AA', 'BB']
        # 'two' by its remapped own code, 'three' by its sovereign's:
        # one unit, dissolved.
        assert out.loc['BB'].geometry.area == pytest.approx(2.0)
