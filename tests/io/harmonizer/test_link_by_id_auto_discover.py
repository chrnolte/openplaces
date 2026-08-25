"""Tests for `link_by_id`'s auto-discovery (replaces a single hardcoded
per-admin `recipe_id` with admin-scoped discovery of every applicable ingest
source, plus an auto-applied `*-remap.csv` crosswalk).

Uses the real bundled MA (MassGIS, with a `property` `additional_layers`
entry declaring `layer_key: parcel_id_admin2`) and NC (nconemap geometry +
standalone nhcgov roll) recipes as discovery fixtures, since the discovery
helpers read real recipe files from the recipe tree.
"""

import pandas as pd

import openplaces.io.harmonizer.links as links
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState


def _state(admin_id, spine=None, verbose=False):
    return HarmonizeState(
        recipe={},
        admin_id=AdminId(admin_id),
        verbose=verbose,
        timer=None,
        spine=spine,
    )


def test_discover_link_sources_ma_property_layer_uses_layer_key():
    state = _state('US-MA-SOM')
    matches = links._discover_link_sources(state, 'parcel')

    primary = next(m for m in matches if m['layer'] is None)
    assert primary['recipe_id'] == 'US-MA_parcel-massgis-2025'
    assert primary['key'] == 'parcel_id_local'

    layer = next(m for m in matches if m['layer'] == 'property')
    assert layer['recipe_id'] == 'US-MA_parcel-massgis-2025'
    assert layer['key'] == 'parcel_id_admin2'


def test_discover_link_sources_nc_finds_geometry_and_standalone_roll():
    state = _state('US-NC-NHA')
    matches = links._discover_link_sources(state, 'parcel')
    recipe_ids = {m['recipe_id'] for m in matches}

    assert 'US-NC_parcel-nconemap-2025' in recipe_ids
    assert 'US-NC-NHA_parcel-nhcgov-2026' in recipe_ids
    nhcgov = next(
        m for m in matches if m['recipe_id'] == 'US-NC-NHA_parcel-nhcgov-2026'
    )
    assert nhcgov['layer'] is None
    assert nhcgov['key'] == 'parcel_id_local'


def test_discover_link_sources_no_roll_for_uncovered_county():
    # Brunswick (US-NC-BR) has no local roll recipe; only the state geometry
    # (which contributes no value columns) should be discovered, no crash.
    state = _state('US-NC-BR')
    matches = links._discover_link_sources(state, 'parcel')
    assert all(m['recipe_id'] == 'US-NC_parcel-nconemap-2025' for m in matches)


def _recipe_row(admin_id, source_id, version, filename_suffix='', **kwargs):
    """Build one find_recipes() row for the _find_admin_scoped_recipe_ids tests."""
    recipe_id = f'{admin_id}_parcel-{source_id}-{version}'
    if filename_suffix:
        recipe_id += f'_{filename_suffix}'
    return {
        'admin_id': admin_id,
        'source_id': source_id,
        'version': version,
        'exclude_from_auto_discover': False,
        'recipe_id': recipe_id,
        'filename_suffix': filename_suffix,
        **kwargs,
    }


def test_find_admin_scoped_recipe_ids_keeps_newest_version(monkeypatch):
    rows = pd.DataFrame(
        [
            _recipe_row('US-MA', 'massgis', '2024'),
            _recipe_row('US-MA', 'massgis', '2025'),
            _recipe_row('US-CA', 'other', '2020'),
        ]
    )
    monkeypatch.setattr(links, 'find_recipes', lambda *a, **k: rows)

    state = _state('US-MA-SOM')
    ids = links._find_admin_scoped_recipe_ids(state, 'parcel')

    assert ids == ['US-MA_parcel-massgis-2025']  # newest version only, US-CA excluded


def test_find_admin_scoped_recipe_ids_keeps_distinct_filename_suffixes(monkeypatch):
    # Two recipes sharing admin_id/source_id/version but distinguished by a
    # filename suffix (e.g. a PACS roll's own APPRAISAL_INFO recipe and its
    # _improvement-detail sibling) must both survive the dedup -- they are
    # not competing versions of the same source.
    rows = pd.DataFrame(
        [
            _recipe_row('US-TX-VIC', 'victoriacad', '2026'),
            _recipe_row('US-TX-VIC', 'victoriacad', '2026', 'improvement-detail'),
        ]
    )
    monkeypatch.setattr(links, 'find_recipes', lambda *a, **k: rows)

    state = _state('US-TX-VIC')
    ids = links._find_admin_scoped_recipe_ids(state, 'parcel')

    assert ids == [
        'US-TX-VIC_parcel-victoriacad-2026',
        'US-TX-VIC_parcel-victoriacad-2026_improvement-detail',
    ]


def test_find_admin_scoped_recipe_ids_orders_by_specificity_then_version(monkeypatch):
    rows = pd.DataFrame(
        [
            _recipe_row('US-NC-NHA', 'nhcgov', '2026'),
            _recipe_row('US-NC', 'nconemap', '2025'),
        ]
    )
    monkeypatch.setattr(links, 'find_recipes', lambda *a, **k: rows)

    state = _state('US-NC-NHA')
    ids = links._find_admin_scoped_recipe_ids(state, 'parcel')

    # Broader-scope 'US-NC' (2025) sorts first, county-scoped 'US-NC-NHA'
    # (2026) sorts last (wins link_by_id's write-priority): admin
    # specificity decides join order here, version merely happens to agree
    # with it in this fixture (see the disagreeing case below).
    assert ids == ['US-NC_parcel-nconemap-2025', 'US-NC-NHA_parcel-nhcgov-2026']


def test_find_admin_scoped_recipe_ids_specificity_beats_newer_version(monkeypatch):
    # A newer-versioned but less-specific state recipe must NOT out-rank an
    # older-versioned but more-specific county recipe: specificity is the
    # primary sort key, version only breaks ties within the same tier.
    rows = pd.DataFrame(
        [
            _recipe_row('US-NC', 'nconemap', '2026'),
            _recipe_row('US-NC-BL', 'bladenco', '2020'),
        ]
    )
    monkeypatch.setattr(links, 'find_recipes', lambda *a, **k: rows)

    state = _state('US-NC-BL')
    ids = links._find_admin_scoped_recipe_ids(state, 'parcel')

    assert ids == ['US-NC_parcel-nconemap-2026', 'US-NC-BL_parcel-bladenco-2020']


def test_find_admin_scoped_recipe_ids_version_tiebreaks_same_specificity(monkeypatch):
    # At the same admin specificity, version remains the tiebreaker.
    rows = pd.DataFrame(
        [
            _recipe_row('US-NC-BL', 'old_source', '2019'),
            _recipe_row('US-NC-BL', 'bladenco', '2026'),
        ]
    )
    monkeypatch.setattr(links, 'find_recipes', lambda *a, **k: rows)

    state = _state('US-NC-BL')
    ids = links._find_admin_scoped_recipe_ids(state, 'parcel')

    assert ids == ['US-NC-BL_parcel-old_source-2019', 'US-NC-BL_parcel-bladenco-2026']


def test_find_admin_scoped_recipe_ids_skips_excluded_recipe(monkeypatch):
    rows = pd.DataFrame(
        [
            _recipe_row('US-MA', 'massgis', '2025'),
            _recipe_row(
                'US-MA', 'placeslab', 'fmv2026', exclude_from_auto_discover=True
            ),
        ]
    )
    monkeypatch.setattr(links, 'find_recipes', lambda *a, **k: rows)

    state = _state('US-MA-SOM')
    ids = links._find_admin_scoped_recipe_ids(state, 'parcel')

    assert ids == ['US-MA_parcel-massgis-2025']


def test_write_prioritized_new_column_is_written_directly():
    spine = pd.DataFrame(index=['a', 'b'])
    new_vals = pd.Series([1.0, 2.0], index=['a', 'b'])
    links._write_prioritized(spine, 'value', new_vals)
    assert spine['value'].tolist() == [1.0, 2.0]


def test_write_prioritized_majority_coverage_overwrites_existing():
    spine = pd.DataFrame({'value': [10.0, 20.0, 30.0]}, index=['a', 'b', 'c'])
    # New source covers all 3 rows (majority) with different values; it should win.
    new_vals = pd.Series([11.0, 21.0, 31.0], index=['a', 'b', 'c'])
    links._write_prioritized(spine, 'value', new_vals)
    assert spine['value'].tolist() == [11.0, 21.0, 31.0]


def test_write_prioritized_sparse_source_only_fills_gaps():
    spine = pd.DataFrame({'value': [10.0, 20.0, None]}, index=['a', 'b', 'c'])
    # New source only covers 1/3 rows (not a majority); must not clobber a/b,
    # but should still fill c's gap.
    new_vals = pd.Series([None, None, 30.0], index=['a', 'b', 'c'])
    links._write_prioritized(spine, 'value', new_vals)
    assert spine['value'].tolist() == [10.0, 20.0, 30.0]


def test_link_by_id_auto_discover_recent_majority_source_wins_use_subgroup(
    monkeypatch,
):
    # Mirrors the NC-NE case: an older statewide source provides a clean
    # use_subgroup label for every parcel; a newer county roll provides its
    # own use_subgroup_code (a different column, so it must not collide) and
    # a value column with full coverage that should win as the more recent
    # source.
    matches = [
        {
            'recipe_id': 'nconemap',
            'layer': None,
            'key': 'parcel_id_local',
            'aggregation_function': None,
        },
        {
            'recipe_id': 'nhcgov',
            'layer': None,
            'key': 'parcel_id_local',
            'aggregation_function': None,
        },
    ]
    monkeypatch.setattr(links, '_discover_link_sources', lambda *a, **k: matches)
    monkeypatch.setattr(links, '_apply_remap_csvs', lambda state, recipe_id: state)

    refs = {
        'nconemap': pd.DataFrame(
            {
                'parcel_id_local': ['A', 'B'],
                'use_subgroup': ['single family', 'multi family'],
                'value': [100.0, 200.0],
            }
        ),
        'nhcgov': pd.DataFrame(
            {
                'parcel_id_local': ['A', 'B'],
                'value': [150.0, 250.0],
            }
        ),
    }
    monkeypatch.setattr(
        links, 'get_entities', lambda recipe_id, admin_id, layer=None: refs[recipe_id]
    )

    spine = pd.DataFrame({'parcel_id_local': ['A', 'B']})
    state = _state('US-NC-NHA', spine=spine)
    state = links.link_by_id(
        state,
        auto_discover=True,
        entity_type='parcel',
        columns=['use_subgroup', 'value'],
    )

    # nhcgov never offered use_subgroup, so nconemap's labels survive untouched.
    assert state.spine['use_subgroup'].tolist() == ['single family', 'multi family']
    # nhcgov is the more recently-joined match and covers all rows, so it wins.
    assert state.spine['value'].tolist() == [150.0, 250.0]


def test_link_by_id_auto_discover_track_provenance_records_winning_source(
    monkeypatch,
):
    # Same setup as the majority-source test above, but with track_provenance
    # requested for 'value': every cell it wrote is stamped with the source
    # that actually supplied it.
    matches = [
        {
            'recipe_id': 'nconemap',
            'layer': None,
            'key': 'parcel_id_local',
            'aggregation_function': None,
        },
        {
            'recipe_id': 'nhcgov',
            'layer': None,
            'key': 'parcel_id_local',
            'aggregation_function': None,
        },
    ]
    monkeypatch.setattr(links, '_discover_link_sources', lambda *a, **k: matches)
    monkeypatch.setattr(links, '_apply_remap_csvs', lambda state, recipe_id: state)

    refs = {
        'nconemap': pd.DataFrame(
            {'parcel_id_local': ['A', 'B'], 'value': [100.0, 200.0]}
        ),
        'nhcgov': pd.DataFrame(
            {'parcel_id_local': ['A', 'B'], 'value': [150.0, 250.0]}
        ),
    }
    monkeypatch.setattr(
        links, 'get_entities', lambda recipe_id, admin_id, layer=None: refs[recipe_id]
    )

    spine = pd.DataFrame({'parcel_id_local': ['A', 'B']})
    state = _state('US-NC-NHA', spine=spine)
    state = links.link_by_id(
        state,
        auto_discover=True,
        entity_type='parcel',
        columns=['value'],
        track_provenance=['value'],
    )

    assert state.spine['value'].tolist() == [150.0, 250.0]
    assert state.spine['value_source'].tolist() == ['nhcgov', 'nhcgov']


def test_link_by_id_auto_discover_skips_self_join_keep_columns(monkeypatch):
    # A standalone roll that is also the spine's own geometry source (e.g. a
    # single-source county with no separate assessor roll) must not
    # re-derive its resolve_spine keep_columns (use_subgroup here) by
    # aggregating across every spine row sharing a non-unique
    # parcel_id_local: two physically distinct parcels ('mh-park' and
    # 'canalfront') collide on the same local id 'DUP', so an aggregated
    # 'first' value would silently overwrite mh-park's own correct label
    # with canalfront's. land_value is not a keep_column, so it still gets
    # the join (summed across the colliding key, same as any other
    # ambiguous aggregate — that risk is inherent to a non-unique key and
    # out of scope here; only keep_columns, which already have a correct
    # per-geometry value, are protected from it). The spine's
    # geometry_source is 'nconemap' (this same source), so the row-level
    # mask path is exercised, not the no-geometry_source fallback.
    matches = [
        {
            'recipe_id': 'nconemap',
            'layer': None,
            'key': 'parcel_id_local',
            'aggregation_function': None,
        },
    ]
    monkeypatch.setattr(links, '_discover_link_sources', lambda *a, **k: matches)
    monkeypatch.setattr(links, '_apply_remap_csvs', lambda state, recipe_id: state)

    ref = pd.DataFrame(
        {
            'parcel_id_local': ['DUP', 'DUP'],
            'use_subgroup': ['MOBILE HOME PARK', 'CANALFRONT'],
            'land_value': [100.0, 200.0],
        },
        index=pd.Index(['mh-park', 'canalfront'], name='parcel_id'),
    )
    monkeypatch.setattr(
        links, 'get_entities', lambda recipe_id, admin_id, layer=None: ref
    )

    spine = pd.DataFrame(
        {
            'parcel_id_local': ['DUP'],
            'use_subgroup': ['MOBILE HOME PARK'],
            'geometry_source': ['nconemap'],
        },
        index=pd.Index(['mh-park'], name='parcel_id'),
    )
    state = _state('US-NC-AR', spine=spine)
    state.metadata['spine_source_recipe_ids'] = {'nconemap'}
    state.metadata['spine_keep_columns'] = {'use_subgroup'}

    state = links.link_by_id(
        state,
        auto_discover=True,
        entity_type='parcel',
        columns=['use_subgroup', 'land_value'],
    )

    assert state.spine['use_subgroup'].tolist() == ['MOBILE HOME PARK']
    assert state.spine['land_value'].tolist() == [300.0]


def test_link_by_id_auto_discover_skips_self_join_for_year_built(monkeypatch):
    # Regression test for a real bug: a footprint on parcel 'old-house'
    # (year_built=1964) ended up with year_built=1865.21... in a county
    # where parcel_id_local was heavily duplicated. Root cause: year_built
    # was not in resolve_spine's keep_columns, so link_by_id's auto_discover
    # self-join aggregated it (mean, per the attribute registry) across
    # every parcel sharing one non-unique parcel_id_local, pooling
    # 'old-house' (1964) with an unrelated 'new-house' (1998) into a
    # meaningless mean and broadcasting it back onto both. Once year_built
    # is added to keep_columns, it's protected the same way use_subgroup
    # already is. geometry_source is 'nconemap' (this same source), so the
    # row-level mask path is exercised, not the no-geometry_source fallback.
    matches = [
        {
            'recipe_id': 'nconemap',
            'layer': None,
            'key': 'parcel_id_local',
            'aggregation_function': None,
        },
    ]
    monkeypatch.setattr(links, '_discover_link_sources', lambda *a, **k: matches)
    monkeypatch.setattr(links, '_apply_remap_csvs', lambda state, recipe_id: state)

    ref = pd.DataFrame(
        {
            'parcel_id_local': ['DUP', 'DUP'],
            'year_built': [1964.0, 1998.0],
        },
        index=pd.Index(['old-house', 'new-house'], name='parcel_id'),
    )
    monkeypatch.setattr(
        links, 'get_entities', lambda recipe_id, admin_id, layer=None: ref
    )

    spine = pd.DataFrame(
        {
            'parcel_id_local': ['DUP'],
            'year_built': [1964.0],
            'geometry_source': ['nconemap'],
        },
        index=pd.Index(['old-house'], name='parcel_id'),
    )
    state = _state('US-NC-AR', spine=spine)
    state.metadata['spine_source_recipe_ids'] = {'nconemap'}
    state.metadata['spine_keep_columns'] = {'year_built'}

    state = links.link_by_id(
        state,
        auto_discover=True,
        entity_type='parcel',
        columns=['year_built'],
    )

    # Protected: stays 1964, not pooled to mean(1964, 1998) = 1981.
    assert state.spine['year_built'].tolist() == [1964.0]


def test_link_by_id_auto_discover_keep_column_fallback_fills_gap_from_other_source(
    monkeypatch,
):
    # The actual bug this row-level mask fixes: row A's geometry came from
    # 'bladenco' itself, so bladenco's own value there is already correct
    # and must not change (protected). Row B's geometry came from a
    # DIFFERENT source ('nconemap', not discovered as a match here -- it
    # left this parcel's year_built null), so bladenco -- despite also
    # being one of the spine's overall geometry sources -- is free to fill
    # row B's gap from its own reference data for that same key, since
    # row B isn't one of bladenco's own winning-geometry rows.
    matches = [
        {
            'recipe_id': 'bladenco',
            'layer': None,
            'key': 'parcel_id_local',
            'aggregation_function': None,
        },
    ]
    monkeypatch.setattr(links, '_discover_link_sources', lambda *a, **k: matches)
    monkeypatch.setattr(links, '_apply_remap_csvs', lambda state, recipe_id: state)

    ref = pd.DataFrame({'parcel_id_local': ['A', 'B'], 'year_built': [1998.0, 2005.0]})
    monkeypatch.setattr(
        links, 'get_entities', lambda recipe_id, admin_id, layer=None: ref
    )

    spine = pd.DataFrame(
        {
            'parcel_id_local': ['A', 'B'],
            'year_built': [1998.0, None],
            'geometry_source': ['bladenco', 'nconemap'],
        },
        index=pd.Index(['row-a', 'row-b'], name='parcel_id'),
    )
    state = _state('US-NC-BL', spine=spine)
    state.metadata['spine_source_recipe_ids'] = {'bladenco', 'nconemap'}
    state.metadata['spine_keep_columns'] = {'year_built'}

    state = links.link_by_id(
        state,
        auto_discover=True,
        entity_type='parcel',
        columns=['year_built'],
    )

    assert state.spine['year_built'].tolist() == [1998.0, 2005.0]


def test_link_by_id_auto_discover_no_geometry_source_falls_back_to_column_drop(
    monkeypatch,
):
    # A union_spine_sources-built (non-spatial) spine has no geometry_source
    # to key row-level protection on -- fall back to dropping the column
    # from this match entirely, the original coarser guard, rather than
    # risk the pooled-duplicate-key corruption the guard exists to prevent.
    matches = [
        {
            'recipe_id': 'nconemap',
            'layer': None,
            'key': 'parcel_id_local',
            'aggregation_function': None,
        },
    ]
    monkeypatch.setattr(links, '_discover_link_sources', lambda *a, **k: matches)
    monkeypatch.setattr(links, '_apply_remap_csvs', lambda state, recipe_id: state)

    ref = pd.DataFrame(
        {
            'parcel_id_local': ['DUP', 'DUP'],
            'year_built': [1964.0, 1998.0],
        },
        index=pd.Index(['old-house', 'new-house'], name='parcel_id'),
    )
    monkeypatch.setattr(
        links, 'get_entities', lambda recipe_id, admin_id, layer=None: ref
    )

    spine = pd.DataFrame(
        {
            'parcel_id_local': ['DUP'],
            'year_built': [1964.0],
        },
        index=pd.Index(['old-house'], name='parcel_id'),
    )
    state = _state('US-NC-AR', spine=spine)
    state.metadata['spine_source_recipe_ids'] = {'nconemap'}
    state.metadata['spine_keep_columns'] = {'year_built'}

    state = links.link_by_id(
        state,
        auto_discover=True,
        entity_type='parcel',
        columns=['year_built'],
    )

    assert state.spine['year_built'].tolist() == [1964.0]


def test_link_by_id_auto_discover_joins_every_match(monkeypatch):
    matches = [
        {
            'recipe_id': 'source-a',
            'layer': None,
            'key': 'parcel_id_local',
            'aggregation_function': None,
        },
        {
            'recipe_id': 'source-b',
            'layer': 'property',
            'key': 'parcel_id_admin2',
            'aggregation_function': None,
        },
    ]
    monkeypatch.setattr(links, '_discover_link_sources', lambda *a, **k: matches)
    monkeypatch.setattr(links, '_apply_remap_csvs', lambda state, recipe_id: state)

    refs = {
        'source-a': pd.DataFrame(
            {'parcel_id_local': ['A', 'B'], 'land_value': [10.0, 20.0]}
        ),
        'source-b': pd.DataFrame(
            {'parcel_id_admin2': ['A', 'A'], 'improvement_value': [1.0, 2.0]}
        ),
    }

    def _fake_get_entities(recipe_id, admin_id, layer=None):
        return refs[recipe_id]

    monkeypatch.setattr(links, 'get_entities', _fake_get_entities)

    spine = pd.DataFrame(
        {'parcel_id_local': ['A', 'B'], 'parcel_id_admin2': ['A', 'C']}
    )
    state = _state('US-MA-SOM', spine=spine)
    state = links.link_by_id(state, auto_discover=True, entity_type='parcel')

    assert state.spine['land_value'].tolist() == [10.0, 20.0]
    # B's parcel_id_admin2 is 'C', which doesn't match source-b's 'A' rows.
    assert state.spine['improvement_value'].iloc[0] == 3.0  # 1.0 + 2.0 summed
    assert pd.isna(state.spine['improvement_value'].iloc[1])


def test_link_by_id_auto_discover_match_own_aggregation_function_is_scoped(
    monkeypatch,
):
    # A match's own declared aggregation_function (e.g. the improvement-
    # detail sibling's area_sqft: sum, year_built: min) must apply to that
    # match's columns only -- a sibling match with no such declaration keeps
    # the registry default ('mean' for both here) even though both matches
    # attach the same two column names.
    matches = [
        {
            'recipe_id': 'source-a',
            'layer': None,
            'key': 'parcel_id_local',
            'aggregation_function': None,
        },
        {
            'recipe_id': 'source-b',
            'layer': None,
            'key': 'parcel_id_local',
            'aggregation_function': {'area_sqft': 'sum', 'year_built': 'min'},
        },
    ]
    monkeypatch.setattr(links, '_discover_link_sources', lambda *a, **k: matches)
    monkeypatch.setattr(links, '_apply_remap_csvs', lambda state, recipe_id: state)

    refs = {
        # Registry default for both columns is 'mean': (10+20)/2 = 15.
        'source-a': pd.DataFrame(
            {
                'parcel_id_local': ['A', 'A'],
                'area_sqft': [10.0, 20.0],
                'year_built': [2000.0, 2010.0],
            }
        ),
        # Own override: area_sqft sums to 30, year_built takes the min 2000.
        'source-b': pd.DataFrame(
            {
                'parcel_id_local': ['B', 'B'],
                'area_sqft': [10.0, 20.0],
                'year_built': [2000.0, 2010.0],
            }
        ),
    }
    monkeypatch.setattr(
        links, 'get_entities', lambda recipe_id, admin_id, layer=None: refs[recipe_id]
    )

    spine = pd.DataFrame({'parcel_id_local': ['A', 'B']})
    state = _state('US-TX-VIC', spine=spine)
    state = links.link_by_id(
        state,
        auto_discover=True,
        entity_type='property',
        columns=['area_sqft', 'year_built'],
    )

    assert state.spine.set_index('parcel_id_local').loc['A', 'area_sqft'] == 15.0
    assert state.spine.set_index('parcel_id_local').loc['A', 'year_built'] == 2005.0
    assert state.spine.set_index('parcel_id_local').loc['B', 'area_sqft'] == 30.0
    assert state.spine.set_index('parcel_id_local').loc['B', 'year_built'] == 2000.0


def test_apply_remap_csvs_applies_matching_crosswalk_and_infers_key_length():
    spine = pd.DataFrame(
        {'use_group_code': ['101', '1010', '102', '999', None]},
        index=pd.Index(['a', 'b', 'c', 'd', 'e'], name='parcel_id'),
    )
    state = HarmonizeState(
        recipe={}, admin_id=None, verbose=False, timer=None, spine=spine
    )
    state = links._apply_remap_csvs(state, 'US-MA_parcel-massgis-2025')
    out = state.spine

    expected = ['residential', 'residential', 'residential']
    assert out['use_group'].tolist()[:3] == expected
    assert pd.isna(out['use_group'].iloc[3])
    assert pd.isna(out['use_group'].iloc[4])
    # '1010' (4 digits) truncates to the crosswalk's 3-digit key and matches.
    assert out['use_subgroup'].iloc[1] == 'single family'


def test_apply_remap_csvs_noop_without_matching_column():
    spine = pd.DataFrame({'other_column': ['x']})
    state = HarmonizeState(
        recipe={}, admin_id=None, verbose=False, timer=None, spine=spine
    )
    state = links._apply_remap_csvs(state, 'US-MA_parcel-massgis-2025')
    assert 'use_group' not in state.spine.columns
