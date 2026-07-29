"""Tests for the harmonizer `link_by_id` step (id-based, non-spatial linking).

Joins a reference entity to the spine on a precomputed key: `attributes` mode
attaches columns (1:1); `count` mode tracks which parcels have been transacted
(1:many -> n_transactions / is_transacted); `aggregate` mode reduces a 1:many
reference onto the spine (e.g. MassGIS condominium assessor records stacked on
one parcel polygon) by the attribute-registry aggregation per column.
"""

import pandas as pd
import pytest

import openplaces.io.harmonizer.links as links
from openplaces.core.schema import Entity
from openplaces.io.harmonizer import HarmonizeState


def _state(spine, entity_type=None):
    recipe = {'entity': Entity(entity_type)} if entity_type else {}
    return HarmonizeState(
        recipe=recipe, admin_id='US-NC-NE', verbose=False, timer=None, spine=spine
    )


def test_attributes_mode_joins_columns(monkeypatch):
    spine = pd.DataFrame({'parcel_id_local': ['A', 'B', 'C']})
    ref = pd.DataFrame({'parcel_id_local': ['A', 'B'], 'land_value': [100, 200]})
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = links.link_by_id(
        _state(spine), 'ref', mode='attributes', columns=['land_value']
    )

    assert state.spine['land_value'].iloc[0] == 100
    assert state.spine['land_value'].iloc[1] == 200
    assert pd.isna(state.spine['land_value'].iloc[2])


def test_count_mode_flags_transacted_parcels(monkeypatch):
    spine = pd.DataFrame({'parcel_id_local': ['A', 'B', 'C']})
    ref = pd.DataFrame({'parcel_id_local': ['A', 'A', 'B']})  # A x2, B x1, C none
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = links.link_by_id(_state(spine), 'tx', mode='count')

    assert state.spine['n_transactions'].tolist() == [2, 1, 0]
    assert state.spine['is_transacted'].tolist() == [True, True, False]


def test_missing_spine_key_skips(monkeypatch):
    spine = pd.DataFrame({'other': [1, 2]})
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: pd.DataFrame())
    state = links.link_by_id(_state(spine), 'ref', mode='count')
    assert 'n_transactions' not in state.spine.columns


def test_aggregate_mode_reduces_many_to_one(monkeypatch):
    # Two condominium records on parcel A, one on B, none on C.
    spine = pd.DataFrame({'parcel_id_admin2': ['A', 'B', 'C']})
    ref = pd.DataFrame(
        {
            'parcel_id_admin2': ['A', 'A', 'B'],
            'land_value': [100.0, 100.0, 50.0],
            'n_dwellings': [1.0, 1.0, 1.0],
            'usecode': ['102', '102', '101'],
        }
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = links.link_by_id(
        _state(spine),
        'property',
        mode='aggregate',
        spine_key='parcel_id_admin2',
        ref_key='parcel_id_admin2',
        columns=['land_value', 'n_dwellings', 'usecode'],
    )
    out = state.spine

    # Registry aggregations: land_value and n_dwellings sum across the records.
    assert out['land_value'].iloc[0] == 200.0
    assert out['land_value'].iloc[1] == 50.0
    assert out['n_dwellings'].iloc[0] == 2.0  # two condo units summed
    assert pd.isna(out['land_value'].iloc[2])  # C unmatched
    # usecode has no registry rule -> first value per parcel.
    assert out['usecode'].iloc[0] == '102'
    # Per-key record count (default name when count_as left at its count default).
    assert out['n_records_per_key'].tolist() == [2, 1, 0]


def test_aggregate_mode_custom_count_name(monkeypatch):
    spine = pd.DataFrame({'parcel_id_admin2': ['A', 'B']})
    ref = pd.DataFrame(
        {'parcel_id_admin2': ['A', 'A', 'B'], 'land_value': [10.0, 20.0, 5.0]}
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = links.link_by_id(
        _state(spine),
        'property',
        mode='aggregate',
        spine_key='parcel_id_admin2',
        ref_key='parcel_id_admin2',
        columns=['land_value'],
        count_as='n_properties_per_parcel',
    )
    assert state.spine['n_properties_per_parcel'].tolist() == [2, 1]
    assert state.spine['land_value'].iloc[0] == 30.0


def test_unknown_mode_raises(monkeypatch):
    spine = pd.DataFrame({'parcel_id_local': ['A']})
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: spine.copy())
    with pytest.raises(ValueError, match='unknown mode'):
        links.link_by_id(_state(spine), 'ref', mode='bogus')


def test_missing_reference_data_skips(monkeypatch):
    # A source-specific roll listed in a shared pipeline but absent for this admin.
    spine = pd.DataFrame({'parcel_id_admin2': ['A', 'B']})

    def _raise(*a, **k):
        raise FileNotFoundError('no such entity for admin')

    monkeypatch.setattr(links, 'get_entities', _raise)
    state = links.link_by_id(
        _state(spine),
        'US-MA_property-massgis-2025',
        mode='aggregate',
        spine_key='parcel_id_admin2',
        ref_key='parcel_id_admin2',
        columns=['land_value'],
    )
    assert 'land_value' not in state.spine.columns


def test_duplicate_spine_key_warns_but_does_not_change_result(monkeypatch):
    # Two spine rows share 'DUP' -- a non-unique join key, e.g. parcel_id_local
    # colliding across genuinely distinct parcels. The join still proceeds
    # (broadcasting the same matched value to both), but the risk must be
    # surfaced, not silent.
    spine = pd.DataFrame({'parcel_id_local': ['DUP', 'DUP', 'C']})
    ref = pd.DataFrame({'parcel_id_local': ['DUP', 'C'], 'land_value': [100, 300]})
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    with pytest.warns(UserWarning, match="'parcel_id_local' is not unique"):
        state = links.link_by_id(
            _state(spine), 'ref', mode='attributes', columns=['land_value']
        )

    assert state.spine['land_value'].tolist() == [100, 100, 300]


def test_duplicate_foreign_spine_key_does_not_warn(recwarn, monkeypatch):
    # A transaction spine's parcel_id_local is a foreign key borrowed
    # from parcel -- many transactions can share one parcel (sold more
    # than once), so this must not warn.
    spine = pd.DataFrame({'parcel_id_local': ['DUP', 'DUP', 'C']})
    ref = pd.DataFrame({'parcel_id_local': ['DUP', 'C'], 'land_value': [100, 300]})
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    links.link_by_id(
        _state(spine, entity_type='transaction'),
        'ref',
        mode='attributes',
        columns=['land_value'],
    )

    assert not any('is not unique' in str(w.message) for w in recwarn.list)


def test_duplicate_own_spine_key_still_warns(monkeypatch):
    # A parcel spine's parcel_id_local is its own identity key -- a
    # duplicate here means two parcel-spine rows collide on one key, a
    # real anomaly, so this must still warn.
    spine = pd.DataFrame({'parcel_id_local': ['DUP', 'DUP', 'C']})
    ref = pd.DataFrame({'parcel_id_local': ['DUP', 'C'], 'land_value': [100, 300]})
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    with pytest.warns(UserWarning, match="'parcel_id_local' is not unique"):
        links.link_by_id(
            _state(spine, entity_type='parcel'),
            'ref',
            mode='attributes',
            columns=['land_value'],
        )


def test_duplicate_attributes_ref_key_warns(monkeypatch):
    # 'attributes' mode keeps an arbitrary first match per key -- no
    # aggregation -- so a duplicated ref_key must be flagged.
    spine = pd.DataFrame({'parcel_id_local': ['A']})
    ref = pd.DataFrame({'parcel_id_local': ['A', 'A'], 'land_value': [100, 200]})
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    with pytest.warns(UserWarning, match='reference key'):
        state = links.link_by_id(
            _state(spine), 'ref', mode='attributes', columns=['land_value']
        )

    # Behavior unchanged: still keeps the first match (100), just now warned.
    assert state.spine['land_value'].iloc[0] == 100


def test_unique_key_emits_no_duplicate_warning(recwarn, monkeypatch):
    spine = pd.DataFrame({'parcel_id_local': ['A', 'B']})
    ref = pd.DataFrame({'parcel_id_local': ['A', 'B'], 'land_value': [100, 200]})
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    links.link_by_id(_state(spine), 'ref', mode='attributes', columns=['land_value'])

    assert not any('is not unique' in str(w.message) for w in recwarn.list)


def test_attributes_mode_dict_columns_renames_on_write(monkeypatch):
    spine = pd.DataFrame({'parcel_id_local': ['A', 'B']})
    ref = pd.DataFrame({'parcel_id_local': ['A', 'B'], 'price': [100, 200]})
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = links.link_by_id(
        _state(spine), 'ref', mode='attributes', columns={'price': 'last_sale_price'}
    )

    assert 'price' not in state.spine.columns
    assert state.spine['last_sale_price'].tolist() == [100, 200]


def test_aggregate_mode_dict_columns_renames_and_uses_output_name_for_registry(
    monkeypatch,
):
    # recorded_date has no registry rule under that name; last_sale_date does
    # (aggregation='first') -- the lookup must use the *output* name.
    spine = pd.DataFrame({'parcel_id_local': ['A']})
    ref = pd.DataFrame(
        {'parcel_id_local': ['A', 'A'], 'recorded_date': ['2020-01-01', '2021-06-01']}
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = links.link_by_id(
        _state(spine),
        'tx',
        mode='aggregate',
        columns={'recorded_date': 'last_sale_date'},
    )

    assert state.spine['last_sale_date'].iloc[0] == '2020-01-01'  # first row in ref


def test_aggregate_mode_count_as_n_transactions_now_honored(monkeypatch):
    # Regression: aggregate mode used to special-case the literal string
    # 'n_transactions' and silently rename it to 'n_records_per_key' even when
    # explicitly requested -- an explicit count_as must now be honored as-is.
    spine = pd.DataFrame({'parcel_id_local': ['A', 'B']})
    ref = pd.DataFrame({'parcel_id_local': ['A', 'A', 'B']})
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = links.link_by_id(
        _state(spine), 'tx', mode='aggregate', count_as='n_transactions'
    )

    assert state.spine['n_transactions'].tolist() == [2, 1]
    assert 'n_records_per_key' not in state.spine.columns


def test_count_mode_flag_as_none_skips_flag_column(monkeypatch):
    spine = pd.DataFrame({'parcel_id_local': ['A', 'B']})
    ref = pd.DataFrame({'parcel_id_local': ['A']})
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = links.link_by_id(_state(spine), 'tx', mode='count', flag_as=None)

    assert state.spine['n_transactions'].tolist() == [1, 0]
    assert 'is_transacted' not in state.spine.columns


def test_ref_sort_by_picks_most_recent_for_first_aggregation(monkeypatch):
    # Raw reference rows are not chronologically ordered; ref_sort_by must
    # sort before the registry's 'first' aggregation picks a row per key.
    spine = pd.DataFrame({'parcel_id_local': ['A']})
    ref = pd.DataFrame(
        {
            'parcel_id_local': ['A', 'A', 'A'],
            'recorded_date': ['2022-06-01', '2020-01-01', '2024-03-15'],
            'price': [200, 100, 400],
        }
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = links.link_by_id(
        _state(spine),
        'tx',
        mode='aggregate',
        columns={'price': 'last_sale_price', 'recorded_date': 'last_sale_date'},
        ref_sort_by='recorded_date',
        ref_sort_ascending=False,
    )

    assert state.spine['last_sale_date'].iloc[0] == '2024-03-15'
    assert state.spine['last_sale_price'].iloc[0] == 400


def test_aggregate_mode_duplicate_spine_key_still_aggregates_and_warns(monkeypatch):
    # mode='aggregate' already aggregates the *reference* side correctly (see
    # test_aggregate_mode_reduces_many_to_one); this confirms a duplicate on
    # the *spine* side is still surfaced even though the result (the same
    # aggregated value broadcast to both spine rows) is unchanged.
    spine = pd.DataFrame({'parcel_id_admin2': ['A', 'A', 'B']})
    ref = pd.DataFrame({'parcel_id_admin2': ['A', 'B'], 'land_value': [100.0, 50.0]})
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    with pytest.warns(UserWarning, match="'parcel_id_admin2' is not unique"):
        state = links.link_by_id(
            _state(spine),
            'property',
            mode='aggregate',
            spine_key='parcel_id_admin2',
            ref_key='parcel_id_admin2',
            columns=['land_value'],
        )

    assert state.spine['land_value'].tolist() == [100.0, 100.0, 50.0]
