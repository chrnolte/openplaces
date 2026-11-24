"""`join_temporal_snapshot` with a series enrichment as its reference.

The long table is fabricated and written, with the footer a series run
writes, to the output path of a real enrich recipe under a temporary
data root. The Florida roll join keeps its own path, pinned by
`test_join_temporal_snapshot.py` and by the last test here.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.io import save_parquet, to_parquet
from openplaces.io.curator import CurateState
from openplaces.io.curator.transactions import join_temporal_snapshot
from openplaces.io.tables import SERIES_METADATA_KEY
from openplaces.recipe import get_output_path, get_recipe_by_id

SERIES_ID = 'CO_parcel_landcover-forest-annual-2026'
UNIT = AdminId('CO-AN-ZZ')


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    dirs = dict(cfg.config['directories'])
    dirs['data_root'] = tmp_path
    for name in ('core', 'external', 'raw', 'cache', 'out', 'share'):
        dirs[name] = tmp_path / 'data' / name
    dirs['heap'] = tmp_path / 'data/cache/_heap'
    dirs['logs'] = tmp_path / 'data/cache/_logs'
    monkeypatch.setitem(cfg.config, 'directories', dirs)
    return tmp_path


def _write_series(pairs, rule='before'):
    index = pd.MultiIndex.from_tuples(
        [(e, p) for e, p, _ in pairs], names=['parcel_id', 'period']
    )
    table = pd.DataFrame({'forest_share': [v for *_, v in pairs]}, index=index)
    meta = {
        'requests': [{'source': 'test', 'rows': len(pairs)}],
        'periods': sorted({p for _, p, _ in pairs}),
        'layers': {
            'forest_share': {
                'periods': [2000, 2024],
                'rule': rule,
                'period_length_years': 1,
            }
        },
    }
    path = get_output_path(get_recipe_by_id(SERIES_ID), UNIT)
    save_parquet(table, path, file_metadata={SERIES_METADATA_KEY: json.dumps(meta)})


def _state(df):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=UNIT,
        verbose=False,
        timer=None,
        curated=df,
    )


def _join(df, **kwargs):
    return join_temporal_snapshot(
        _state(df),
        recipe_id=SERIES_ID,
        join_key='parcel_id',
        date_column='sale_year',
        vintage_column='period',
        columns=['period', 'forest_share'],
        prefix='at_sale_',
        match_type_column='forest_match_type',
        **kwargs,
    ).curated


def test_a_sale_takes_the_period_before_it_and_nothing_else(data_root):
    _write_series([('p1', 2015, 0.6), ('p1', 2016, 0.5), ('p2', 2009, 0.9)])
    tx = pd.DataFrame(
        {
            'parcel_id': ['p1', 'p1', 'p2', 'p2', 'p3'],
            'sale_year': [2016, 2017, 2011, 1990, None],
        },
        index=pd.Index(['t1', 't2', 't3', 't4', 't5'], name='transaction_id'),
    )
    out = _join(tx, mark_unmatched='not_requested')
    assert out.loc['t1', 'at_sale_period'] == 2015
    assert out.loc['t1', 'at_sale_forest_share'] == 0.6
    assert out.loc['t2', 'at_sale_period'] == 2016
    # p2 sold in 2011 needs 2010, which no request asked for: it does
    # not borrow 2009.
    assert pd.isna(out.loc['t3', 'at_sale_forest_share'])
    assert out['forest_match_type'].to_dict() == {
        't1': 'exact',
        't2': 'exact',
        't3': 'not_requested',
        't4': 'no_period',
        't5': 'no_sale_date',
    }


def test_the_rule_can_be_overridden_and_now_ignores_the_date(data_root):
    _write_series([('p1', 2016, 0.5), ('p1', 2024, 0.1)])
    tx = pd.DataFrame(
        {'parcel_id': ['p1'], 'sale_year': [2016]},
        index=pd.Index(['t1'], name='transaction_id'),
    )
    assert _join(tx, period_rule='at_or_before').loc['t1', 'at_sale_period'] == 2016
    assert _join(tx, period_rule='now').loc['t1', 'at_sale_period'] == 2024


def test_the_roll_join_refuses_series_parameters(data_root):
    recipe = get_recipe_by_id('US-FL_property-fldor-2026')
    path = get_output_path(recipe, AdminId('US-FL-PO'), partition_id='all')
    to_parquet(
        pd.DataFrame(
            {'parcel_id_assessor': ['A', 'A'], 'tax_year': [2018, 2020], 'v': [1, 2]}
        ),
        path,
    )
    tx = pd.DataFrame({'parcel_id_assessor': ['A'], 'sale_year': [2019]})
    state = CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US-FL-PO'),
        verbose=False,
        timer=None,
        curated=tx,
    )
    kwargs = dict(
        recipe_id='US-FL_property-fldor-2026',
        join_key='parcel_id_assessor',
        date_column='sale_year',
        vintage_column='tax_year',
        columns=['tax_year', 'v'],
        match_type_column='m',
    )
    with pytest.raises(ValueError, match='series enrichment only'):
        join_temporal_snapshot(state, period_rule='before', **kwargs)
    # The as-of pass is unchanged: 2019 takes the 2018 roll.
    out = join_temporal_snapshot(state, **kwargs).curated
    assert out.loc[0, 'tax_year'] == 2018
    assert out.loc[0, 'm'] == 'backward_fallback'
