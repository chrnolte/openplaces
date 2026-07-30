"""Tests for the merge_enrichments curate step."""

import pandas as pd
import pytest

from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.io import to_parquet
from openplaces.io.curator import CurateState
from openplaces.io.curator.evidence import merge_enrichments
from openplaces.recipe import get_output_path, get_recipe_by_id

SPINE = 'US_footprint-spine-2026'
ROOF_SHAPE = 'US_footprint_built-roof-shape-brails-2026'
N_STORIES = 'US_footprint_built-n-stories-brails-2026'
COUNTY = 'US-FL-LK'


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


def _write_roof_shape_evidence():
    evidence = pd.DataFrame(
        {'roof_shape_brails': ['Gable', 'Hip']},
        index=pd.Index(['f1', 'f2'], name='footprint_id'),
    )
    ref_recipe = get_recipe_by_id(ROOF_SHAPE)
    path = get_output_path(ref_recipe, AdminId(COUNTY), entity_recipe_id=SPINE)
    to_parquet(evidence, path)


def _state(verbose=False):
    curated = pd.DataFrame(
        {'roof_shape': [None, None], 'n_stories_brails': [None, None]},
        index=pd.Index(['f1', 'f2'], name='footprint_id'),
    )
    return CurateState(
        recipe={'recipe_id': 'US_footprint-cheer-2026'},
        entity_recipe=get_recipe_by_id(SPINE),
        admin_id=AdminId(COUNTY),
        verbose=verbose,
        timer=None,
        curated=curated,
    )


def test_missing_evidence_file_skipped_present_still_merges(data_root, capsys):
    # roof-shape evidence exists (e.g. Google Satellite ran); n-stories
    # evidence does not (e.g. --no_streetview skipped Street View + its
    # enrichment this pass).
    _write_roof_shape_evidence()
    state = merge_enrichments(
        _state(verbose=True),
        recipes=[
            {'recipe_id': ROOF_SHAPE, 'columns': {'roof_shape_brails': 'roof_shape'}},
            {
                'recipe_id': N_STORIES,
                'columns': {'n_stories_brails': 'n_stories_brails'},
            },
        ],
    )
    assert list(state.curated['roof_shape']) == ['Gable', 'Hip']
    assert state.curated['n_stories_brails'].isna().all()
    assert 'skipping' in capsys.readouterr().out


def test_present_evidence_missing_column_still_raises(data_root):
    _write_roof_shape_evidence()
    with pytest.raises(ValueError, match='missing from'):
        merge_enrichments(
            _state(),
            recipes=[
                {
                    'recipe_id': ROOF_SHAPE,
                    'columns': {'not_a_real_column': 'roof_shape'},
                }
            ],
        )


def test_default_does_not_create_source_column(data_root):
    # record_source defaults to False: a bulk, single-source enrichment
    # merge (the common case, e.g. dozens of placeslab-fmv2026 columns)
    # should never add a `_source` sidecar unless explicitly asked for.
    _write_roof_shape_evidence()
    state = merge_enrichments(
        _state(),
        recipes=[
            {'recipe_id': ROOF_SHAPE, 'columns': {'roof_shape_brails': 'roof_shape'}}
        ],
    )
    assert list(state.curated['roof_shape']) == ['Gable', 'Hip']
    assert 'roof_shape_source' not in state.curated.columns


def test_record_source_true_creates_sidecar(data_root):
    _write_roof_shape_evidence()
    state = merge_enrichments(
        _state(),
        recipes=[
            {
                'recipe_id': ROOF_SHAPE,
                'record_source': True,
                'columns': {'roof_shape_brails': 'roof_shape'},
            }
        ],
    )
    assert 'roof_shape_source' in state.curated.columns
    assert state.curated['roof_shape_source'].notna().all()


def _write_n_stories_evidence(values):
    evidence = pd.DataFrame(
        {'n_stories_brails': values},
        index=pd.Index(['f1', 'f2'], name='footprint_id'),
    )
    ref_recipe = get_recipe_by_id(N_STORIES)
    path = get_output_path(ref_recipe, AdminId(COUNTY), entity_recipe_id=SPINE)
    to_parquet(evidence, path)


def test_later_spec_only_fills_rows_earlier_spec_left_missing(data_root):
    # Two recipe specs mapped onto the SAME canonical column, simulating a
    # priority merge: the first-declared spec's value wins wherever it has
    # one; a later spec only fills rows still missing. This must survive
    # the batched (pending-dict) rewrite, since nothing here reads back
    # from `curated` between the two specs -- only the earlier spec's own
    # in-memory pending value does.
    evidence_a = pd.DataFrame(
        {'roof_shape_brails': ['Gable', None]},
        index=pd.Index(['f1', 'f2'], name='footprint_id'),
    )
    to_parquet(
        evidence_a,
        get_output_path(
            get_recipe_by_id(ROOF_SHAPE), AdminId(COUNTY), entity_recipe_id=SPINE
        ),
    )
    _write_n_stories_evidence(['Flat', 'Hip'])

    state = merge_enrichments(
        _state(),
        recipes=[
            {'recipe_id': ROOF_SHAPE, 'columns': {'roof_shape_brails': 'roof_shape'}},
            {'recipe_id': N_STORIES, 'columns': {'n_stories_brails': 'roof_shape'}},
        ],
    )
    # f1 keeps the first spec's value; f2 (missing from the first) is
    # filled by the second.
    assert list(state.curated['roof_shape']) == ['Gable', 'Hip']


def test_pre_existing_column_and_source_are_overwritten_not_duplicated(data_root):
    _write_roof_shape_evidence()
    state = _state()
    state.curated['roof_shape'] = [None, 'Existing Hip']
    state.curated['roof_shape_source'] = [None, 'legacy']

    state = merge_enrichments(
        state,
        recipes=[
            {
                'recipe_id': ROOF_SHAPE,
                'record_source': True,
                'columns': {'roof_shape_brails': 'roof_shape'},
            }
        ],
    )
    # f1 was missing -> filled from evidence; f2 already had a real value
    # -> untouched (combine_first never overwrites a real existing value).
    assert list(state.curated['roof_shape']) == ['Gable', 'Existing Hip']
    assert list(state.curated['roof_shape_source']) == [
        state.curated['roof_shape_source'].iloc[0],
        'legacy',
    ]
    assert pd.notna(state.curated['roof_shape_source'].iloc[0])
    # No duplicate columns from the drop-then-concat batching.
    assert list(state.curated.columns).count('roof_shape') == 1
    assert list(state.curated.columns).count('roof_shape_source') == 1
