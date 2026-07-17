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
