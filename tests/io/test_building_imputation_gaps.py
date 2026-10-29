"""Tests for gap-trained boosting, real-gap evaluation and repairs."""

import numpy as np
import pandas as pd

from openplaces.io import building_imputation as bi
from openplaces.io import building_imputation_gaps as gaps


def test_implausible_story_counts_are_voided():
    frame = pd.DataFrame(
        {
            'n_stories': [2.0, 75.0, 12.0, 12.0, 0.0],
            'n_stories_source': ['nsi'] * 5,
            'occupancy_type': [
                'Single-Family',
                'Retail',
                'Secondary',
                'High-Rise Multi-Family',
                'Retail',
            ],
        }
    )
    out, counts = gaps.void_implausible(frame)
    assert out['n_stories'].isna().tolist() == [False, True, True, False, True]
    assert out['n_stories_source'].isna().tolist() == [False, True, True, False, True]
    assert sum(counts.values()) == 3


def _rows(n=1000):
    rng = np.random.default_rng(0)
    return pd.DataFrame({'a': rng.normal(size=n), 'b': rng.normal(size=n)})


def test_half_the_training_rows_get_real_gap_patterns():
    x = _rows()
    y = pd.Series(np.arange(len(x)), index=x.index, dtype=float)
    w = np.ones(len(x))
    # Every incomplete building lacks 'a' and keeps 'b'.
    patterns = pd.DataFrame({'a': [np.nan] * 10, 'b': [1.0] * 10})
    out, y_out, w_out = gaps.gap_trained_rows(
        x, y, w, patterns, np.random.default_rng(1)
    )
    share = out['a'].isna().mean()
    assert 0.45 < share < 0.55
    assert out['b'].notna().all()
    # Rows are masked, never added or dropped; labels and weights stay.
    assert out.index.equals(x.index)
    assert y_out is y and w_out is w


def test_heldout_rows_get_the_patterns_of_incomplete_buildings():
    xt = _rows(200)
    patterns = pd.DataFrame({'a': [np.nan, 1.0], 'b': [1.0, np.nan]})
    out = gaps.real_gap_heldout(xt, patterns, np.random.default_rng(2))
    # Each row lost exactly one input, as one of the two patterns says.
    assert (out.isna().sum(axis=1) == 1).all()
    assert out['a'].isna().any() and out['b'].isna().any()


def test_extensions_plug_into_the_imputation():
    assert gaps.EXTENSIONS.train_rows is gaps.gap_trained_rows
    assert gaps.EVALUATION.train_rows is None
    assert isinstance(gaps.EXTENSIONS, bi.ImputationExtensions)
    loaded = bi.load_extensions('openplaces.io.building_imputation_gaps:EXTENSIONS')
    assert loaded is gaps.EXTENSIONS
