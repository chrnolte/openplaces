"""Gap-trained boosting, real-gap evaluation and validity repairs.

Method: Mohammad Askari (UCLA), "Imputing All Nine Building Attributes
in CHEER Places North Carolina", technical memo, 2026-10-15
(repository CHEER-Hub/building-inventory-imputation v2.0), sections 3,
4 and 8: the validity repairs, gap-trained boosting, and the Stage 2b
test on held-out buildings as sparse as real gaps. Implemented in
openplaces from the memo's description, without code from that
repository; the method and its evaluation are Mohammad Askari's
research. Licensed Apache-2.0 with the rest of openplaces.

The three steps plug into openplaces.io.building_imputation through
its ImputationExtensions (a recipe names `impute: extensions:
openplaces.io.building_imputation_gaps:EXTENSIONS`):

- repair: observed values that cannot be true are emptied, so they are
  imputed (memo, section 3).
- train_rows: before a model is fitted, half of its training rows,
  chosen at random, are given the pattern of missing inputs of a
  building that genuinely lacks the target, so the trees learn what to
  do when the other attributes are absent (memo, section 4,
  "gap-trained boosting").
- holdout_rows: each held-out building gets the pattern of missing
  inputs of a randomly drawn building that genuinely lacks the target
  (memo, section 5, "Stage 2b").

Only values are ever removed, never added, and the masks are input
patterns, not a row subsample of the trees (design limit L12 is not
touched).

Measured 2026-10-25 on cheer-eastern-nc: 12.8% low with gap training,
21.9% low without, the median county's total structure value, with
v0.6 holding out 14 counties one at a time under Stage 2b. Year built
MAE 15.4 versus 16.4 years; story count accuracy 68.5% versus 66.2%;
sections 84.2% versus 82.5%; foundation type 39.2% versus 41.9%, the
one attribute gap training does not help.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.io.building_imputation import ImputationExtensions

# Occupancy classes that cannot exceed ten floors (memo, section 3):
# counts above 60 floors, or above ten on a class that cannot have
# them, are not observations.
LOW_RISE_OCCUPANCY = {
    'Secondary',
    'Single-Family',
    'Manufactured Home',
    'Agricultural',
    'Low-Rise Multi-Family',
}
MAX_STORIES = 60
MAX_LOW_RISE_STORIES = 10
# Share of training rows given a real gap pattern (memo, section 4).
GAP_SHARE = 0.5


def void_implausible(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Empty observed values that cannot be true, so they are imputed.

    Story counts above MAX_STORIES, above MAX_LOW_RISE_STORIES on a
    LOW_RISE_OCCUPANCY class, or below one. Two-digit and out-of-range
    years need no rule here: year labels already require a plausible
    year (building_imputation.YEAR_RANGE).

    Measured 2026-10-15 on cheer-eastern-nc: 73 counts above 60 and 396
    above 10 on low-rise classes.

    Returns
    -------
    frame : DataFrame
        A copy with the implausible cells set to missing.
    counts : dict
        Cells voided per rule.
    """
    out = frame.copy()
    stories = pd.to_numeric(out['n_stories'], errors='coerce')
    low = out['occupancy_type'].astype('string').isin(LOW_RISE_OCCUPANCY)
    rules = {
        'n_stories above 60': stories > MAX_STORIES,
        'n_stories above 10, low-rise class': (stories > MAX_LOW_RISE_STORIES)
        & low
        & (stories <= MAX_STORIES),
        'n_stories below 1': stories < 1,
    }
    voided = pd.Series(False, index=out.index)
    for mask in rules.values():
        voided |= mask.fillna(False)
    out.loc[voided, 'n_stories'] = np.nan
    if 'n_stories_source' in out.columns:
        out.loc[voided, 'n_stories_source'] = np.nan
    return out, {name: int(mask.fillna(False).sum()) for name, mask in rules.items()}


def _impose(rows: pd.DataFrame, patterns: pd.DataFrame, rng) -> pd.DataFrame:
    """Give each of *rows* the missing inputs of a randomly drawn pattern."""
    columns = [c for c in rows.columns if c in patterns.columns]
    if not len(rows) or not len(patterns) or not columns:
        return rows
    missing = patterns[columns].isna().to_numpy()
    drawn = missing[rng.integers(0, len(missing), size=len(rows))]
    out = rows.copy()
    for j, column in enumerate(columns):
        hide = drawn[:, j]
        if hide.any():
            out.loc[out.index[hide], column] = np.nan
    return out


def gap_trained_rows(x, y, w, patterns, rng):
    """Training rows with GAP_SHARE of them given real gap patterns.

    Parameters
    ----------
    x, y, w : DataFrame, Series, ndarray
        A model's training inputs, labels and weights; y and w are
        returned unchanged, since rows are masked, not added.
    patterns : DataFrame
        Input rows of buildings that genuinely lack the target.
    rng : numpy Generator
        Chooses the rows and the patterns.
    """
    if patterns is None or not len(patterns) or not len(x):
        return x, y, w
    chosen = rng.random(len(x)) < GAP_SHARE
    if not chosen.any():
        return x, y, w
    x = x.copy()
    x.loc[x.index[chosen]] = _impose(x.loc[x.index[chosen]], patterns, rng)
    return x, y, w


def real_gap_heldout(xt, patterns, rng):
    """Held-out rows, each given a real gap pattern (Stage 2b)."""
    if patterns is None or not len(patterns):
        return xt
    return _impose(xt, patterns, rng)


# Everything: repairs, gap-trained boosting and the Stage 2b hook.
EXTENSIONS = ImputationExtensions(
    repair=void_implausible,
    train_rows=gap_trained_rows,
    holdout_rows=real_gap_heldout,
)
# Without gap training: the existing models under the same repairs and
# Stage 2b hook, for a like-for-like comparison.
EVALUATION = ImputationExtensions(
    repair=void_implausible,
    holdout_rows=real_gap_heldout,
)
