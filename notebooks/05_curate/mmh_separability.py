"""How separable are manufactured homes from site-built houses, and where?

Written 2026-08-21 against a specific future: an inventory that no longer
depends on NSI or FEMA. Both are point sources this pipeline currently leans
on heavily for occupancy, and both are the reason the manufactured-home class
scores as well as it does today. If they go, the class has to be carried by
evidence the pipeline owns -- the assessor's own land-use text, footprint
morphology, the assessed-value pattern, and neighborhood composition.

This module measures whether that is possible, and more importantly *where it
stops being possible*. The motivating observation is that a manufactured-home
community is not a homogeneous thing: some blocks are exclusively
manufactured homes, and some accumulate site-built houses on lots that began
as manufactured-home lots. The second kind is where every neighborhood-based
signal fails, and a single aggregate accuracy figure hides it completely --
an earlier neighborhood signal scored -0.0071 F1 overall while being 100%
wrong on the seven points it actually moved.

So the report is **stratified by block composition**, not pooled. A source
that is 95% precise on pure blocks and 40% precise on mixed ones is a
different proposition from one that is 70% precise everywhere, and the
pooled number cannot tell them apart.

Usage
-----
Build a block share with `block_composition`, label the strata with
`label_block_type`, then score whatever evidence you care about with
`separability_report`. The forward-looking question is asked by choosing
which columns go into `block_composition` and which signals go into the
evidence dict: pass NSI and FEMA columns to see today's baseline, omit them
to see what the assessor, morphology and value evidence carry alone.

Nothing here asserts a threshold or writes to the pipeline. It is a
measurement, meant to be re-run when the inventory's source mix changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Truth codes used by the CHEER survey.
MANUFACTURED = 'MMH'
SINGLE_FAMILY = 'SFH'

# How homogeneous a block has to be before it counts as "pure". Blocks
# between the two bounds are the mixed case this module exists to isolate.
PURE_MANUFACTURED_MIN = 0.8
PURE_SITE_BUILT_MAX = 0.2

# A block needs at least this many other buildings before its composition
# means anything.
MIN_NEIGHBORS = 3


def block_composition(
    footprints: pd.DataFrame,
    class_columns: list[str],
    match_value: str = 'Manufactured Home',
    match_pattern: str | None = None,
    block_column: str = 'census_block_id',
    min_neighbors: int = MIN_NEIGHBORS,
) -> pd.Series:
    """Share of each building's block that reads as *match_value*, self excluded.

    A groupby on an id the building already carries -- deliberately not a
    spatial operation (no buffering, no boundary union, no nearest-neighbor
    search). See `io/curator/inferers.derive_group_class_share`, which uses
    the same mechanism in the pipeline, for why that distinction is kept.

    Parameters
    ----------
    class_columns : list of str
        Evidence columns to read the class from. Vary this to ask the
        forward-looking question -- pass only assessor-derived columns to
        see what remains without NSI and FEMA.
    match_value : str, optional
        Exact value that counts as the class. Right for the tidy
        vocabularies (`group_building_nsi`, `group_footprint_fema`).
    match_pattern : str, optional
        Case-insensitive regex, used instead of *match_value* when given.
        Required for assessor land-use text, which is free-form: a county
        writes 'DOUBLE WIDE MOHO' or 'RESIDENTIAL | MOBILE HOME', never the
        canonical label. Matching those by equality silently finds nothing --
        measured on the ten surveyed counties, exact matching identified
        **zero** predominantly-manufactured blocks where the NSI/FEMA
        vocabularies found 9,216.
    """
    present = [c for c in class_columns if c in footprints.columns]
    if not present or block_column not in footprints.columns:
        return pd.Series(np.nan, index=footprints.index, dtype='float64')

    is_match = pd.Series(False, index=footprints.index)
    for column in present:
        values = footprints[column].astype(object)
        if match_pattern:
            hit = values.str.contains(match_pattern, case=False, na=False, regex=True)
        else:
            hit = values.eq(match_value)
        is_match = is_match | hit
    is_match = is_match.fillna(False).astype(float)

    groups = footprints[block_column]
    others = is_match.groupby(groups).transform('sum') - is_match
    n_others = groups.groupby(groups).transform('size').astype(float) - 1.0
    share = others / n_others.where(n_others > 0)
    return share.where(n_others >= min_neighbors).where(groups.notna())


def label_block_type(
    share: pd.Series,
    pure_manufactured_min: float = PURE_MANUFACTURED_MIN,
    pure_site_built_max: float = PURE_SITE_BUILT_MAX,
) -> pd.Series:
    """Split blocks into pure-manufactured, mixed, and pure-site-built.

    The middle band is the interesting one: a site-built house standing on a
    lot in a manufactured-home community, or a manufactured home in an
    otherwise conventional subdivision. Neighborhood evidence is close to
    worthless there by construction, and any rule that leans on it will make
    its mistakes in this stratum.
    """
    return pd.Series(
        np.select(
            [
                share.isna(),
                share >= pure_manufactured_min,
                share <= pure_site_built_max,
            ],
            ['no block context', 'pure manufactured', 'pure site-built'],
            default='mixed',
        ),
        index=share.index,
        dtype=object,
    )


def score_evidence(
    truth: pd.Series, fires: pd.Series, positive: str = MANUFACTURED
) -> dict:
    """Precision, recall and support of one piece of evidence for one class."""
    is_positive = truth.eq(positive)
    tp = int((fires & is_positive).sum())
    fp = int((fires & ~is_positive).sum())
    fn = int((~fires & is_positive).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and not np.isnan(precision) and not np.isnan(recall)
        else np.nan
    )
    return {
        'fires': int(fires.sum()),
        'precision': round(precision, 3) if precision == precision else np.nan,
        'recall': round(recall, 3) if recall == recall else np.nan,
        'f1': round(f1, 3) if f1 == f1 else np.nan,
        'support': int(is_positive.sum()),
    }


def separability_report(
    points: pd.DataFrame,
    evidence: dict[str, pd.Series],
    truth_column: str = 'occupancy_type',
    block_share_column: str = 'block_share',
) -> pd.DataFrame:
    """Score each evidence source per block stratum.

    Parameters
    ----------
    points : pandas.DataFrame
        Survey points with a truth column and a block share, restricted to
        the two classes being separated.
    evidence : dict of {name: boolean Series}
        Each entry is one signal's per-point firing, aligned to *points*.

    Returns
    -------
    pandas.DataFrame
        One row per (stratum, evidence), with precision, recall, F1 and
        support. Read the mixed stratum first: it is where a source that
        looks good in aggregate goes wrong, and where a future
        NSI-and-FEMA-free inventory will live or die.
    """
    truth = points[truth_column].astype(object)
    strata = label_block_type(points[block_share_column])
    rows = []
    for stratum in (
        'pure manufactured',
        'mixed',
        'pure site-built',
        'no block context',
        'ALL',
    ):
        mask = (
            pd.Series(True, index=points.index)
            if stratum == 'ALL'
            else (strata == stratum)
        )
        if not mask.any():
            continue
        for name, fires in evidence.items():
            scored = score_evidence(truth[mask], fires.reindex(points.index)[mask])
            rows.append(
                {
                    'stratum': stratum,
                    'n_points': int(mask.sum()),
                    'evidence': name,
                    **scored,
                }
            )
    return pd.DataFrame(rows)
