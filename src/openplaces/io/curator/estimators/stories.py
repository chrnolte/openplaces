"""Estimate a story count from a measured building height."""

from __future__ import annotations

import pandas as pd

from openplaces.io.curator import CurateState, _register


@_register('derive_stories_from_height', phase='infer')
def derive_stories_from_height(
    state: CurateState,
    column: str = 'n_stories_footprint_fema',
    height_column: str = 'height_footprint_fema',
    floor_height_m: float = 3.05,
) -> CurateState:
    """Derive a story count from a measured building height.

    Approximates the story count as ``height / floor_height_m``, rounded and
    floored at one story. A missing or non-positive height yields a missing
    story count rather than a fabricated minimum.

    Parameters
    ----------
    state : CurateState
        The curation state with the target GeoDataFrame in state.curated.
    column : str, optional
        Output column name for the derived story count.
    height_column : str, optional
        Source column holding measured building height (metres). No-op if
        absent from ``state.curated``.
    floor_height_m : float, optional
        Assumed height per story, in metres.
    """
    curated = state.curated
    if height_column not in curated.columns:
        return state

    height = pd.to_numeric(curated[height_column], errors='coerce')
    stories = (height / floor_height_m).round().clip(lower=1)
    curated[column] = stories.where(height > 0)

    state.curated = curated
    return state
