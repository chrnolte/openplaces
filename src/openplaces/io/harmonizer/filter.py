"""
Pipeline step for filtering and flagging spine entries:
  - filter_entities: apply spatial or attribute-based criteria [stub]
"""

from __future__ import annotations

from openplaces.io.harmonizer import HarmonizeState, _register


@_register('filter_entities', phase='geometry')
def filter_entities(
    state: HarmonizeState,
    criteria: list[dict] | None = None,
    **_params,
) -> HarmonizeState:
    """Flag or drop spine entries by spatial or attribute criteria [stub].

    Planned criteria include:

    ``largest_on_parcel``
        Flag the footprint with the largest area on each parcel, useful for
        identifying the primary structure when multiple footprints share a
        parcel.

    Parameters
    ----------
    criteria : list of dict, optional
        Each dict must have a ``type`` key naming the criterion plus any
        criterion-specific parameters.

    Raises
    ------
    NotImplementedError
        Always — this step is not yet implemented.
    """
    raise NotImplementedError(
        "The 'filter_entities' harmonization step is not yet implemented. "
        'Remove it from the recipe pipeline or implement it in '
        'openplaces/io/harmonizer/filter.py.'
    )
