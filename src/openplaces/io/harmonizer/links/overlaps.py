"""Remove remaining geometry overlaps from the spine (resolve_overlaps)."""

from __future__ import annotations

from openplaces.geo.polygon import (
    clean_polygons,
    resolve_overlapping_polygons,
)
from openplaces.io.harmonizer import (
    HarmonizeState,
    _register,
)


@_register('resolve_overlaps', phase='geometry')
def resolve_overlaps(
    state: HarmonizeState,
    **_params,
) -> HarmonizeState:
    """Resolve remaining geometry overlaps in the spine.

    Calls :func:`~openplaces.geo.polygon.resolve_overlapping_polygons` on
    ``state.spine`` (with ``keep=False``).
    """
    if state.spine is None:
        return state

    state.spine = clean_polygons(state.spine)
    state.spine = resolve_overlapping_polygons(state.spine, keep=False)
    if state.verbose:
        print(f'  Resolve: {len(state.spine):,d} after resolving overlaps')
    if state.timer:
        state.timer.mark('Resolve')
    return state
