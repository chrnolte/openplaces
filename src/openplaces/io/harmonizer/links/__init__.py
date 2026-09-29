"""
Pipeline steps that create and use relationships between the spine and
reference datasets, one module per concern since 2026-09-29 (until then a
single 4,259-line module):

- spatial: link_to_reference, the overlay crosswalk and its sidecar
- points: the point join (dwellings, NSI, address points)
- by_id: link_by_id, the exact-key join and its guards
- address_ranges, additions, condo_clusters, overlaps: one step each
- discovery, sidecars, _shared: what the steps share

Every top-level name, private ones included, is re-exported here because
dag, the curator, entity_links and the tests import them from this
package.

**Patching.** A function looks names up in its own module's globals, so a
patch on this package would not reach `link_by_id`'s `get_entities`
(AGENTS.md, module layer hierarchy). Until 2026-09-29 this package was
one module and about 130 test sites patch names on it
(`monkeypatch.setattr(links, 'get_entities', ...)`, `find_recipes`,
`overlay_polygons`, `_apply_remap_csvs`), so the package forwards: an
attribute set on it is also set on every submodule whose namespace holds
that name, and an attribute it does not define is read from the first
submodule that has it. That keeps every existing patch meaningful, at the
cost of a package that is not a plain module. The debt is in
plans/stage-contract-audit.md: tests move to patching the submodule the
step lives in, and the forwarder goes.
"""

from __future__ import annotations

import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

from openplaces.io.harmonizer.links import (  # noqa: E402
    _shared,
    additions,
    address_ranges,
    by_id,
    combine,
    condo_clusters,
    discovery,
    overlaps,
    points,
    sidecars,
    spatial,
)
from openplaces.io.harmonizer.links._shared import (  # noqa: F401
    _COVERAGE_SCORE_EPS,
    _CROSSWALK_COLS,
    _LINK_INDEX_KEY,
    _LINK_KEY_COLUMNS,
    _LINK_METADATA_KEY,
    _REAL_FOOTPRINT_TOUCH_TOLERANCE_M,
    DEFAULT_LINK_KEY,
    PARCEL_ID_LOCAL_KEYS,
)
from openplaces.io.harmonizer.links.additions import (  # noqa: F401
    _positive_value_density,
    infer_spine_additions,
)
from openplaces.io.harmonizer.links.address_ranges import (  # noqa: F401
    link_address_ranges,
)
from openplaces.io.harmonizer.links.by_id import (  # noqa: F401
    _ADDRESS_SCOPE_COLUMNS,
    _COUNT_COLUMNS_KEY,
    _REF_ADDRESS_COLUMNS,
    DEGENERATE_KEY_MAX_SHARE,
    DEGENERATE_KEY_MIN_ROWS,
    _accumulate_count,
    _columns_as_pairs,
    _derive_address_key,
    _move_units_to_lots,
    _neutralize_degenerate_keys,
    _placeholder_key_mask,
    _upstream_tokens,
    _warn_if_duplicate_key,
    _warn_if_link_underperforms,
    link_by_id,
)
from openplaces.io.harmonizer.links.combine import (  # noqa: F401
    _align_for_combine,
    _write_prioritized,
)
from openplaces.io.harmonizer.links.condo_clusters import (  # noqa: F401
    consolidate_condo_cluster_footprints,
)
from openplaces.io.harmonizer.links.discovery import (  # noqa: F401
    _apply_remap_csvs,
    _discover_link_sources,
    _find_admin_scoped_recipe_ids,
    _select_supplements,
)
from openplaces.io.harmonizer.links.overlaps import (  # noqa: F401
    resolve_overlaps,
)
from openplaces.io.harmonizer.links.points import (  # noqa: F401
    _POINT_QUALITY_ORDER,
    _aggregate_multipoint,
    _build_size_limit_dict,
    _dedup_address_points,
    _filter_by_size_limit,
    _link_spatial_point,
    _point_quality_sort,
    _point_step_config,
    flag_duplicate_points,
)
from openplaces.io.harmonizer.links.sidecars import (  # noqa: F401
    _fingerprint_safe_step,
    _fingerprints_match,
    _hash_file,
    _link_fingerprint,
    _load_link_sidecar,
    _load_point_link_sidecar,
    _with_source_hashes,
    _write_link_sidecar,
    _write_point_link_sidecar,
)
from openplaces.io.harmonizer.links.spatial import (  # noqa: F401
    _build_crosswalk,
    _find_reference_recipe,
    _link_spatial_overlay,
    _prepare_reference,
    _resolve_reference_recipe,
    _superseded_by_consolidation,
    link_to_reference,
    snap_chained_links,
)

_SUBMODULES = (
    _shared,
    spatial,
    sidecars,
    points,
    discovery,
    combine,
    by_id,
    address_ranges,
    additions,
    condo_clusters,
    overlaps,
)


class _ForwardingPackage(_types.ModuleType):
    """The package as a patch target: see the module docstring."""

    def __getattr__(self, name):
        for module in _SUBMODULES:
            if name in vars(module):
                return vars(module)[name]
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    def __setattr__(self, name, value):
        for module in _SUBMODULES:
            if name in vars(module):
                setattr(module, name, value)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        super().__delattr__(name)


_sys.modules[__name__].__class__ = _ForwardingPackage
