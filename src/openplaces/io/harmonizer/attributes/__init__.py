"""
Pipeline steps that attach reference-dataset evidence to the spine,
one module per concern since 2026-09-29 (until then a single
2,615-line module):

- reconcile: reconcile_attributes and rename_columns, with the
  polygon and point attribution in polygon and point
- priority: classify_footprint_priority
- use_classes: derive_use_classes
- morphology: summarize_footprint_morphology
- shared_land, condo_clusters: the group detections
- property_counts: estimate_property_counts
- dwelling_address: attribute_dwelling_address
- _shared: suffix helpers, reference column lists, reverse_occ_units

Value selection, gap-filling, and occupancy inference run in the
curation stage (see openplaces.io.curator), not here. Every
top-level name is re-exported here because the links package, the
enricher and the tests import them from this package.
"""

from __future__ import annotations

from openplaces.io.harmonizer.attributes._shared import (  # noqa: F401
    _ID_COLUMN,
    _OCC_UNITS,
    _POINT_REF_COLS,
    _POLYGON_REF_COLS,
    _attributed_name,
    _dominant_by_area,
    _join_distinct,
    _point_suffix,
    _resolve_suffix,
    reverse_occ_units,
)
from openplaces.io.harmonizer.attributes.condo_clusters import (  # noqa: F401
    _NON_CONDO_USE_SUBGROUP_TERMS,
    _cluster_condo_parcels,
    detect_condo_building_clusters,
)
from openplaces.io.harmonizer.attributes.dwelling_address import (  # noqa: F401
    attribute_dwelling_address,
)
from openplaces.io.harmonizer.attributes.morphology import (  # noqa: F401
    summarize_footprint_morphology,
)
from openplaces.io.harmonizer.attributes.point import (  # noqa: F401
    _attribute_point_reference,
)
from openplaces.io.harmonizer.attributes.polygon import (  # noqa: F401
    _attribute_polygon_reference,
)
from openplaces.io.harmonizer.attributes.priority import (  # noqa: F401
    classify_footprint_priority,
)
from openplaces.io.harmonizer.attributes.property_counts import (  # noqa: F401
    estimate_property_counts,
)
from openplaces.io.harmonizer.attributes.reconcile import (  # noqa: F401
    _attribute_absent_point_reference,
    _collect_dwelling_linked,
    reconcile_attributes,
    rename_columns,
)
from openplaces.io.harmonizer.attributes.shared_land import (  # noqa: F401
    detect_shared_land_groups,
)
from openplaces.io.harmonizer.attributes.use_classes import (  # noqa: F401
    derive_use_classes,
)
