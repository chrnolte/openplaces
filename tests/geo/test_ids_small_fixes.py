"""Two narrow `geo/ids.py` contracts.

`add_ubid_index(duplicates='drop')` must drop the rows whose UBID was
dropped, and a bundled conversion with no `conv` code must stay absent
rather than falling back to 'simple'.
"""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from openplaces.geo.ids import _parcel_id_links, _resolve_instruction, add_ubid_index


def test_add_ubid_index_drops_the_duplicate_rows():
    # Two identical footprints share one UBID.
    gdf = gpd.GeoDataFrame(
        {
            'name': ['a', 'b', 'c'],
            'geometry': [
                box(0, 0, 0.001, 0.001),
                box(0, 0, 0.001, 0.001),
                box(1, 1, 1.001, 1.001),
            ],
        },
        crs='epsg:4326',
    )

    result = add_ubid_index(gdf, duplicates='drop')

    assert len(result) == 2
    assert result.index.name == 'ubid'
    assert not result.index.duplicated().any()


def test_absent_conv_resolves_to_none_not_simple():
    links = _parcel_id_links()
    missing = links.index[links['conv_parcel'].isna()]
    assert len(missing), 'expected at least one bundled rule with no conv'

    _, conv, _ = _resolve_instruction(missing[0], None, 'parcel')

    assert conv is None
