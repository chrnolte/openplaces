"""Street-view enrichments of buildings after the cut at parcel lines.

plans/buildings-from-parcel-pieces.md, step 3 (decision D6): the
story count from street view (image-googlestreetview-2026) is keyed by
building. An uncut outline keeps `building_id == footprint_id`, so its
output from an earlier run still joins; a piece has a new id, which no
earlier output names, and stays missing until street view runs on it.
An outline enrichment (roof shape) is joined on `footprint_id` and
reaches every piece. Every id and value is fabricated
(docs/5_contribute/no-personal-data.rst).
"""

from __future__ import annotations

import pandas as pd
from shapely.geometry import box

from openplaces.core.schema import AdminId
from openplaces.io import to_parquet
from openplaces.io.curator import CurateState
from openplaces.io.curator.evidence import merge_enrichments
from openplaces.recipe import get_output_path, get_recipe_by_id
from tests.io.curator import test_merge_enrichments as base
from tests.io.harmonizer.test_parcel_pieces import _lots, _row_of_four, _spine, _split

COUNTY = base.COUNTY
N_STORIES = base.N_STORIES
ROOF_SHAPE = base.ROOF_SHAPE
# The fixture pointing the data root at a temporary directory.
data_root = base.data_root
BUILDING_SPINE = 'US_building-spine-2026'
FOOTPRINT_SPINE = 'US_footprint-spine-2026'


def _buildings():
    """f1 a row cut into four pieces, f2 a house on a lot of its own."""
    spine, lots, points, values = _row_of_four()
    spine = pd.concat([spine, _spine(f2=box(100, 0, 110, 10))])
    lots = pd.concat([lots, _lots(Q=box(95, -5, 115, 15))])
    out, _, stats = _split(spine, lots, points, values)
    assert stats['outlines_cut'] == 1
    return out


def _write(recipe_id, entity_recipe_id, frame):
    path = get_output_path(
        get_recipe_by_id(recipe_id), AdminId(COUNTY), entity_recipe_id=entity_recipe_id
    )
    to_parquet(frame, path)


def test_uncut_ids_keep_their_outputs_and_piece_ids_are_new(data_root):
    buildings = _buildings()
    # The story counts an earlier run wrote, one building per outline.
    _write(
        N_STORIES,
        BUILDING_SPINE,
        pd.DataFrame(
            {'n_stories_brails': [2.0, 1.0]},
            index=pd.Index(['f1', 'f2'], name='building_id'),
        ),
    )
    _write(
        ROOF_SHAPE,
        FOOTPRINT_SPINE,
        pd.DataFrame(
            {'roof_shape_brails': ['Gable', 'Hip']},
            index=pd.Index(['f1', 'f2'], name='footprint_id'),
        ),
    )
    curated = pd.DataFrame(
        {'footprint_id': buildings['footprint_id'].to_numpy()},
        index=pd.Index(buildings.index, name='building_id'),
    )
    state = CurateState(
        recipe={'recipe_id': 'US_building-openplaces-2026'},
        entity_recipe=get_recipe_by_id(BUILDING_SPINE),
        admin_id=AdminId(COUNTY),
        verbose=False,
        timer=None,
        curated=curated,
    )
    out = merge_enrichments(
        state,
        recipes=[
            {'recipe_id': N_STORIES, 'columns': {'n_stories_brails': 'n_stories'}},
            {
                'recipe_id': ROOF_SHAPE,
                'entity_recipe': FOOTPRINT_SPINE,
                'key': 'footprint_id',
                'columns': {'roof_shape_brails': 'roof_shape'},
            },
        ],
    ).curated
    pieces = out.index[out['footprint_id'] == 'f1']
    assert len(pieces) == 4
    assert 'f1' not in out.index
    assert out.loc['f2', 'n_stories'] == 1.0
    assert out.loc[pieces, 'n_stories'].isna().all()
    assert out.loc[pieces, 'roof_shape'].tolist() == ['Gable'] * 4
    assert out.loc['f2', 'roof_shape'] == 'Hip'
