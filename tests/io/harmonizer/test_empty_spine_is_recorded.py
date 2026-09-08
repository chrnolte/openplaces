"""A unit with nothing to build from gets an empty table, not no file.

`US_property-spine-2026` has sources in 4 of the 86 CHEER counties. In
the other 82 its pipeline resolves no spine at all, which is a correct
answer about the data rather than a failure. Writing nothing made the
two indistinguishable to an orchestrator: the job exits 0, the declared
output never appears, and Snakemake fails the job and everything
downstream of it.
"""

import geopandas as gpd
import pandas as pd
import pytest

import openplaces.io.harmonizer as harmonizer_module
from openplaces.core.schema import AdminId
from openplaces.io import read_parquet
from openplaces.io.harmonizer.links import link_by_id

ADMIN_ID = AdminId('US-NC-TYR')


def _harmonizer(monkeypatch, tmp_path, recipe_id, filename, geometry):
    """A Harmonizer stub whose output path is inside tmp_path."""
    out_path = tmp_path / filename
    monkeypatch.setattr(harmonizer_module, 'get_output_path', lambda *a, **k: out_path)
    monkeypatch.setattr(harmonizer_module, 'saves_geometry', lambda recipe: geometry)
    harmonizer = harmonizer_module.Harmonizer.__new__(harmonizer_module.Harmonizer)
    harmonizer.recipe = {'recipe_id': recipe_id, 'admin_id': AdminId('US')}
    harmonizer.verbose = False
    return harmonizer, out_path


def test_a_geometry_recipe_writes_an_empty_geodataframe(monkeypatch, tmp_path):
    harmonizer, out_path = _harmonizer(
        monkeypatch, tmp_path, 'US_property-spine-2026', 'property.parquet', True
    )

    with pytest.warns(UserWarning, match='saved an empty table'):
        harmonizer._save_empty_spine(ADMIN_ID)

    assert out_path.exists()
    # The geometry sidecar the orchestrator declares as an output too.
    assert out_path.with_stem(out_path.stem + '_geo').exists()
    assert len(read_parquet(out_path)) == 0


def test_an_attribute_only_recipe_writes_a_plain_empty_table(monkeypatch, tmp_path):
    harmonizer, out_path = _harmonizer(
        monkeypatch, tmp_path, 'US_footprint-spine-2026', 'attributes.parquet', False
    )

    with pytest.warns(UserWarning, match='saved an empty table'):
        harmonizer._save_empty_spine(ADMIN_ID)

    assert out_path.exists()
    assert not out_path.with_stem(out_path.stem + '_geo').exists()
    back = read_parquet(out_path)
    assert len(back) == 0
    assert not isinstance(back, gpd.GeoDataFrame)


def test_a_link_skips_the_empty_table_instead_of_raising(monkeypatch):
    """The consumer takes the branch it already takes for a missing file.

    `link_by_id` returns the state untouched when the reference has no
    key column, and an empty table has no columns at all, so recording
    the empty answer introduces no new failure mode downstream.
    """

    class _State:
        spine = pd.DataFrame(
            {'address_id_local': ['a', 'b']},
            index=pd.Index([1, 2], name='footprint_id'),
        )
        admin_id = ADMIN_ID
        references: dict = {}
        metadata: dict = {}
        verbose = False

    state = _State()
    monkeypatch.setattr(
        'openplaces.io.harmonizer.links.get_entities', lambda *a, **k: pd.DataFrame()
    )

    with pytest.warns(UserWarning):
        out = link_by_id(
            state,
            recipe_id='US_property-spine-2026',
            spine_key='address_id_local',
            ref_key='address_id_local',
            columns={'occupancy_type_raw': 'occupancy_type_property_shovels'},
        )

    assert out is state
    assert 'occupancy_type_property_shovels' not in out.spine.columns
