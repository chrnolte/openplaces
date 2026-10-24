"""The share bucket receives redacted bundles and nothing else.

Curated per-county files used to be saved into the share bucket, and
one carried a restricted roll's attributes unredacted. Curated files
now stay in a private bucket, a recipe saving to `share` does not
load, and every table write into the share bucket outside
`export_delivery`'s redacted writes is refused. Rows and sources here
are fabricated.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.delivery as delivery_module
from openplaces.config import cfg
from openplaces.io import read_parquet, save_parquet, to_csv, to_parquet
from openplaces.io.delivery import export_delivery
from openplaces.io.delivery.redaction import WITHHELD
from openplaces.io.tables import (
    UnredactedShareWriteError,
    redacted_share_writes,
)
from openplaces.recipe import get_output_path, get_recipe_by_id
from openplaces.recipe.loading import get_recipe_dict
from tests.io.test_delivery import _county, _recipe
from tests.io.test_delivery_restriction_gate import _source


def _rows():
    return pd.DataFrame(
        {'year_built': [1990, 2001], 'year_built_source': ['parcel', 'parcel']},
        index=pd.Index(['a', 'b'], name='property_id'),
    )


def _share(*parts) -> Path:
    return Path(cfg.get_dir('share')).joinpath(*parts)


def test_a_table_write_into_share_is_refused(mock_data_root):
    target = _share('US', 'XX', 'YY', 'US-XX-YY_property-test-2026.parquet')

    with pytest.raises(UnredactedShareWriteError):
        to_parquet(_rows(), target)
    with pytest.raises(UnredactedShareWriteError):
        to_csv(_rows(), target.with_suffix('.csv'))

    assert not target.exists()
    assert not target.with_suffix('.csv').exists()


def test_a_split_geo_write_into_share_leaves_no_file(mock_data_root):
    frame = gpd.GeoDataFrame(
        _rows(), geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)], crs='EPSG:4326'
    )
    target = _share('US', 'XX', 'YY', 'US-XX-YY_parcel-test-2026.parquet')

    with pytest.raises(UnredactedShareWriteError):
        save_parquet(frame, target)

    assert not target.exists()
    assert not target.with_stem(f'{target.stem}_geo').exists()


def test_a_team_folder_is_no_exception(mock_data_root):
    target = _share('US', 'XX', '_all', 'region-x', 'team', 'bundle.parquet')

    with pytest.raises(UnredactedShareWriteError):
        to_parquet(_rows(), target)


def test_writes_elsewhere_and_redacted_writes_pass(mock_data_root):
    private = Path(cfg.get_dir('core')) / 'US-XX-YY_property-test-2026.parquet'
    to_parquet(_rows(), private)
    assert private.exists()

    shared = _share('bundle.parquet')
    with redacted_share_writes():
        to_parquet(_rows(), shared)
    assert shared.exists()
    # The permission ends with the block.
    with pytest.raises(UnredactedShareWriteError):
        to_parquet(_rows(), shared)


def test_a_recipe_saving_to_share_does_not_load(tmp_path):
    path = tmp_path / 'US_property-test-2026.yaml'
    path.write_text(
        'stage: curate\n'
        'admin_id: US\n'
        'entity:\n'
        '  entity_type: property\n'
        '  source: {source_id: test}\n'
        '  version: 2026\n'
        'save_to:\n'
        '  data_dir: share\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='redacted bundles'):
        get_recipe_dict(path, 'US')


@pytest.mark.parametrize(
    'recipe_id',
    [
        'US_parcel-openplaces-2026',
        'US_property-openplaces-2026',
        'US_footprint-openplaces-2026',
        'US_transaction-openplaces-2026',
    ],
)
def test_curated_recipes_save_to_a_private_bucket(recipe_id):
    recipe = get_recipe_by_id(recipe_id)
    data_dir = recipe['save_to']['data_dir']

    assert data_dir != 'share'
    assert cfg.config['directories'].get(data_dir) is not None


def test_export_files_a_redacted_bundle_in_share_from_private_files(
    mock_data_root, monkeypatch
):
    """The curated county keeps the restricted values; the bundle not."""
    _county(['a', 'b'], admin3_id='US-NC-AL')
    _county(['c', 'd'], admin3_id='US-NC-BB')
    monkeypatch.setattr(
        delivery_module,
        'restricted_inputs',
        lambda *_a, **_kw: [_source('closedcounty', 'US-NC-AL', ['year_built'])],
    )

    paths = export_delivery(_recipe(), 'US-NC', admin_ids=['US-NC-AL', 'US-NC-BB'])

    share_root = Path(cfg.get_dir('share')).resolve()
    curated = get_output_path(_recipe(), 'US-NC-AL').resolve()
    assert share_root not in curated.parents
    for path in paths.values():
        assert share_root in path.resolve().parents

    canonical = read_parquet(paths['canonical'])
    assert canonical.loc[['a', 'b'], 'year_built'].isna().all()
    assert (canonical.loc[['a', 'b'], 'occupancy_type_source'] == WITHHELD).all()
    assert canonical.loc[['c', 'd'], 'year_built'].notna().all()
    # The private file is untouched by the redaction.
    assert read_parquet(curated, geom=True)['year_built'].notna().all()
