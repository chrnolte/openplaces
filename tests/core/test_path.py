import pytest

from openplaces import cfg
from openplaces.path import external_path, path  # adjust import


@pytest.mark.parametrize(
    'call,expected',
    [
        (lambda: path(), lambda: cfg.dir_core / '_all'),
        (lambda: path(filename='data.csv'), lambda: cfg.dir_core / '_all/data.csv'),
        (
            lambda: path(
                admin_id='US-MA-MI',
                entity='parcel-massgis-2018',
                dataset='water-lake-nhd-2022',
            ),
            lambda: (
                cfg.dir_core / 'US/MA/MI/_all/parcel/massgis/2018/US-MA-MI_parcel-'
                'massgis-2018_water-lake-nhd-2022.parquet'
            ),
        ),
        (
            lambda: external_path(
                admin_id='US-ND',
                dataset='water-lake-nhd-2019hr',
                filename='NHD_H_North_Dakota_State_GDB.zip',
            ),
            lambda: (
                cfg.dir_external
                / 'US/ND/_all/water/lake/nhd/2019hr/NHD_H_North_Dakota_State_GDB.zip'
            ),
        ),
    ],
)
def test_path_generation(call, expected):
    """Test that path functions generate correct file paths."""
    assert call() == expected()
