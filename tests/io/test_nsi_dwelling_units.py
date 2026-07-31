import pandas as pd

from openplaces.io.harmonizer.attributes import _OCC_UNITS

NSI_REMAP = (
    'src/openplaces/recipes/US/_all/building/nsi/2022/'
    'US_building-nsi-2022_occupancy-type-remap.csv'
)
NSI_REMAP_2026 = (
    'src/openplaces/recipes/US/_all/building/nsi/2026/'
    'US_building-nsi-2026_occupancy-type-remap.csv'
)


def test_nsi_occupancy_labels_map_to_dwelling_units():
    labels = [
        'Single Family, 1 story, no basement',
        'Single Family, split-level, with basement',
        'Manufactured Home',
        'Multi-Family, 2 units',
        'Multi-Family, 3-4 units',
        'Multi-Family, 5-10 units',
        'Multi-Family, 10-19 units',
        'Multi-Family, 20-50 units',
        'Multi-Family, 50 plus units',
    ]

    units = pd.Series(labels).map(_OCC_UNITS)

    assert units.notna().all()
    assert units.tolist() == [1.0, 1.0, 1.0, 2.0, 3.5, 7.0, 14.5, 35.0, 51.0]


def test_nsi_remap_dwelling_units_are_numeric():
    remap = pd.read_csv(NSI_REMAP)
    residential = remap['group'].isin(
        ['Single Family', 'Manufactured Home', 'Multi Family']
    )

    units = pd.to_numeric(remap.loc[residential, 'n_dwellings'], errors='coerce')

    assert units.notna().all()


def test_nsi_remap_2026_dwelling_units_are_numeric():
    remap = pd.read_csv(NSI_REMAP_2026)
    residential = remap['group'].isin(
        ['Single Family', 'Manufactured Home', 'Multi Family']
    )

    units = pd.to_numeric(remap.loc[residential, 'n_dwellings'], errors='coerce')

    assert units.notna().all()
