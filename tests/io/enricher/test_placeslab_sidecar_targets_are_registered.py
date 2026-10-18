"""Every canonical name the placeslab column-notes sidecars map to is a
registered attribute, so the crosswalk enrichment never emits a column
the registry cannot describe. The sidecar once mapped `ct_p` to a name
no registry row defined."""

from pathlib import Path

import pandas as pd
import pytest

import openplaces
from openplaces.core.attribute_registry import load_registry

PLACESLAB = Path(openplaces.__file__).parent / 'recipes/US/_all/parcel/placeslab'
SIDECAR = PLACESLAB / 'fmv2026' / 'US_parcel-placeslab-fmv2026_column-notes.csv'
OPEN_SIDECAR = (
    PLACESLAB / 'fmv2026open' / 'US_parcel-placeslab-fmv2026open_column-notes.csv'
)
BOTH = pytest.mark.parametrize(
    'sidecar', [SIDECAR, OPEN_SIDECAR], ids=['fmv2026', 'fmv2026open']
)


@BOTH
def test_every_mapped_canonical_name_is_registered(sidecar):
    notes = pd.read_csv(sidecar)
    mapped = notes['canonical_name'].dropna()
    unregistered = sorted(set(mapped) - set(load_registry().index))
    assert unregistered == []


@BOTH
def test_no_two_raw_columns_land_on_one_evidence_name(sidecar):
    notes = pd.read_csv(sidecar).dropna(subset=['canonical_name'])
    names = notes['canonical_name'] + '_' + notes['suffix'].fillna('').astype(str)
    assert not names.duplicated().any()
    assert not notes['name'].duplicated().any()


def test_the_clipped_footprint_pair_is_mapped():
    notes = pd.read_csv(SIDECAR).set_index('name')
    assert notes.loc['m2_bld_fp', 'canonical_name'] == 'footprint_area_m2_in_parcel'
    assert notes.loc['n_bld_fp', 'canonical_name'] == 'n_footprints_per_parcel'
    assert (
        notes.loc['f_soil_farmlandofuniqueimportance', 'canonical_name']
        == 'soil_farmland_unique_share'
    )
