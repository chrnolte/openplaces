"""Every canonical name the placeslab column-notes sidecar maps to is a
registered attribute, so the crosswalk enrichment never emits a column
the registry cannot describe. The sidecar once mapped `ct_p` to a name
no registry row defined."""

from pathlib import Path

import pandas as pd

import openplaces
from openplaces.core.attribute_registry import load_registry

SIDECAR = (
    Path(openplaces.__file__).parent
    / 'recipes/US/_all/parcel/placeslab/fmv2026'
    / 'US_parcel-placeslab-fmv2026_column-notes.csv'
)


def test_every_mapped_canonical_name_is_registered():
    notes = pd.read_csv(SIDECAR)
    mapped = notes['canonical_name'].dropna()
    unregistered = sorted(set(mapped) - set(load_registry().index))
    assert unregistered == []


def test_the_clipped_footprint_pair_is_mapped():
    notes = pd.read_csv(SIDECAR).set_index('name')
    assert notes.loc['m2_bld_fp', 'canonical_name'] == 'footprint_area_m2_in_parcel'
    assert notes.loc['n_bld_fp', 'canonical_name'] == 'n_footprints_per_parcel'
    assert (
        notes.loc['f_soil_farmlandofuniqueimportance', 'canonical_name']
        == 'soil_farmland_unique_share'
    )
