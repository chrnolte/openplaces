import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import openplaces.io.curator.occupancy as occ_mod
from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import classify_manufactured_homes

# Class names and geometry thresholds the step reads from the recipe occupancy
# block; the test asserts these values flow through rather than hardcoded ones.
OCC_CONFIG = {
    'residential_classes': ['Single-Family', 'Multi-Family', 'Manufactured Home'],
    'rules': {
        'manufactured_home_geometry': {
            'aspect_min': 2.5,
            'area_max_m2': 185,
            'class': 'Manufactured Home',
        },
        'single_family_dwellings': {'class': 'Single-Family'},
    },
}

# Keyword ruleset coercing the parcel use string to coarse classes (Tier 1).
RULESET = [
    {
        'pattern': 'Manufactured',
        'match_type': 'contains',
        'occupancy_type': 'Manufactured Home',
        'reviewed': False,
    },
    {
        'pattern': 'Single',
        'match_type': 'contains',
        'occupancy_type': 'Single-Family',
        'reviewed': False,
    },
]


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={
            'entity': 'footprint-openplaces-2026',
            'admin_id': 'US',
            'occupancy': OCC_CONFIG,
        },
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=True,
        timer=None,
        curated=df,
    )


def _patch(monkeypatch):
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda state, ruleset: RULESET)


# Projected (metre) CRS so footprint areas land in the morphology thresholds'
# range; lat/lon degrees would give geodesic areas of millions of m2.
CRS = 32617  # UTM 17N
BX, BY = 250000.0, 3900000.0


def test_classify_manufactured_homes_fallback_rule(monkeypatch):
    _patch(monkeypatch)
    # conventional: aspect ratio 1.0 (10x10m); manufactured home: aspect ratio
    # 4.0 (20x5m)
    geoms = [
        box(BX, BY, BX + 10, BY + 10),  # SF
        box(BX + 20, BY, BX + 40, BY + 5),  # MFG
        box(BX + 50, BY, BX + 70, BY + 5),  # MFG
        box(BX + 80, BY, BX + 90, BY + 10),  # SF
    ]

    df = gpd.GeoDataFrame(
        {
            'geometry': geoms,
            'use_group_combined_parcel': [
                'Single-Family',
                pd.NA,
                pd.NA,
                'Single-Family',
            ],
            'parcel_id_local': ['p1', 'p2', 'p3', 'p4'],
        },
        geometry='geometry',
        crs=CRS,
    )

    # Too few labels per class -> falls back to rule-based morphology.
    out_state = classify_manufactured_homes(
        _state(df),
        ruleset='kw.csv',
        model_type='calibrated_logistic',
        min_training_samples=10,
    )

    curated = out_state.curated
    assert 'p_manufactured_home' in curated.columns

    # Elongated boxes (index 1, 2) read as manufactured-home-like; squares do
    # not.
    assert curated.loc[1, 'p_manufactured_home'] > 0.5
    assert curated.loc[0, 'p_manufactured_home'] < 0.5


def test_fallback_area_score_grades_continuously_without_a_plateau(monkeypatch):
    _patch(monkeypatch)
    # Same aspect ratio (4.0), different small areas: 12x3 = 36 m2 vs
    # 20x5 = 100 m2. The retired /100 denominator saturated the area term
    # at 0.5 for everything under ~150 m2, so these two scored identically
    # and p >= 0.5 fired on 17-58% of all footprints per county. Graded
    # over the full plausible range, the smaller building must score
    # strictly higher.
    geoms = [
        box(BX, BY, BX + 12, BY + 3),
        box(BX + 50, BY, BX + 70, BY + 5),
    ]
    df = gpd.GeoDataFrame(
        {
            'geometry': geoms,
            'use_group_combined_parcel': [pd.NA, pd.NA],
            'parcel_id_local': ['p1', 'p2'],
        },
        geometry='geometry',
        crs=CRS,
    )
    out = classify_manufactured_homes(
        _state(df), ruleset='kw.csv', min_training_samples=10
    ).curated
    p_small = out.loc[0, 'p_manufactured_home']
    p_large = out.loc[1, 'p_manufactured_home']
    assert p_small > p_large


def test_classify_manufactured_homes_local_ml(monkeypatch):
    _patch(monkeypatch)
    geoms = []
    labels = []
    parcels = []

    # 10 SF (squares)
    for i in range(10):
        geoms.append(box(BX + i * 100, BY, BX + i * 100 + 10, BY + 10))
        labels.append('Single-Family')
        parcels.append(f'sf_p{i}')

    # 10 MFG (rectangles)
    for i in range(10):
        geoms.append(box(BX + i * 100, BY + 200, BX + i * 100 + 20, BY + 205))
        labels.append('Manufactured Home')
        parcels.append(f'mfg_p{i}')

    # 2 unlabeled queries placed next to their clusters (so they share the same
    # neighborhood context as training and the model separates them on shape).
    geoms.append(box(BX + 1000, BY, BX + 1010, BY + 10))
    labels.append(None)
    parcels.append('unlabeled_sf')

    geoms.append(box(BX + 1000, BY + 200, BX + 1020, BY + 205))
    labels.append(None)
    parcels.append('unlabeled_mfg')

    df = gpd.GeoDataFrame(
        {
            'geometry': geoms,
            'use_group_combined_parcel': labels,
            'parcel_id_local': parcels,
        },
        geometry='geometry',
        crs=CRS,
    )

    out_state = classify_manufactured_homes(
        _state(df),
        ruleset='kw.csv',
        model_type='calibrated_logistic',
        min_training_samples=10,
    )

    curated = out_state.curated

    # Unlabeled SF (index 20) low, unlabeled MFG (index 21) high.
    assert curated.loc[20, 'p_manufactured_home'] < 0.3
    assert curated.loc[21, 'p_manufactured_home'] > 0.7


def test_update_occupancy_false_leaves_occupancy_untouched(monkeypatch):
    _patch(monkeypatch)
    df = gpd.GeoDataFrame(
        {
            'geometry': [box(BX + 20, BY, BX + 40, BY + 5)],  # elongated
            'occupancy_type': ['Single-Family'],
            'use_group_combined_parcel': [pd.NA],
            'parcel_id_local': ['p1'],
        },
        geometry='geometry',
        crs=CRS,
    )
    out = classify_manufactured_homes(_state(df), ruleset='kw.csv').curated
    # Default update_occupancy=False: evidence is emitted, occupancy_type is not
    # touched (resolve_by_vote owns that decision).
    assert 'p_manufactured_home' in out.columns
    assert out['occupancy_type'].iloc[0] == 'Single-Family'


def test_candidate_gate_excludes_large_and_nonresidential(monkeypatch, recwarn):
    _patch(monkeypatch)
    # Row 0: large commercial -> non-candidate (fails size and residential).
    # Row 1: small residential, elongated -> candidate scored by morphology.
    df = gpd.GeoDataFrame(
        {
            'geometry': [
                box(BX, BY, BX + 60, BY + 60),  # 3600 m2, commercial
                box(BX + 200, BY, BX + 220, BY + 5),  # 100 m2, aspect 4
            ],
            'occupancy_type': ['Retail', 'Single-Family'],
            'use_group_combined_parcel': [pd.NA, pd.NA],
            'parcel_id_local': ['p0', 'p1'],
        },
        geometry='geometry',
        crs=CRS,
    )
    out = classify_manufactured_homes(_state(df), ruleset='kw.csv').curated

    # Non-candidate keeps the default "not a manufactured home" evidence.
    assert out.loc[0, 'p_manufactured_home'] == 0.0
    # Candidate is scored by morphology (elongated small footprint).
    assert out.loc[1, 'p_manufactured_home'] > 0.5
    # Reprojection to a metric CRS means no geographic-CRS distance warning.
    assert not any('geographic CRS' in str(w.message) for w in recwarn.list)
