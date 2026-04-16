"""Default category-color mappings for openplaces standard columns.

Keys are label strings as stored in parquet files after ingestion
(i.e. post-labels-CSV remapping, not raw source codes).
"""

# building.openplaces_group
# Standardized building use groups, as defined in the purpose-subgroup-remap
# CSV. Covers NSI HAZUS occupancy classes; additional categories from other
# datasets (FEMA, OBM, etc.) may be added with consistent color shading.
_OPENPLACES_GROUP = {
    # Residential — single family
    'Single Family': '#E07850',
    # Residential — other
    'Manufactured': '#D4A830',
    'Multi Family': '#9070C8',
    'Hotel': '#E8D09A',
    'Institutional Dormitory': '#E8C880',
    'Nursing Home': '#F0D8A8',
    # Commercial
    'Retail': '#C55A11',
    'Wholesale': '#B85010',
    'Personal & Repair Services': '#D06820',
    'Professional Technical Services': '#D07830',
    'Bank': '#A84000',
    'Hospital': '#C86050',
    'Medical Office': '#D07060',
    'Entertainment/Recreation': '#D88040',
    'Theater': '#C87040',
    'Garage': '#A06030',
    # Industrial
    'Heavy Industrial': '#606060',
    'Light Industrial': '#787878',
    'Food/Drug/Chemical': '#909090',
    'Metals/Minerals processing': '#505050',
    'High Technology': '#A0A0A0',
    'Construction': '#B0A090',
    # Agricultural
    'Agricultural': '#D0A100',
    # Government
    'Government Services': '#00B8E0',
    'Emergency Response': '#0090B8',
    # Education
    'Average School': '#A090D8',
    'College/University': '#B8A0E8',
    # Religious
    'Church': '#F680CB',
}

# building.purpose_group  (NSI: RES / COM / IND / PUB, post-label remapping)
_PURPOSE_GROUP = {
    'Residential': '#EF643F',
    'Commercial': '#C55A11',
    'Industrial': '#808080',
    'Public': '#7B9FD4',
}

# building.source  (NSI, post-label remapping from source-labels CSV)
# Green for highest-quality (Parcel), grading to grey for legacy/fallback,
# matching the ordered categorical priority set in the source-labels CSV.
_SOURCE = {
    'Parcel': '#5a9e6f',
    'National Center for Education Statistics': '#7a9ec8',
    'HIFLD Nursing Home': '#9eb8d8',
    'HIFLD Hospital': '#8aafd0',
    'ESRI': '#e68a3c',
    'HAZUS/NSI-2015': '#b0b0b0',
}

# Public registry  {column_name: {label: color}}
CATEGORY_COLORS = {
    'openplaces_group': _OPENPLACES_GROUP,
    'purpose_group': _PURPOSE_GROUP,
    'source': _SOURCE,
}


def match_palette(values, col_name=None, weights=None, threshold=0.5):
    """Return the best-matching color palette for a set of category values.

    Tries matches in order:

    1. Exact column-name key lookup.
    2. Any palette key is a substring of the column name.
    3. Frequency-weighted coverage: fraction of total weight (row count)
       whose category label has a defined color, using ``weights`` if
       provided, otherwise falling back to unweighted unique-label coverage.

    Parameters
    ----------
    values : iterable
        The category labels present in the plot (e.g. ``widths.columns``).
    col_name : str, optional
        Column name hint; checked first for an exact or substring key match.
    weights : array-like, optional
        Total weight (e.g. row count) for each value in ``values``, in the
        same order. When provided, coverage is computed as the fraction of
        total weight that falls in palette-covered categories, making the
        match robust to many rare/uncovered categories.
    threshold : float
        Minimum coverage required for a value-based match.

    Returns
    -------
    dict or None
        Matching ``{label: color}`` palette, or ``None`` if no good match.
    """
    str_values = [str(v) for v in values]

    def _coverage(palette):
        palette_keys = set(palette)
        if weights is not None:
            total = sum(weights)
            return (
                sum(w for v, w in zip(str_values, weights) if v in palette_keys) / total
            )
        return sum(1 for v in str_values if v in palette_keys) / len(str_values)

    # 1. Exact column name match
    if col_name in CATEGORY_COLORS:
        return CATEGORY_COLORS[col_name]
    # 2. Any palette key is a substring of the column name — only if values match
    if col_name:
        for key, palette in CATEGORY_COLORS.items():
            if key in col_name and _coverage(palette) >= threshold:
                return palette
    # 3. Frequency-weighted (or unweighted) value coverage
    best, best_score = None, 0.0
    for palette in CATEGORY_COLORS.values():
        score = _coverage(palette)
        if score > best_score:
            best_score, best = score, palette
    if best_score >= threshold:
        return best
    return None
