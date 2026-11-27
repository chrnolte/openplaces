"""Link point records to parcels and buildings, and build reference labels.

A reference used to validate a classification (building permits, a
registry of licensed homes) arrives as one row per record, each carrying
a point, often a parcel number, and a street address, and each naming a
class. Scoring a curated entity against it needs the records reduced to
one label per parcel and fanned out to the footprints on that parcel.
:func:`link_records_to_buildings` instead pairs each record with the
buildings it concerns (point in outline first, then the parcel's
buildings). Nothing here names a source, a class, or a geography:
callers pass the tables and column names.

**How a record reaches a parcel.** Three deterministic rules, tried in a
fixed order, each only on the records the previous ones left:

1. `point`: the record's point lies inside the parcel polygon. Where
   polygons overlap (stacked condominium parcels), the smallest
   containing polygon wins, ties broken by parcel id, so the pick never
   depends on row order.
2. `parcel_id_local` (or another key column): the record's standardized
   parcel number equals the parcel's, counting only keys held by exactly
   one parcel.
3. `address`: the record's street and house number, normalized exactly
   as :func:`~openplaces.io.harmonizer.addresses.add_address_id_local`
   keys spine addresses, equal one parcel's, again unique keys only.

The point comes first because it is the only one of the three that does
not depend on two publishers agreeing on an identifier format.
Measured 2026-09-16 on cheer-eastern-nc: across its 44 eastern North
Carolina counties, exact parcel-number equality linked no permits at all
in 38 of them, while 94-96% of permit points lay within 50 m of a
footprint.

Why this shape and not a scored match: each link is decided by one
exact rule, the rule is recorded in `matched_via` as a label rather than
a strength score, nothing is learned from the links made, and no link is
ever revisited or removed. There is no fuzzy string comparison. The
module deliberately has no fall-through to fuzzy matching, no link
strength score, no scorer recalibrated from its own links and no
unlinking step, and a change adding one of those parts is a design
decision for the maintainer (see "Design limits" in AGENTS.md).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

#: Link rules in priority order; also the `matched_via` labels.
LINK_POINT = 'point'
LINK_ADDRESS = 'address'

#: Record-to-building rules (`link_method` labels, before the
#: ", several buildings" suffix), see :func:`link_records_to_buildings`.
LINK_FOOTPRINT = 'footprint'
LINK_PARCEL = 'parcel'

_ADDRESS_KEY = '_address_key'


def _normalized_key(values: pd.Series) -> pd.Series:
    """Upper-case alphanumeric form of an identifier, missing if empty."""
    text = values.astype('string').str.upper().str.replace(r'[^0-9A-Z]', '', regex=True)
    return text.where(text.ne('') & text.notna())


def _unique_key_map(keys: pd.Series) -> pd.Series:
    """Map each key held by exactly one row to that row's index label."""
    keys = keys.dropna()
    counts = keys.value_counts()
    unique = keys[keys.map(counts).eq(1)]
    return pd.Series(unique.index, index=unique.values)


def _planar_area(polygons: gpd.GeoSeries) -> np.ndarray:
    """Polygon areas in a projected CRS, for ranking overlaps only.

    Areas in degrees shrink with latitude and are skewed by aspect, so a
    geographic layer is measured in its local UTM zone instead.
    """
    if polygons.crs is not None and polygons.crs.is_geographic and len(polygons):
        polygons = polygons.to_crs(polygons.estimate_utm_crs())
    return polygons.area.to_numpy()


def _link_by_point(records, parcels) -> pd.Series:
    """Polygon id containing each record's point (smallest polygon wins)."""
    out = pd.Series(pd.NA, index=records.index, dtype=object)
    geometry = records.geometry
    has_point = geometry.notna() & ~geometry.is_empty
    if not has_point.any() or parcels.empty:
        return out
    points = gpd.GeoDataFrame(
        {'_row': np.flatnonzero(has_point.to_numpy())},
        geometry=geometry[has_point].to_numpy(),
        crs=records.crs,
    )
    if parcels.crs is not None and points.crs != parcels.crs:
        points = points.to_crs(parcels.crs)
    polygons = gpd.GeoDataFrame(
        {
            '_parcel': parcels.index.to_numpy(),
            '_area': _planar_area(parcels.geometry),
        },
        geometry=parcels.geometry.to_numpy(),
        crs=parcels.crs,
    )
    hits = gpd.sjoin(points, polygons, how='inner', predicate='within')
    if hits.empty:
        return out
    hits = hits.sort_values(['_row', '_area', '_parcel'], kind='stable')
    hits = hits.drop_duplicates('_row', keep='first')
    out.iloc[hits['_row'].to_numpy()] = hits['_parcel'].to_numpy()
    return out


def link_records_to_parcels(
    records: gpd.GeoDataFrame | pd.DataFrame,
    parcels: gpd.GeoDataFrame | pd.DataFrame,
    *,
    key_columns: tuple[tuple[str, str], ...] = (
        ('parcel_id_local', 'parcel_id_local'),
    ),
    record_address: dict | None = None,
    parcel_address: dict | None = None,
    admin_id=None,
) -> pd.DataFrame:
    """Link each record to at most one parcel, recording the rule used.

    Parameters
    ----------
    records : geopandas.GeoDataFrame or pandas.DataFrame
        One row per record. Its geometry (points) is used when present.
    parcels : geopandas.GeoDataFrame or pandas.DataFrame
        One row per parcel, indexed by parcel id. Its polygons are used
        when present.
    key_columns : tuple of (str, str) pairs
        (record column, parcel column) identifier pairs, tried in order
        after the point rule. Values are compared in upper-case
        alphanumeric form, and only keys unique among parcels link. The
        label written to `matched_via` is the parcel column's name.
    record_address, parcel_address : dict, optional
        Keyword arguments of
        :func:`~openplaces.io.harmonizer.addresses.add_address_id_local`
        naming each side's street and number columns. The address rule
        runs only when both are given; the town scope is dropped on both
        sides, so the key is scoped by the unit the tables cover.
    admin_id : str or AdminId, optional
        Admin unit the tables cover; selects street normalization rules.

    Returns
    -------
    pandas.DataFrame
        Indexed like *records*, with `parcel_id` (missing where no rule
        linked) and `matched_via` (`point`, a key column name, `address`,
        or missing).
    """
    parcel_id = pd.Series(pd.NA, index=records.index, dtype=object)
    matched_via = pd.Series(pd.NA, index=records.index, dtype=object)

    def _take(candidates: pd.Series, label: str) -> None:
        fill = parcel_id.isna() & candidates.notna()
        parcel_id[fill] = candidates[fill]
        matched_via[fill] = label

    if isinstance(records, gpd.GeoDataFrame) and isinstance(parcels, gpd.GeoDataFrame):
        _take(_link_by_point(records, parcels), LINK_POINT)

    for record_column, parcel_column in key_columns:
        if record_column not in records or parcel_column not in parcels:
            continue
        todo = parcel_id.isna()
        if not todo.any():
            break
        lookup = _unique_key_map(_normalized_key(parcels[parcel_column]))
        keys = _normalized_key(records.loc[todo, record_column])
        _take(keys.map(lookup).reindex(records.index), parcel_column)

    if record_address and parcel_address and parcel_id.isna().any():
        _take(
            _link_by_address(
                records[parcel_id.isna()],
                parcels,
                record_address,
                parcel_address,
                admin_id,
            ).reindex(records.index),
            LINK_ADDRESS,
        )

    return pd.DataFrame({'parcel_id': parcel_id, 'matched_via': matched_via})


def link_records_to_buildings(
    records: gpd.GeoDataFrame | pd.DataFrame,
    footprints: gpd.GeoDataFrame,
    buildings: pd.DataFrame,
    parcels: gpd.GeoDataFrame | pd.DataFrame,
    *,
    key_columns: tuple[tuple[str, str], ...] = (
        ('parcel_id_local', 'parcel_id_local'),
    ),
    record_address: dict | None = None,
    parcel_address: dict | None = None,
    admin_id=None,
) -> pd.DataFrame:
    """Link records to the buildings they concern, one row per pair.

    A record (a building permit) names an address, a parcel number and a
    geocoded point, almost never a structure. Two exact rules reach
    buildings, the second only on records the first left:

    1. `footprint`: the record's point lies inside a building outline
       (the smallest, where outlines overlap) that has buildings; it
       links to each building under that outline, labeled `footprint`
       for one and `footprint, several buildings` for more (a townhome
       row drawn as one outline).
    2. `parcel`: :func:`link_records_to_parcels` reaches a parcel, and
       the record links to each building whose `parcel_id` it is,
       labeled `parcel` for one and `parcel, several buildings` for
       more. A record on a parcel without buildings links nothing.

    Why every building and not a chosen one: the record does not say
    which structure it concerns, and picking one (the largest, the main
    house) would be a guess presented as a link. The label tells a
    reader where a record fans out, so a reader that needs one building
    per record leaves the "several" labels out. A point outside every
    outline is not moved to the nearest one: a distance tolerance behind
    the exact rule would be a fall-through tier, which the module rules
    out (see the module docstring); such points still reach buildings
    through their parcel.

    Measured 2026-11-15 on US-MA: 93.0% of 1,330,580 permits reach one building.

    Parameters
    ----------
    records : geopandas.GeoDataFrame or pandas.DataFrame
        One row per record. Its geometry (points) is used when present.
    footprints : geopandas.GeoDataFrame
        Building outlines, indexed by `footprint_id`.
    buildings : pandas.DataFrame
        One row per building, indexed by building id, with `footprint_id`
        and `parcel_id` (either may be missing).
    parcels : geopandas.GeoDataFrame or pandas.DataFrame
        One row per parcel, indexed by parcel id, as for
        :func:`link_records_to_parcels`.
    key_columns, record_address, parcel_address, admin_id : optional
        Passed to :func:`link_records_to_parcels`.

    Returns
    -------
    pandas.DataFrame
        One row per record-building pair, with the record's index label
        (column named after *records*' index, `record_id` if unnamed),
        `building_id` and `link_method`. A record no rule reaches has no
        row.
    """
    record_col = records.index.name or 'record_id'
    pairs = []

    def _fan_out(container: pd.Series, by: str, label: str) -> pd.Index:
        """Pair each record with the buildings sharing its container id."""
        members = buildings[[by]].dropna()
        members = pd.DataFrame(
            {by: members[by].to_numpy(), 'building_id': members.index.to_numpy()}
        )
        found = (
            container.dropna()
            .rename(by)
            .rename_axis(record_col)
            .reset_index()
            .merge(members, on=by, how='inner')
        )
        if found.empty:
            return pd.Index([])
        n = found.groupby(record_col)['building_id'].transform('size')
        found['link_method'] = np.where(n > 1, f'{label}, several buildings', label)
        pairs.append(found[[record_col, 'building_id', 'link_method']])
        return pd.Index(found[record_col].unique())

    done = pd.Index([])
    if isinstance(records, gpd.GeoDataFrame) and not footprints.empty:
        done = _fan_out(
            _link_by_point(records, footprints), 'footprint_id', LINK_FOOTPRINT
        )

    rest = records.loc[~records.index.isin(done)]
    if len(rest):
        links = link_records_to_parcels(
            rest,
            parcels,
            key_columns=key_columns,
            record_address=record_address,
            parcel_address=parcel_address,
            admin_id=admin_id,
        )
        _fan_out(links['parcel_id'], 'parcel_id', LINK_PARCEL)

    if not pairs:
        return pd.DataFrame(columns=[record_col, 'building_id', 'link_method'])
    return pd.concat(pairs, ignore_index=True)


def _address_keys(frame, spec, admin_id) -> pd.Series | None:
    from openplaces.io.harmonizer.addresses import add_address_id_local

    spec = {
        **dict(spec),
        'admin4_column': None,
        'city_column': None,
        'output_column': _ADDRESS_KEY,
        'output_column_city': None,
    }
    street = spec.get('street_column', 'address_street')
    number = spec.get('number_column', 'address_number')
    if street not in frame or number not in frame:
        return None
    keyed = add_address_id_local(frame[[street, number]].copy(), admin_id, **spec)
    return keyed[_ADDRESS_KEY]


def _link_by_address(records, parcels, record_spec, parcel_spec, admin_id):
    record_keys = _address_keys(records, record_spec, admin_id)
    parcel_keys = _address_keys(parcels, parcel_spec, admin_id)
    if record_keys is None or parcel_keys is None:
        return pd.Series(pd.NA, index=records.index, dtype=object)
    return record_keys.map(_unique_key_map(parcel_keys))


def summarize_labels_by_parcel(
    records: pd.DataFrame,
    links: pd.DataFrame,
    *,
    class_column: str,
    date_columns: tuple[str, ...] = (),
    year_built_column: str | None = None,
    area_column: str | None = None,
    address_columns: tuple[str, ...] = (),
    label: str = 'occupancy_type',
    link_order: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Reduce linked records to one reference label row per parcel.

    Parameters
    ----------
    records : pandas.DataFrame
        One row per record.
    links : pandas.DataFrame
        Output of :func:`link_records_to_parcels` for *records*.
    class_column : str
        Column naming each record's class.
    date_columns : tuple of str
        Date columns in order of preference; the first non-missing one
        dates a record and orders "most recent".
    year_built_column, area_column : str, optional
        Numeric columns reported as most recent (and median, for area).
    address_columns : tuple of str
        Columns joined with spaces into the most recent record's address.
    label : str
        Prefix of the class summary columns.
    link_order : tuple of str
        `matched_via` labels from strongest to weakest, for breaking a
        tie in the parcel's majority rule.

    Returns
    -------
    pandas.DataFrame
        Indexed by `parcel_id`, with `address`, `{label}_most_recent`,
        `{label}_mode`, `{label}_mode_pct`, `year_built_most_recent`,
        `area_sqft_most_recent`, `area_sqft_median`, `n_permits`,
        `n_permits_with_{label}`, `latest_permit_date`, and
        `matched_via`: the rule that linked most label-bearing records
        (the strongest on a tie), `none` where no linked record names a
        class.
    """
    frame = pd.DataFrame(index=records.index)
    frame['parcel_id'] = links['parcel_id']
    frame['matched_via'] = links['matched_via']
    frame['cls'] = (
        records[class_column].astype(object).where(records[class_column].notna(), None)
    )
    date = pd.Series(pd.NaT, index=records.index, dtype='datetime64[ns]')
    for column in date_columns:
        if column in records:
            parsed = pd.to_datetime(records[column], errors='coerce', format='mixed')
            parsed = parsed.dt.tz_localize(None) if parsed.dt.tz else parsed
            date = date.fillna(parsed.astype('datetime64[ns]'))
    frame['date'] = date
    for out, column in (('year_built', year_built_column), ('area', area_column)):
        frame[out] = (
            pd.to_numeric(records[column], errors='coerce')
            if column and column in records
            else np.nan
        )
    parts = [
        records[c].astype('string').fillna('') for c in address_columns if c in records
    ]
    if parts:
        address = parts[0]
        for part in parts[1:]:
            address = address.str.cat(part, sep=' ')
        frame['address'] = address.str.split().str.join(' ').replace('', pd.NA)
    else:
        frame['address'] = pd.NA

    frame = frame[frame['parcel_id'].notna()]
    frame = frame.sort_values('date', kind='stable', na_position='first')
    grouped = frame.groupby('parcel_id', sort=False)
    out = pd.DataFrame(index=pd.Index(grouped.size().index, name='parcel_id'))
    out['address'] = grouped['address'].last()
    typed = frame[frame['cls'].notna()]
    typed_groups = typed.groupby('parcel_id', sort=False)
    out[f'{label}_most_recent'] = typed_groups['cls'].last()

    counts = typed.groupby(['parcel_id', 'cls'], sort=False).agg(
        n=('cls', 'size'), last=('date', 'max')
    )
    counts = counts.reset_index().sort_values(
        ['parcel_id', 'n', 'last', 'cls'],
        ascending=[True, False, False, True],
        kind='stable',
        na_position='last',
    )
    top = counts.drop_duplicates('parcel_id').set_index('parcel_id')
    n_typed = typed_groups.size()
    out[f'{label}_mode'] = top['cls']
    out[f'{label}_mode_pct'] = top['n'] / n_typed
    out['year_built_most_recent'] = grouped['year_built'].last()
    out['area_sqft_most_recent'] = grouped['area'].last()
    out['area_sqft_median'] = grouped['area'].median()
    out['n_permits'] = grouped.size().astype('float64')
    out[f'n_permits_with_{label}'] = n_typed.reindex(out.index).fillna(0)
    out['latest_permit_date'] = grouped['date'].max()

    rank = {name: i for i, name in enumerate(link_order)}
    via = typed.groupby(['parcel_id', 'matched_via'], sort=False).size()
    via = via.rename('n').reset_index()
    via['rank'] = via['matched_via'].map(rank).fillna(len(rank))
    via = via.sort_values(
        ['parcel_id', 'n', 'rank'], ascending=[True, False, True], kind='stable'
    ).drop_duplicates('parcel_id')
    out['matched_via'] = via.set_index('parcel_id')['matched_via']
    out['matched_via'] = out['matched_via'].astype(object).fillna('none')
    return out


REFERENCE_COLUMNS = (
    'parcel_id_local',
    'address',
    'occupancy_type_most_recent',
    'occupancy_type_mode',
    'occupancy_type_mode_pct',
    'year_built_most_recent',
    'area_sqft_most_recent',
    'area_sqft_median',
    'n_permits',
    'n_permits_with_occupancy_type',
    'latest_permit_date',
    'matched_via',
)


def build_parcel_reference_files(
    admin_id,
    out_dir,
    *,
    record_recipe_id: str,
    parcel_recipe_id: str = 'US_parcel-geospine-2026',
    parcel_address_recipe_id: str | None = 'US_parcel-spine-2026',
    building_recipe_id: str = 'US_building-spine-2026',
    class_column: str = 'occupancy_type_raw',
    record_key_columns: tuple[tuple[str, str], ...] = (
        ('parcel_id_local', 'parcel_id_local'),
        ('parcel_id_alnum', 'parcel_id_alnum'),
    ),
    record_address: dict | None = None,
    parcel_address: dict | None = None,
    date_columns: tuple[str, ...] = (
        'permit_file_date',
        'permit_issue_date',
        'permit_start_date',
    ),
    year_built_column: str = 'year_built',
    area_column: str = 'area_sqft',
    address_columns: tuple[str, ...] = ('street_no', 'street', 'city'),
    name: str = 'occupancy',
) -> dict:
    """Write one admin unit's parcel and footprint reference label files.

    Writes `{admin_id}_parcel_{name}_validation.parquet` (every parcel,
    with a label row where records linked) and
    `{admin_id}_footprint_{name}_validation.parquet` (every footprint,
    carrying its parcel's labels) into *out_dir*, the schema the
    validation notebooks read. The reference may be licence-restricted:
    *out_dir* must be a private location, never a delivery or share
    directory.

    A footprint's parcel is its buildings' (b3 step 6): the building
    spine holds each building's `parcel_id` and the `footprint_id` of
    the outline it stands under. The file stays footprint-level, as the
    inventory CHEER ships is. An outline whose buildings name different
    parcels (a townhome row cut at parcel lines) takes none, so it is
    scored as unlabeled rather than on a parcel chosen for it; while one
    building is one outline every outline takes its building's.

    Parameters
    ----------
    admin_id : str
        Admin unit to build.
    out_dir : str or pathlib.Path
        Directory to write into (created if missing).
    record_recipe_id : str
        Ingest recipe of the records (read with geometry).
    parcel_recipe_id : str
        Recipe whose output holds the parcel polygons and id keys.
    parcel_address_recipe_id : str, optional
        Recipe holding reconciled parcel addresses on the same parcel
        index (the geospine's attribute recipe); None skips the address
        rule.
    building_recipe_id : str
        Recipe holding each building's `parcel_id` and `footprint_id`.
    class_column : str
        Record column naming its class.
    record_key_columns : tuple of (str, str) pairs
        Identifier pairs for the key rule, see
        :func:`link_records_to_parcels`.
    record_address, parcel_address : dict, optional
        Street and number column names for the address rule; default to
        `street`/`street_no` on records and
        `address_street`/`address_number` on parcels.
    date_columns, year_built_column, area_column, address_columns : optional
        Record columns, see :func:`summarize_labels_by_parcel`.
    name : str
        File name component (`occupancy` gives the notebook schema).

    Returns
    -------
    dict
        Aggregate counts: records, records linked per rule, parcels and
        footprints with labels.
    """
    from openplaces.io.readers import get_entities

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_address = record_address or {
        'street_column': 'street',
        'number_column': 'street_no',
    }
    parcel_address = parcel_address or {
        'street_column': 'address_street',
        'number_column': 'address_number',
    }

    records = get_entities(record_recipe_id, admin_id, geom=True, missing='ignore')
    parcels = get_entities(parcel_recipe_id, admin_id, geom=True)
    if parcel_address_recipe_id:
        address_cols = [
            parcel_address['street_column'],
            parcel_address['number_column'],
        ]
        addresses = get_entities(
            parcel_address_recipe_id, admin_id, columns=address_cols, missing='ignore'
        )
        if addresses is not None and not addresses.empty:
            parcels = parcels.drop(
                columns=[c for c in address_cols if c in parcels.columns]
            ).join(addresses[[c for c in address_cols if c in addresses]])
    buildings = get_entities(
        building_recipe_id, admin_id, columns=['parcel_id', 'footprint_id']
    )
    by_outline = buildings.dropna(subset=['footprint_id']).groupby('footprint_id')[
        'parcel_id'
    ]
    footprints = pd.DataFrame(
        {'parcel_id': by_outline.first().where(by_outline.nunique() <= 1)}
    )

    summary = {'admin_id': str(admin_id), 'records': 0}
    label = 'occupancy_type' if name == 'occupancy' else name
    if records is None or records.empty:
        per_parcel = pd.DataFrame(
            columns=[c for c in REFERENCE_COLUMNS if c != 'parcel_id_local']
        )
        per_parcel.index.name = 'parcel_id'
    else:
        summary['records'] = len(records)
        links = link_records_to_parcels(
            records,
            parcels,
            key_columns=record_key_columns,
            record_address=record_address,
            parcel_address=parcel_address if parcel_address_recipe_id else None,
            admin_id=admin_id,
        )
        for rule, n in links['matched_via'].value_counts().items():
            summary[f'linked_{rule}'] = int(n)
        summary['linked'] = int(links['parcel_id'].notna().sum())
        link_order = (LINK_POINT,) + tuple(p for _, p in record_key_columns)
        per_parcel = summarize_labels_by_parcel(
            records,
            links,
            class_column=class_column,
            date_columns=date_columns,
            year_built_column=year_built_column,
            area_column=area_column,
            address_columns=address_columns,
            label=label,
            link_order=link_order + (LINK_ADDRESS,),
        )

    columns = [c.replace('occupancy_type', label) for c in REFERENCE_COLUMNS]
    parcel_out = pd.DataFrame(index=parcels.index.rename('parcel_id'))
    parcel_out['parcel_id_local'] = (
        parcels['parcel_id_local'] if 'parcel_id_local' in parcels else pd.NA
    )
    parcel_out = parcel_out.join(per_parcel, how='left')
    parcel_out['matched_via'] = parcel_out['matched_via'].astype(object).fillna('none')
    parcel_out = parcel_out.reindex(columns=columns)
    footprint_out = (
        footprints[['parcel_id']]
        .join(parcel_out, on='parcel_id', how='left')
        .reindex(columns=columns)
    )
    footprint_out.index.name = 'footprint_id'
    parcel_out.to_parquet(out_dir / f'{admin_id}_parcel_{name}_validation.parquet')
    footprint_out.to_parquet(
        out_dir / f'{admin_id}_footprint_{name}_validation.parquet'
    )
    summary['parcels_labeled'] = int(parcel_out['n_permits'].notna().sum())
    summary['footprints_labeled'] = int(footprint_out['n_permits'].notna().sum())
    return summary


def build_building_reference_links(
    admin_id,
    out_dir,
    *,
    record_recipe_id: str,
    footprint_recipe_id: str = 'US_footprint-geospine-2026',
    building_recipe_id: str = 'US_building-spine-2026',
    parcel_recipe_id: str = 'US_parcel-geospine-2026',
    parcel_address_recipe_id: str | None = 'US_parcel-spine-2026',
    record_key_columns: tuple[tuple[str, str], ...] = (
        ('parcel_id_local', 'parcel_id_local'),
        ('parcel_id_alnum', 'parcel_id_alnum'),
    ),
    record_address: dict | None = None,
    parcel_address: dict | None = None,
    name: str = 'permit',
) -> dict:
    """Write one admin unit's record-to-building link table.

    Writes `{admin_id}_building_{name}_links.parquet` into *out_dir*:
    the pairs of :func:`link_records_to_buildings`, with the record's
    index label, `building_id` and `link_method` and nothing else. The
    records may be licence-restricted: *out_dir* must be a private
    location, never a delivery or share directory.

    Parameters
    ----------
    admin_id : str
        Admin unit to build.
    out_dir : str or pathlib.Path
        Directory to write into (created if missing).
    record_recipe_id : str
        Ingest recipe of the records (read with geometry).
    footprint_recipe_id : str
        Recipe holding the building outlines, indexed by `footprint_id`.
    building_recipe_id : str
        Recipe holding each building's `parcel_id` and `footprint_id`.
    parcel_recipe_id, parcel_address_recipe_id : str, optional
        Parcel polygons and their reconciled addresses, as for
        :func:`build_parcel_reference_files`.
    record_key_columns, record_address, parcel_address : optional
        See :func:`link_records_to_parcels`.
    name : str
        File name component.

    Returns
    -------
    dict
        Aggregate counts: records, records linked, and records per
        `link_method`.
    """
    from openplaces.io.readers import get_entities

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_address = record_address or {
        'street_column': 'street',
        'number_column': 'street_no',
    }
    parcel_address = parcel_address or {
        'street_column': 'address_street',
        'number_column': 'address_number',
    }

    records = get_entities(record_recipe_id, admin_id, geom=True, missing='ignore')
    summary = {'admin_id': str(admin_id), 'records': 0, 'linked': 0}
    out_path = out_dir / f'{admin_id}_building_{name}_links.parquet'
    if records is None or records.empty:
        pd.DataFrame(columns=['record_id', 'building_id', 'link_method']).to_parquet(
            out_path
        )
        return summary

    parcels = get_entities(parcel_recipe_id, admin_id, geom=True)
    if parcel_address_recipe_id:
        address_cols = [
            parcel_address['street_column'],
            parcel_address['number_column'],
        ]
        addresses = get_entities(
            parcel_address_recipe_id, admin_id, columns=address_cols, missing='ignore'
        )
        if addresses is not None and not addresses.empty:
            parcels = parcels.drop(
                columns=[c for c in address_cols if c in parcels.columns]
            ).join(addresses[[c for c in address_cols if c in addresses]])
    footprints = get_entities(footprint_recipe_id, admin_id, geom=True)
    buildings = get_entities(
        building_recipe_id, admin_id, columns=['parcel_id', 'footprint_id']
    )

    pairs = link_records_to_buildings(
        records,
        footprints,
        buildings,
        parcels,
        key_columns=record_key_columns,
        record_address=record_address,
        parcel_address=parcel_address if parcel_address_recipe_id else None,
        admin_id=admin_id,
    )
    pairs.to_parquet(out_path, index=False)

    record_col = pairs.columns[0]
    per_record = pairs.drop_duplicates(record_col)
    summary['records'] = len(records)
    summary['linked'] = len(per_record)
    for method, n in per_record['link_method'].value_counts().items():
        summary[f'linked_{method}'] = int(n)
    return summary
