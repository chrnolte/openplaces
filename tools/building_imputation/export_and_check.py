"""Export delivery bundles that impute building attributes, and check them.

For each region, runs `export_delivery` and reports, for primary
buildings: the share of each primary column that is filled, the
sources of year built and structure value, the number of year or value
cells still sourced from NSI (must be 0), and any column name that
mentions a land-value region. With --restricted-notes, every bundle
file is also checked against a restricted source's column-notes CSV
(raw, canonical and canonical-plus-suffix names), so a source a newer
checkout marks restricted cannot leave in an older one's export.

With --capture, the inputs of each public bundle's imputation are saved
beside its cached bundle ('loco-<version>/'), for the leave-county-out
tool to rebuild the exact model.

Usage: pass the region ids (cheer-eastern-nc cheer-eastern-nc.team
cheer-coastal-tx), optionally with --capture and --restricted-notes.
"""

from __future__ import annotations

import argparse
import json
import re
import time

import pandas as pd
import pyarrow.parquet as pq

from openplaces.io import building_imputation as bi
from openplaces.io.delivery import delivery_paths, export_delivery, imputation_spec
from openplaces.recipe import get_recipe_by_id

RECIPE = 'US_footprint-openplaces-2026'
PRIMARY = [
    'area_m2',
    'year_built',
    'n_stories',
    'n_dwellings',
    'structure_value',
    'structure_value_adjusted',
    'appraisal_year',
    'construction_type',
    'foundation_type',
    'roof_shape',
    'has_garage',
]


def capture_dir(region: str, version: str):
    canonical = delivery_paths(RECIPE, region=region, output_dir='cache')['canonical']
    return canonical.parent / f'loco-{version}'


def install_capture(target):
    """Wrap impute_delivery so the next public export saves its inputs."""
    real = bi.impute_delivery

    def capture(
        pooled,
        spec,
        appraisal_years,
        hpi,
        county_area_km2,
        log=print,
        location=None,
        census=None,
    ):
        target.mkdir(parents=True, exist_ok=True)
        pooled.to_parquet(target / 'pooled.parquet')
        if location is not None:
            location.to_parquet(target / 'location.parquet')
        for key, table in (census or {}).items():
            table.to_parquet(target / f'census_{key}.parquet')
        appraisal_years.to_csv(target / 'appraisal_years.csv', index=False)
        hpi.to_parquet(target / 'hpi.parquet')
        county_area_km2.rename('km2').to_frame().to_parquet(
            target / 'county_area.parquet'
        )
        (target / 'spec.json').write_text(json.dumps(spec, default=str))
        print(f'saved imputation inputs to {target}', flush=True)
        return real(
            pooled,
            spec,
            appraisal_years,
            hpi,
            county_area_km2,
            log=log,
            location=location,
            census=census,
        )

    bi.impute_delivery = capture
    return real


def restricted_names(notes_path) -> set[str]:
    notes = pd.read_csv(notes_path)
    names = set(notes['name']) | set(notes['canonical_name'])
    for _, row in notes.iterrows():
        if isinstance(row.get('suffix'), str) and row['suffix']:
            names.add(f'{row["canonical_name"]}_{row["suffix"]}')
    return names


def check_bundle(paths: dict, restricted: set[str]):
    for role, path in paths.items():
        if path.suffix != '.parquet':
            text = path.read_text(encoding='utf-8').lower()
            print(f'  {path.name}: {len(text):,d} characters')
            continue
        columns = pq.read_schema(path).names
        hits = [
            c
            for c in columns
            if c in restricted or c.removesuffix('_source') in restricted
        ]
        regions = [c for c in columns if re.search('region', c)]
        print(
            f'  {path.name}: {len(columns)} columns; '
            f'restricted-name matches {hits}; region columns {regions}'
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('regions', nargs='+')
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--restricted-notes')
    parser.add_argument(
        '--output-dir',
        help="STANDARD_DIRS name to write into, e.g. 'cache' for a trial "
        'that must not replace the shipped bundle',
    )
    args = parser.parse_args()
    restricted = (
        restricted_names(args.restricted_notes) if args.restricted_notes else set()
    )
    recipe = get_recipe_by_id(RECIPE)
    t0 = time.time()
    real = None
    for region in args.regions:
        spec = imputation_spec(recipe, region)
        if real is not None:
            bi.impute_delivery = real
            real = None
        if args.capture and spec and not region.endswith('.team'):
            real = install_capture(capture_dir(region, spec['version']))
        print(f'[{time.time() - t0:6.0f}s] export {region}', flush=True)
        paths = export_delivery(RECIPE, region=region, output_dir=args.output_dir)
        table = pd.read_parquet(paths['canonical'])
        primary = table['occupancy_type'].astype('string').ne('Secondary')
        print(f'{region}: {len(table):,d} rows, {int(primary.sum()):,d} primary')
        present = [c for c in PRIMARY if c in table.columns]
        print(table.loc[primary, present].notna().mean().round(4).to_string())
        for column in ('year_built_source', 'structure_value_source'):
            counts = table.loc[primary, column].astype('string').value_counts()
            print(column, counts.head(6).to_dict())
        sources = table[['year_built_source', 'structure_value_source']].fillna('')
        nsi = sum(int(bi.is_nsi_token(sources[c]).sum()) for c in sources)
        print(f'NSI-sourced year or value cells: {nsi}')
        check_bundle(paths, restricted)
    if real is not None:
        bi.impute_delivery = real
    print(f'[{time.time() - t0:6.0f}s] done')


if __name__ == '__main__':
    main()
