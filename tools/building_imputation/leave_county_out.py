"""Leave-county-out evaluation of a delivered imputation configuration.

Rebuilds the imputer with `building_imputation.build_imputer`, as the
delivery does, from the inputs the export tool's capture option saved
during a real export, fills every target once (the chained models need
the filled columns), then hides each listed county's year built and
structure value labels and predicts them from donors only, with the
county's parcel value inputs masked. Scores go to 'loco_scores.csv'
beside the inputs.

The default counties are the 14 eastern NC counties every earlier run
held out, so results stay comparable.

Usage: pass the capture folder, optionally followed by --counties and
the admin ids to hold out (US-NC-BEA US-NC-BRU ...).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from openplaces.io import building_imputation as bi

COUNTIES = [
    'US-NC-BEA',
    'US-NC-BRU',
    'US-NC-CAR',
    'US-NC-CRA',
    'US-NC-CUR',
    'US-NC-DAR',
    'US-NC-NHA',
    'US-NC-ONS',
    'US-NC-PIT',
    'US-NC-ROB',
    'US-NC-WAY',
    'US-NC-PEN',
    'US-NC-JON',
    'US-NC-PAS',
]


# Input groups an ablation can leave out.
ABLATIONS = {
    # Every census input.
    'census': tuple(bi.CENSUS_INPUTS),
    # The census inputs added after the tract home value and year.
    'settlement': tuple(
        c
        for c in bi.CENSUS_INPUTS
        if c not in ('log_acs_median_home_value_tract', 'acs_median_year_built_tract')
    ),
    # The places land-market regions: their cores rest on a population
    # raster built partly from ZTRAX land use.
    'places': ('region_type', 'distance_to_core_km'),
    # Every land-market input, the 2010 land value included.
    'location': tuple(bi.LOCATION_INPUTS),
}


def build_imputer(
    folder: Path,
    without: tuple[str, ...] = (),
    extensions: str | None = None,
    use_location: str | None = None,
):
    """The delivery's imputer, from captured inputs.

    *without* names inputs to leave out of every model, for an ablation;
    *extensions* ('module:attribute') overrides the captured recipe's;
    *use_location* (a location recipe id) adds the captured land-market
    inputs to a recipe that does not read them, to measure what leaving
    them out costs.
    """
    pooled = pd.read_parquet(folder / 'pooled.parquet')
    spec = json.loads((folder / 'spec.json').read_text())
    if use_location is not None:
        spec = {**spec, 'location_recipe': use_location}
    appraisal = pd.read_csv(folder / 'appraisal_years.csv', dtype={'county_fips': str})
    hpi = pd.read_parquet(folder / 'hpi.parquet')
    area = pd.read_parquet(folder / 'county_area.parquet')['km2']
    location_path = folder / 'location.parquet'
    # As the delivery: land-market inputs only where the recipe reads them.
    location = (
        pd.read_parquet(location_path)
        if spec.get('location_recipe') and location_path.exists()
        else None
    )
    census = {
        key: pd.read_parquet(folder / f'census_{key}.parquet')
        for key in ('tract', 'blockgroup', 'place', 'county')
        if (folder / f'census_{key}.parquet').exists()
    }
    if census:
        for key in ('tract', 'blockgroup', 'place'):
            census.setdefault(key, pd.DataFrame())
    imputer, _ = bi.build_imputer(
        pooled,
        spec,
        appraisal,
        hpi,
        area,
        location=location,
        census=census or None,
        without=without,
        extensions=extensions,
        log=lambda message: None,
    )
    return imputer, bi.delivery_config(spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--counties', nargs='+', default=COUNTIES)
    parser.add_argument(
        '--without',
        nargs='+',
        default=[],
        choices=sorted(ABLATIONS),
        help='input groups to leave out (ablation); scores get a suffix',
    )
    parser.add_argument(
        '--extensions',
        help="'module:attribute' of ImputationExtensions, or 'none'; "
        "default: the captured recipe's",
    )
    parser.add_argument(
        '--thin-heldout',
        action='store_true',
        help="give held-out rows real gap patterns (the extensions' held-out hook)",
    )
    parser.add_argument(
        '--with-location',
        metavar='RECIPE_ID',
        help='add the captured land-market inputs (location.parquet) under this '
        'location recipe id',
    )
    parser.add_argument(
        '--targets',
        choices=['per-county', 'pooled', 'all'],
        default='per-county',
        help='year built and structure value (per-county models), the other '
        'seven attributes (one regional model each), or all nine',
    )
    args = parser.parse_args()
    t0 = time.time()

    def log(message):
        print(f'[{time.time() - t0:7.0f}s] {message}', flush=True)

    without = tuple(c for group in args.without for c in ABLATIONS[group])
    suffix = ''.join(f'_without_{group}' for group in args.without)
    extensions = '' if args.extensions == 'none' else args.extensions
    if args.extensions:
        suffix += f'_ext_{(args.extensions.split(":")[-1] or "none").lower()}'
    if args.thin_heldout:
        suffix += '_thin'
    if args.with_location:
        suffix += '_with_location'
    if args.targets != 'per-county':
        suffix += f'_{args.targets}'
    imputer, config = build_imputer(
        args.folder,
        without,
        extensions,
        use_location=args.with_location,
    )
    filled, _ = imputer.run(log=lambda message: None)
    log('filled; chain inputs ready')
    specs = {spec.column: spec for spec in bi.TARGETS}
    tables = []
    if args.targets in {'pooled', 'all'}:
        for spec in bi.TARGETS:
            if spec.per_county:
                continue
            results = imputer.leave_county_out_pooled(
                spec, args.counties, filled, log=log, thin_heldout=args.thin_heldout
            )
            results.to_parquet(args.folder / f'loco_{spec.column}{suffix}.parquet')
            table = bi.score_pooled(results, spec.kind)
            table['target'] = spec.column
            tables.append(table)
    per_county = (('year_built', 'year'), ('structure_value', 'value'))
    if args.targets == 'pooled':
        per_county = ()
    for column, kind in per_county:
        results = imputer.leave_county_out(
            specs[column],
            args.counties,
            [config],
            filled,
            log=log,
            thin_heldout=args.thin_heldout,
        )
        results.to_parquet(args.folder / f'loco_{column}{suffix}.parquet')
        table = bi.score(results, kind)
        table['target'] = column
        tables.append(table)
    scores = pd.concat(tables)
    scores.round(4).to_csv(args.folder / f'loco_scores{suffix}.csv')
    # What reproduces these scores: the seed every random step derives
    # from, and the options of the run.
    record = {
        'seed': imputer.seed,
        'options': {k: str(v) for k, v in vars(args).items()},
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    (args.folder / f'loco_run{suffix}.json').write_text(json.dumps(record, indent=1))
    pd.set_option('display.width', 250)
    print(scores.round(4).to_string())
    log('done')


if __name__ == '__main__':
    main()
