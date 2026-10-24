"""Leave-county-out evaluation of a delivered imputation configuration.

Rebuilds the imputer exactly as `impute_delivery` does, from the inputs
`export_and_check.py --capture` saved during a real export, fills every
target once (the chained models need the filled columns), then hides
each listed county's year built and structure value labels and predicts
them from donors only, with the county's parcel value inputs masked.
Scores go to ``loco_scores.csv`` beside the inputs.

The default counties are the 14 eastern NC counties every earlier run
held out, so results stay comparable.

Usage
-----
    python tools/building_imputation/leave_county_out.py <capture-dir> \
        [--counties US-NC-BEA US-NC-BRU ...]
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
}


def build_imputer(
    folder: Path, without: tuple[str, ...] = (), extensions: str | None = None
):
    """The delivery's imputer, from captured inputs.

    *without* names inputs to leave out of every model, for an ablation;
    *extensions* ('module:attribute') overrides the captured recipe's.
    """
    pooled = pd.read_parquet(folder / 'pooled.parquet')
    spec = json.loads((folder / 'spec.json').read_text())
    appraisal = pd.read_csv(folder / 'appraisal_years.csv', dtype={'county_fips': str})
    hpi = pd.read_parquet(folder / 'hpi.parquet')
    area = pd.read_parquet(folder / 'county_area.parquet')['km2']
    frame = bi.delivery_frame(pooled)
    excluded = bi.UNUSED_LOCATION_INPUTS
    location_path = folder / 'location.parquet'
    if location_path.exists():
        location = pd.read_parquet(location_path).reindex(frame.index)
        frame['region_type'] = location['land_value_region_type'].astype('string')
        frame['distance_to_core_km'] = pd.to_numeric(
            location['distance_to_land_value_core_km'], errors='coerce'
        )
        frame['land_value_2010'] = pd.to_numeric(
            location['land_value_per_ha_log_2010'], errors='coerce'
        )
        excluded = ()
    census = {
        key: pd.read_parquet(folder / f'census_{key}.parquet')
        for key in ('tract', 'blockgroup', 'place', 'county')
        if (folder / f'census_{key}.parquet').exists()
    }
    if census:
        for key in ('tract', 'blockgroup', 'place'):
            census.setdefault(key, pd.DataFrame())
        bi.attach_census_inputs(frame, census)
    excluded = tuple(excluded) + tuple(spec.get('exclude_inputs', ())) + tuple(without)
    scales = bi.appraisal_scales(
        frame['county_fips'], appraisal, hpi, int(spec['common_year'])
    )
    fips = frame.groupby('admin3_id')['county_fips'].first().astype(str)
    config = bi.delivery_config(spec)
    acs_value = None
    if 'county' in census:
        acs_value = bi.acs_value_levels(
            census['county']['acs_median_home_value_county'],
            fips,
            hpi,
            int(spec['common_year']),
        )
    imputer = bi.Imputer(
        frame,
        area,
        orders={},
        config=config,
        value_scale=fips.map(scales['scale']),
        exclude_inputs=excluded,
        acs_value=acs_value,
        extensions=bi.load_extensions(
            spec.get('extensions') if extensions is None else extensions
        ),
    )
    return imputer, config


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
    imputer, config = build_imputer(args.folder, without, extensions)
    filled, _ = imputer.run(log=lambda message: None)
    log('filled; chain inputs ready')
    specs = {spec.column: spec for spec in bi.TARGETS}
    tables = []
    for column, kind in (('year_built', 'year'), ('structure_value', 'value')):
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
    pd.set_option('display.width', 250)
    print(scores.round(4).to_string())
    log('done')


if __name__ == '__main__':
    main()
