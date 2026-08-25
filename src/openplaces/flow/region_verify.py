"""Verify a curated region county-by-county, then export its delivery.

The gate that shipped the 2026 CHEER inventories. For every member of a
registered region it checks that the curated output exists, is
non-empty, and stays inside its admin unit's bounding box, with a
tolerance calibrated on measured failure modes: genuine cross-unit
contamination (a mis-keyed assessor roll) puts dozens to thousands of
clustered rows far outside the unit, while healthy counties carry a
handful of source records with garbage coordinates (two txgio rows in
Liberty County, TX sit at Nebraska latitudes with a Liberty county
attribute). A unit therefore passes with a note when at most
``max_outliers`` rows exceed the margin, and fails beyond that.

Optionally the same outlier-count test runs over named input layers, to
catch a mis-keyed ingest input whose curated output happens to pass.

Only when every member passes does ``--deliver`` export the bundle.

Usage::

    python -m openplaces.flow.region_verify RECIPE_ID REGION_ID
        [--deliver] [--input-layer RECIPE_ID ...]
        [--margin 0.1] [--max-outliers 10]
"""

import argparse
import sys
import time
import warnings


def _bbox_outliers(gdf, unit_bounds, margin):
    """Count rows whose bounds exceed the unit's bbox by margin degrees."""
    bx = gdf.geometry.bounds
    a = unit_bounds
    return int(
        (
            (bx['minx'] < a[0] - margin)
            | (bx['maxx'] > a[2] + margin)
            | (bx['miny'] < a[1] - margin)
            | (bx['maxy'] > a[3] + margin)
        ).sum()
    )


def verify_region(
    recipe_id,
    region_id,
    input_layers=(),
    margin=0.1,
    max_outliers=10,
):
    """Gate every member of a region; return (bad, total_rows).

    Parameters
    ----------
    recipe_id : str
        The curated entity recipe whose per-unit outputs are checked.
    region_id : str
        A region from the shared registry; its members define the units.
    input_layers : sequence of str, optional
        Ingest/harmonize recipe ids whose per-unit outputs get the same
        outlier-count bbox test, catching a mis-keyed input behind a
        passing output.
    margin : float, optional
        Degrees of bbox overshoot tolerated before a row counts as an
        outlier. Boundary simplification and genuinely straddling
        parcels stay inside 0.1; contamination misses by whole degrees.
    max_outliers : int, optional
        Rows beyond the margin tolerated (with a printed note) before
        the unit fails. Source coordinate noise is this small;
        systematic mis-keying never is.

    Returns
    -------
    tuple of (list of (str, str), int)
        Failed units with reasons, and the summed row count.
    """
    import openplaces as op

    members = op.get_region_admin_ids(region_id)
    parent = '-'.join(members[0].split('-')[:-1])
    level = len(members[0].split('-'))
    adm = op.get_admin(parent, level=level, geom=True)
    bad, total_rows = [], 0
    for unit in members:
        try:
            g = op.get_entities(recipe_id, unit, geom=True)
        except Exception as exc:
            bad.append((unit, f'missing: {type(exc).__name__}'))
            continue
        a = adm.loc[unit].geometry.bounds
        n_out = _bbox_outliers(g, a, margin)
        if len(g) == 0:
            bad.append((unit, 'empty'))
        elif n_out > max_outliers:
            bad.append((unit, f'{n_out} rows beyond {margin} deg'))
        else:
            if n_out:
                print(
                    f'  NOTE {unit}: {n_out} outlier row(s) kept (source noise)',
                    flush=True,
                )
            for layer_id in input_layers:
                try:
                    d = op.get_entities(layer_id, unit, geom=True)
                except Exception:
                    continue
                d_out = _bbox_outliers(d, a, margin)
                if d_out > max_outliers:
                    bad.append((unit, f'{layer_id}: {d_out} rows out of bounds'))
                    break
                if d_out:
                    print(
                        f'  NOTE {unit}: {layer_id} has {d_out} outlier '
                        'row(s) (source noise)',
                        flush=True,
                    )
        total_rows += len(g)
        print(f'  {unit} {len(g):9,} rows', flush=True)
    print(
        f'{region_id}: {len(members) - len(bad)}/{len(members)} verified, '
        f'{total_rows:,} rows total',
        flush=True,
    )
    for unit, why in bad:
        print(f'  FAILED {unit}: {why}', flush=True)
    return bad, total_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('recipe_id')
    parser.add_argument('region_id')
    parser.add_argument('--deliver', action='store_true')
    parser.add_argument('--input-layer', action='append', default=[])
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--max-outliers', type=int, default=10)
    args = parser.parse_args(argv)
    warnings.filterwarnings('ignore')

    import openplaces as op
    from openplaces.io.delivery import delivery_paths

    bad, _ = verify_region(
        args.recipe_id,
        args.region_id,
        input_layers=args.input_layer,
        margin=args.margin,
        max_outliers=args.max_outliers,
    )
    if bad:
        sys.exit(1)
    if not args.deliver:
        print('(dry verification only; pass --deliver to ship)')
        return
    t0 = time.time()
    op.export_delivery(args.recipe_id, region=args.region_id, verbose=True)
    print(f'DELIVERED {args.region_id} in {time.time() - t0:.0f}s', flush=True)
    for role, path in delivery_paths(args.recipe_id, region=args.region_id).items():
        size = path.stat().st_size / 1048576 if path.exists() else 0
        print(f'  {role:10s} {size:8.1f} MB  {path.name}', flush=True)


if __name__ == '__main__':
    main()
