"""One runnable rebuild of the administrative spine.

The five phases were a prose runbook in ``README.md``. Prose drifts from
code silently, and the order is not guessable: level 3 must be filled
before level 4, and zeros repaired before minting.
Getting it wrong produces a *plausible* spine rather than an error, which
is the worst possible failure for a file every other dataset is keyed on.

Three things this module adds over running the phases by hand.

**Convergence is detected, not counted.** Phases 4-5 repeat until a
dry-run mint reports zero moved identifiers. The runbook said "two or
three passes"; a fixed number is a guess that silently under-runs when
the data changes. :func:`rebuild_spine` loops until the fixed point and
raises if it has not arrived within *max_passes*.

**Prerequisites are declared and checked first.** The population raster
and every override recipe must already be ingested.
:func:`check_prerequisites` names what is missing instead of letting
phase 2 fail partway with half the overrides applied.

**The country arguments are data.** Which recipe supplies weights for
which scope, at which level, now lives in ``population-overrides.csv``
beside this module. Adding a country is a row, not a code edit. The
open geometry layer adds its own rows without anyone writing them: every
country-level the geoBoundaries licence sidecar marks shippable weighs
its units from that layer (:func:`geometry_overrides`), unless the table
names a national layer for the same scope and level, which wins.

Nothing here writes by default: ``apply=False`` runs every read-only
phase and reports what a real run would change.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from openplaces.io.admin_codes import build

LEVELS = (2, 3, 4)
OVERRIDES = Path(__file__).with_name('population-overrides.csv')
RASTER_RECIPE = 'population-ghsl-r2023a'

# Named join-key strategies an override row may ask for. A name, not an
# expression: the table is data read from disk, and data that is executed
# is not data. Add a strategy here and reference it by name.
KEY_STRATEGIES = {
    # Connecticut renumbered its counties into planning regions in 2022,
    # so the middle of the GEOID no longer identifies anything joinable.
    # State plus subdivision survives that renumbering.
    'state_plus_subdivision': lambda g: f'{g[:2]}{g[-5:]}',
}


def load_overrides(path=OVERRIDES, with_geometry=True) -> pd.DataFrame:
    """Read the population-override table, plus the geometry layer's rows.

    Parameters
    ----------
    path : pathlib.Path, optional
        The committed table.
    with_geometry : bool, optional
        Append :func:`geometry_overrides` for every level. Default True.

    Returns
    -------
    pandas.DataFrame
        Columns ``recipe_id``, ``scope``, ``level``, ``join_column``,
        ``key``, ``note``, ``required``. Blank ``join_column``/``key``
        mean "not used". A row from the table is required (its failure
        stops the rebuild); a geometry-layer row is not, since a
        country-level whose polygons pin to no unit only leaves those
        units on gap-filled weights, as they would be without the file.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame['level'] = frame['level'].astype(int)
    unknown = set(frame['key']) - set(KEY_STRATEGIES) - {''}
    if unknown:
        raise ValueError(
            f'population-overrides.csv names unknown key strategies '
            f'{sorted(unknown)}; add them to KEY_STRATEGIES or fix the table.'
        )
    frame['required'] = True
    if with_geometry:
        derived = geometry_overrides(exclude=set(zip(frame['scope'], frame['level'])))
        frame = pd.concat([frame, derived], ignore_index=True)
    return frame


def geometry_overrides(exclude=(), levels=LEVELS) -> pd.DataFrame:
    """Return one override row per shippable country-level of the geometry layer.

    Read from the geoBoundaries licence sidecar rather than written into
    the table: 280-odd rows that restate a sidecar drift from it. Empty
    when no geometry layer is configured (`build.GEOMETRY_RECIPE` unset).

    Parameters
    ----------
    exclude : iterable of (str, int), optional
        (scope, level) pairs the table already covers; a national layer
        wins over the world one.
    levels : iterable of int, optional
        Levels to derive rows for.

    Returns
    -------
    pandas.DataFrame
        Same columns as :func:`load_overrides`, ``required`` False.
    """
    columns = ['recipe_id', 'scope', 'level', 'join_column', 'key', 'note', 'required']
    if not build.GEOMETRY_RECIPE:
        return pd.DataFrame(columns=columns)
    from openplaces.io.scrapers.geoboundaries_scraper import (
        SHIPPABLE_TIER,
        load_licenses,
    )

    table = load_licenses()
    table = table[table['tier'] == SHIPPABLE_TIER]
    exclude = set(exclude)
    rows = []
    for level in levels:
        recipe_id = build.GEOMETRY_RECIPE.format(level=level)
        for _, row in table[table['admin_level'] == str(level)].iterrows():
            scope = row['admin1_id']
            if (scope, level) in exclude:
                continue
            rows.append(
                [
                    recipe_id,
                    scope,
                    level,
                    f'admin{level}_id',
                    '',
                    f'geoBoundaries {row["geoboundaries_level"]}, '
                    f'{row["license_spdx"]}',
                    False,
                ]
            )
    return pd.DataFrame(rows, columns=columns)


def check_prerequisites(overrides=None, verbose=True) -> list[str]:
    """Return the recipes a rebuild needs that are not ingested yet.

    Phase 2 reads one already-ingested entity per override row, and phase
    1 reads the population raster. Neither states that anywhere, so a
    rebuild used to fail partway with some overrides applied and others
    not -- a spine that looks finished and is not.
    """
    from openplaces.recipe import get_recipe_by_id

    overrides = load_overrides() if overrides is None else overrides
    missing = []
    for recipe_id in [RASTER_RECIPE, *dict.fromkeys(overrides['recipe_id'])]:
        try:
            get_recipe_by_id(recipe_id)
        except Exception:  # noqa: BLE001 - any resolution failure is "missing"
            missing.append(recipe_id)
    if verbose:
        if missing:
            print(f'  missing prerequisites ({len(missing)}): {missing}')
        else:
            print('  prerequisites: all present')
    return missing


def apply_population_overrides(overrides=None, verbose=True, strict=True) -> int:
    """Phase 2 -- weights from a country's own admin entity.

    Applied in table order, level 3 before level 4, because a parent's
    total is what phase 3 apportions a child's shortfall against.

    Every row is attempted before anything is raised, so one run reports
    every broken override rather than only the first.

    Parameters
    ----------
    overrides : pandas.DataFrame, optional
        Override table. Read from the committed CSV when omitted.
    verbose : bool, optional
        Print one line per row, saying whether it was applied.
    strict : bool, optional
        Raise if any row failed. Default True: a failed override leaves
        its units on gap-filled weights, and the mint that follows turns
        those weights into identifiers, which is the spine that looks
        finished and is not. Pass False only to survey the damage.

    Returns
    -------
    int
        Number of overrides applied.

    Raises
    ------
    RuntimeError
        If *strict* and any override failed.
    """
    overrides = load_overrides() if overrides is None else overrides
    if 'required' not in overrides:
        overrides = overrides.assign(required=True)
    applied = 0
    failed = []
    skipped = []
    for _, row in overrides.sort_values('level').iterrows():
        kwargs = {}
        if row['join_column']:
            kwargs['join_column'] = row['join_column']
        if row['key']:
            kwargs['key'] = KEY_STRATEGIES[row['key']]
        label = f'{row["scope"]:6s} level {row["level"]} <- {row["recipe_id"]}'
        try:
            build.build_population_from_entity(
                row['recipe_id'], row['scope'], level=row['level'], **kwargs
            )
            applied += 1
        except Exception as exc:  # noqa: BLE001
            message = (
                f'{row["recipe_id"]} @ {row["scope"]} (level {row["level"]}): {exc}'
            )
            (failed if row['required'] else skipped).append(message)
            warnings.warn(f'population override {message}', stacklevel=2)
            if verbose:
                print(f'    {label} {"FAILED" if row["required"] else "skipped"}')
            continue
        if verbose:
            print(f'    {label}')
    if skipped and verbose:
        print(
            f'    {len(skipped)} geometry-layer override(s) matched no unit; gap-filled'
        )
    if failed and strict:
        raise RuntimeError(
            f'{len(failed)} of {len(overrides)} population override(s) failed; '
            'their units would be minted on gap-filled weights: ' + '; '.join(failed)
        )
    return applied


def rebuild_spine(
    apply=False,
    levels=LEVELS,
    max_passes=6,
    backup_dir=None,
    skip_population=False,
    verbose=True,
):
    """Run every phase, then loop the mint until it stops moving.

    Parameters
    ----------
    apply : bool, optional
        Write. Default False runs the read-only phases and reports what a
        real run would change, which is the only safe default for a file
        every other dataset is keyed on.
    max_passes : int, optional
        Give up after this many mint passes (default 6). Reaching it
        raises: a mint that will not converge is a bug in the assignment,
        not something to accept a partial result from.
    skip_population : bool, optional
        Skip phases 1-3. The population phases are the slow part and do
        not change between mint passes, so a re-run that only needs to
        re-check convergence can skip them.

    Raises
    ------
    RuntimeError
        If prerequisites are missing, if a phase-2 population override
        failed, or if the mint has not reached a fixed point within
        *max_passes*. Every one of those leaves weights the mint would
        turn into identifiers, so none of them may be minted through.
    """
    if missing := check_prerequisites(verbose=verbose):
        raise RuntimeError(
            f'cannot rebuild: {len(missing)} prerequisite recipe(s) not '
            f'ingested: {missing}. Ingest them first; see this package README.'
        )

    if not skip_population:
        if verbose:
            print('  phase 1: zonal population, coarsest level first')
        if apply:
            for level in levels:
                build.build_population(level, verbose=verbose)
        if verbose:
            print('  phase 2: country overrides')
        if apply:
            # Raises on a failed override rather than minting on the
            # gap-filled weights it left behind.
            apply_population_overrides(verbose=verbose)
        if verbose:
            print('  phase 3: apportion each parent shortfall')
        if apply:
            for level in (3, 4):
                build.fill_population_gaps(level, verbose=verbose)

    history = []
    for attempt in range(1, max_passes + 1):
        if apply:
            build.repair_zero_weights(levels=levels, verbose=verbose)
        moved = build.remint_spine(
            levels=levels, apply=apply, backup_dir=backup_dir, verbose=verbose
        )
        changed = sum(v['changed'] for v in moved.values())
        history.append(changed)
        if verbose:
            print(f'  pass {attempt}: {changed:,} identifier(s) moved')
        if changed == 0:
            return {'passes': attempt, 'history': history, 'converged': True}
        if not apply:
            # A dry run cannot converge by itself: nothing was written, so
            # the next pass would see the same input and move the same
            # identifiers. Report the first pass and stop.
            return {'passes': attempt, 'history': history, 'converged': False}

    raise RuntimeError(
        f'the mint did not reach a fixed point in {max_passes} passes '
        f'(identifiers moved per pass: {history}). Do not ship this spine: '
        'a mint that keeps moving means the assignment is unstable, and '
        'every downstream reference would be chasing it.'
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--apply', action='store_true', help='write (default: dry run)')
    parser.add_argument('--backup-dir', default=None)
    parser.add_argument('--max-passes', type=int, default=6)
    parser.add_argument('--skip-population', action='store_true')
    args = parser.parse_args()
    print(
        rebuild_spine(
            apply=args.apply,
            max_passes=args.max_passes,
            backup_dir=args.backup_dir,
            skip_population=args.skip_population,
        )
    )
