"""Offline generator for `parcel_id_local` pattern/conversion overrides.

Ports `places/lib/pc.py`'s `find_best_apn_link` -- a grid search over each
side's dominant extraction pattern x candidate conversion options, scored by
cross-dataset match rate -- but driven by openplaces' own ingested entity
data instead of the ZTRAX-derived source `parcel_id_links.csv` was ported
from. Not used on the ingest path; a maintenance tool for extending
per-admin-unit `parcel_id_local` coverage (see :mod:`openplaces.geo.ids`) as
new recipes are ingested, including to countries the bundled
`parcel_id_links.csv` never covered.

Two entry points:

- :func:`find_best_parcel_id_link` -- pure algorithm, two raw-id Series in,
  best pattern/conversion pair out. No I/O.
- :func:`propose_parcel_id_overrides` -- orchestration: for each requested
  admin unit, discovers the most specific ingested `parcel` recipe (the
  "pc" side) and any co-located `transaction`/`property` recipe that also
  derives `parcel_id_assessor` (the "za"/tax side), runs the search, and
  returns proposed rows in the same schema as the existing
  `{country}_{entity_type}_id-overrides.csv` files -- optionally appending
  them there directly.
"""

from functools import cache
from pathlib import Path

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.diagnostics import find_recipes
from openplaces.geo.ids import (
    _PARCEL_ID_DIR,
    PARCEL_ID_RELINK_THRESHOLD,
    _resolve_instruction,
    convert_parcel_id,
    dominant_parcel_id_pattern,
    simplest_parcel_id_pattern,
)
from openplaces.io.readers import get_entities
from openplaces.recipe import get_recipe_by_id, get_save_admin_level

_CONV_OPTIONS_PATH = _PARCEL_ID_DIR / 'parcel_id_conversion_options.csv'

# Bare, pattern-less whole-string operations tried for every pattern.
# 'pipe' preserves separator structure ('1-23' -> '1|23') where 'simple'
# discards it ('1-23' -> '123', now indistinguishable from '12-3'), so it
# can never introduce more cross-dataset collisions than 'simple' -- always
# trying both, and preferring 'pipe' on ties, is how the "prefer keeping
# separators" principle enters the search (see the tie-break in
# :func:`find_best_parcel_id_link`).
_BARE_CANDIDATES = [('Ignored', 'simple'), ('Ignored', 'pipe')]


@cache
def _parcel_id_conv_options() -> pd.DataFrame:
    """Candidate conversion codes per pattern (see the CSV's own header).

    Ported from `places/cfg/key/apn/apn_conversion_options.csv`.
    """
    return pd.read_csv(_CONV_OPTIONS_PATH, dtype=str)


def _conv_option_candidates(pattern: str) -> list[tuple[str, str]]:
    """Return `(pattern, conv_code)` candidates to try for *pattern*.

    Always includes :data:`_BARE_CANDIDATES`, plus any pattern-specific
    options from `parcel_id_conversion_options.csv`.
    """
    options = _parcel_id_conv_options()
    matched = options[options['pattern'] == pattern]
    return _BARE_CANDIDATES + list(zip(matched['pattern'], matched['code']))


def _unique_converted_values(raw: pd.Series, pattern: str, conv_code: str) -> set:
    """Non-null values of *raw* converted via (pattern, conv_code) that are
    not duplicated -- the only values usable as a join target."""
    converted = convert_parcel_id(raw, pattern, conv_code)
    dup = converted.duplicated(keep=False)
    return set(converted[converted.notna() & ~dup])


def find_best_parcel_id_link(
    pc: pd.Series,
    za: pd.Series,
    pattern_pc: str | None = None,
    pattern_za: str | None = None,
    min_match_ratio: float = 0.5,
) -> dict:
    """Find the (pattern, conversion) pair that best cross-links *pc* to *za*.

    Generalized port of `places/lib/pc.py:find_best_apn_link`: identifies
    each side's dominant extraction pattern (unless given), then grid-searches
    that pattern's candidate conversion options (:func:`_conv_option_candidates`)
    for the pair that maximizes the fraction of *pc* rows whose converted,
    non-duplicated value also appears among *za*'s converted, non-duplicated
    values. Unlike the original, *pc* and *za* are never assumed row-aligned
    (openplaces' pc/za candidates always come from different entity tables,
    e.g. a parcel recipe and a transaction recipe) -- matching is always by
    set membership, not positional equality.

    Parameters
    ----------
    pc : pd.Series
        Raw parcel-side ids (e.g. a parcel recipe's `parcel_id_assessor`).
    za : pd.Series
        Raw tax/transaction-side ids to cross-link against (e.g. a
        transaction or property recipe's `parcel_id_assessor` for the same
        admin unit).
    pattern_pc, pattern_za : str, optional
        Extraction pattern for each side, if already known. Resolved via
        :func:`~openplaces.geo.ids.dominant_parcel_id_pattern` otherwise.
    min_match_ratio : float
        Passed through to `dominant_parcel_id_pattern` when a pattern isn't
        given.

    Returns
    -------
    dict
        ``pattern_pc, conv_pc, pattern_za, conv_za, success, n_pc, n_za``.
        ``success`` is the fraction of all *pc* rows (including empty/
        duplicated ones) that matched -- a realistic real-world yield
        estimate, not a match rate over only the clean subset. Ties for
        the best ``success`` are broken in favor of candidates that don't
        use the lossy ``'simple'`` op on either side.
    """
    pc = pc.astype('string').str.strip().str.upper()
    za = za.astype('string').str.strip().str.upper()

    if pattern_pc is None:
        pattern_pc = dominant_parcel_id_pattern(pc, min_match_ratio)
    if pattern_za is None:
        pattern_za = dominant_parcel_id_pattern(za, min_match_ratio)

    za_candidates = _conv_option_candidates(pattern_za)
    za_unique_values = {
        (pat, conv): _unique_converted_values(za, pat, conv)
        for pat, conv in za_candidates
    }

    n = len(pc)
    best_success = -1.0
    ties: list[dict] = []
    for pc_pat, pc_conv in _conv_option_candidates(pattern_pc):
        converted_pc = convert_parcel_id(pc, pc_pat, pc_conv)
        unique_mask = converted_pc.notna() & ~converted_pc.duplicated(keep=False)

        for za_pat, za_conv in za_candidates:
            values = za_unique_values[(za_pat, za_conv)]
            success = (
                (unique_mask & converted_pc.isin(values)).sum() / n
                if values and n
                else 0.0
            )
            if success < best_success:
                continue
            result = {
                'pattern_pc': pc_pat,
                'conv_pc': pc_conv,
                'pattern_za': za_pat,
                'conv_za': za_conv,
                'success': success,
            }
            if success > best_success:
                best_success = success
                ties = [result]
            else:
                ties.append(result)

    ties.sort(key=lambda r: r['conv_pc'] == 'simple' or r['conv_za'] == 'simple')
    winner = ties[0]
    winner['n_pc'] = n
    winner['n_za'] = len(za)
    return winner


def _override_path(country: str, entity_type: str) -> Path:
    """Path to the recipe-tree override table for *country*/*entity_type*."""
    return (
        _PARCEL_ID_DIR.parent
        / 'recipes'
        / country
        / '_all'
        / entity_type
        / '_all'
        / f'{country}_{entity_type}_id-overrides.csv'
    )


def _most_specific_recipe(recipes: pd.DataFrame, admin_id: AdminId) -> dict | None:
    """Pick the most specific recipe row covering *admin_id*, if any.

    Mirrors the tie-break used by the harmonizer's own auto-discovery
    (`io/harmonizer/links.py::_find_reference_recipe`,
    `io/harmonizer/spine.py::_expand_auto_discover`): longest matching
    `admin_id` prefix wins; the first match at that specificity is kept.
    """
    covering = [
        row
        for _, row in recipes.iterrows()
        if row['admin_id'] and AdminId(row['admin_id']).is_parent_or_equal_of(admin_id)
    ]
    if not covering:
        return None
    return max(covering, key=lambda row: len(row['admin_id']))


def _load_assessor_ids(recipe_row: pd.Series, admin_id: AdminId) -> pd.Series | None:
    """Load `parcel_id_assessor` for *admin_id* from an ingest-stage recipe row.

    Explicitly requests the recipe's own per-row admin-id column alongside
    `parcel_id_assessor`: `get_entities` only restricts its output to
    *admin_id* when that column is among the requested `columns` -- asking
    for `parcel_id_assessor` alone silently disables the restriction and
    returns the recipe's *entire* output instead of just this admin unit's
    slice.

    Skips (returns None) rather than loading when the recipe saves data
    *coarser* than *admin_id* (e.g. a state-wide transaction file requested
    at county granularity): the coarser file's own admin-id column can't be
    equality-matched against a finer target, so naively loading it would
    either silently return zero rows or -- worse -- fall through to
    `get_entities`'s unfiltered fallback and load the entire state just to
    discard it. Matching a coarser source down to a finer admin unit needs
    name-based restriction (`io.harmonizer.restrict_to_admin_by_name`, per
    `plans/implemented/2026-07-29_..._results.md`'s WI finding) -- a
    harmonizer-layer concern this offline, geo-layer tool deliberately
    doesn't duplicate.
    """
    recipe_id = (
        f'{recipe_row["admin_id"]}_{recipe_row["entity_type"]}-'
        f'{recipe_row["source_id"]}-{recipe_row["version"]}'
    )
    try:
        recipe = get_recipe_by_id(recipe_id)
        save_level = get_save_admin_level(recipe)
        if 0 < save_level < admin_id.get_level():
            return None
        admin_col = f'admin{save_level}_id' if save_level > 0 else None
        columns = ['parcel_id_assessor'] + ([admin_col] if admin_col else [])
        df = get_entities(
            recipe,
            admin_id=str(admin_id),
            columns=columns,
            missing='ignore',
        )
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return None
    if df.empty or 'parcel_id_assessor' not in df.columns:
        return None
    if admin_col and admin_col in df.columns:
        df = df[df[admin_col] == str(admin_id)]
    return df['parcel_id_assessor']


def propose_parcel_id_overrides(
    admin_ids: list[str],
    min_success: float = 0.5,
    write: bool = False,
) -> pd.DataFrame:
    """Propose `parcel_id_local` overrides for real, already-ingested data.

    For each admin unit in *admin_ids*, finds the most specific ingested
    `parcel` recipe covering it (the pc side) and, if one exists, the most
    specific ingested `transaction` or `property` recipe also covering it
    that derives `parcel_id_assessor` (the za/tax side). When both sides
    exist, runs :func:`find_best_parcel_id_link` and proposes one
    `kind='parcel'` and one `kind='tax'` row. When only the parcel side
    exists, proposes a single `kind='parcel'` row via
    :func:`~openplaces.geo.ids.simplest_parcel_id_pattern` (self-consistency
    only -- not cross-validated against a second source) and does not
    fabricate a `kind='tax'` row.

    Parameters
    ----------
    admin_ids : list of str
        Admin units to evaluate (e.g. `['US-WI-ON', 'US-WI-VI']`). Not
        auto-discovered -- pass the specific units whose data has actually
        been ingested.
    min_success : float
        Skip proposing a cross-validated (`kind='tax'`/`kind='parcel'` pair)
        row when the winning `find_best_parcel_id_link` success rate is
        below this.
    write : bool
        If True, append proposed rows to the appropriate
        `{country}_{entity_type}_id-overrides.csv` (parcel side to the
        `parcel` file, tax side to the source recipe's own entity_type
        file), skipping any admin_id + kind combination already present.
        If False (default), only returns the proposed rows.

    Returns
    -------
    pd.DataFrame
        Columns: `admin_id, kind, pattern, conv, success, cross_validated,
        source_entity_type, source_id`. `success` is `NaN` for single-source
        (`cross_validated=False`) rows.
    """
    parcel_recipes = find_recipes('parcel', stage='ingest')
    tax_side_recipes = pd.concat(
        [
            find_recipes('transaction', stage='ingest'),
            find_recipes('property', stage='ingest'),
        ],
        ignore_index=True,
    )

    rows = []
    for admin_id_str in admin_ids:
        admin_id = AdminId(admin_id_str)

        pc_recipe_row = _most_specific_recipe(parcel_recipes, admin_id)
        if pc_recipe_row is None:
            continue
        pc = _load_assessor_ids(pc_recipe_row, admin_id)
        if pc is None or pc.empty:
            continue

        za_recipe_row = _most_specific_recipe(tax_side_recipes, admin_id)
        za = (
            _load_assessor_ids(za_recipe_row, admin_id)
            if za_recipe_row is not None
            else None
        )

        if za is not None and not za.empty:
            best = find_best_parcel_id_link(pc, za)
            if best['success'] >= min_success:
                rows.append(
                    {
                        'admin_id': admin_id_str,
                        'kind': 'parcel',
                        'pattern': best['pattern_pc'],
                        'conv': best['conv_pc'],
                        'success': best['success'],
                        'cross_validated': True,
                        'source_entity_type': 'parcel',
                        'source_id': pc_recipe_row['source_id'],
                    }
                )
                rows.append(
                    {
                        'admin_id': admin_id_str,
                        'kind': 'tax',
                        'pattern': best['pattern_za'],
                        'conv': best['conv_za'],
                        'success': best['success'],
                        'cross_validated': True,
                        'source_entity_type': za_recipe_row['entity_type'],
                        'source_id': za_recipe_row['source_id'],
                    }
                )
                continue

        pattern = simplest_parcel_id_pattern(pc)
        rows.append(
            {
                'admin_id': admin_id_str,
                'kind': 'parcel',
                'pattern': pattern,
                'conv': 'skip_empty: 1',
                'success': float('nan'),
                'cross_validated': False,
                'source_entity_type': 'parcel',
                'source_id': pc_recipe_row['source_id'],
            }
        )

    result = pd.DataFrame(
        rows,
        columns=[
            'admin_id',
            'kind',
            'pattern',
            'conv',
            'success',
            'cross_validated',
            'source_entity_type',
            'source_id',
        ],
    )

    if write and not result.empty:
        _write_overrides(result)

    return result


def _write_overrides(proposed: pd.DataFrame) -> None:
    """Append *proposed* rows to their target override CSVs.

    Groups by (country, target entity_type) -- `kind='parcel'` rows always
    target the `parcel` override file; `kind='tax'` rows target whichever
    entity_type actually produced the za-side data (`transaction` or
    `property`), matching how `TableIngester._load_parcel_id_overrides`
    resolves the file to read at ingest time. Never overwrites an existing
    admin_id + kind row -- only appends genuinely new ones.
    """
    proposed = proposed.copy()
    proposed['country'] = proposed['admin_id'].str.split('-').str[0]
    proposed['target_entity_type'] = proposed['source_entity_type']
    proposed.loc[proposed['kind'] == 'parcel', 'target_entity_type'] = 'parcel'

    for (country, entity_type), group in proposed.groupby(
        ['country', 'target_entity_type']
    ):
        path = _override_path(country, entity_type)
        if path.exists():
            existing = pd.read_csv(path, dtype=str)
        else:
            existing = pd.DataFrame(
                columns=[
                    'admin_id',
                    'source_id',
                    'kind',
                    'pattern',
                    'conv',
                    'tolerance',
                    'source',
                ]
            )

        existing_keys = set(zip(existing['admin_id'], existing['kind']))
        new_rows = group[
            ~group.apply(lambda r: (r['admin_id'], r['kind']) in existing_keys, axis=1)
        ]
        if new_rows.empty:
            continue

        to_append = pd.DataFrame(
            {
                'admin_id': new_rows['admin_id'],
                'source_id': '',
                'kind': new_rows['kind'],
                'pattern': new_rows['pattern'],
                'conv': new_rows['conv'],
                'tolerance': '',
                'source': '',
            }
        )
        out = pd.concat([existing, to_append], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)


def measure_parcel_id_linkage(pc: pd.Series, za: pd.Series, admin_id) -> float:
    """Return the share of *pc* ids the bundled rules link into *za*.

    Applies each side's own bundled conversion (the `parcel` kind to *pc*,
    the `tax` kind to *za*) and reports the fraction of all *pc* rows whose
    key also appears among *za*'s non-duplicated keys -- the same measure
    `find_best_parcel_id_link` maximizes, so the two are comparable.

    Parameters
    ----------
    pc : pd.Series
        Raw parcel-side ids for *admin_id*.
    za : pd.Series
        Raw tax or transaction-side ids for the same unit.
    admin_id : str or AdminId
        Unit whose bundled rules to apply.

    Returns
    -------
    float
        Between 0 and 1; 0 when either side is empty.
    """
    if pc is None or za is None or pc.empty or za.empty:
        return 0.0
    pattern_pc, conv_pc, _ = _resolve_instruction(admin_id, None, 'parcel')
    pattern_za, conv_za, _ = _resolve_instruction(admin_id, None, 'tax')
    converted = convert_parcel_id(
        pc.astype('string').str.strip().str.upper(), pattern_pc, conv_pc
    )
    targets = _unique_converted_values(
        za.astype('string').str.strip().str.upper(), pattern_za, conv_za
    )
    return float(converted.isin(targets).sum() / len(pc))


def _converted_share(pc: pd.Series, pattern: str, conv: str) -> float:
    """Return the share of populated ids in *pc* that convert to a key."""
    raw = pc.astype('string').str.strip().str.upper()
    populated = raw.notna() & raw.ne('')
    n = int(populated.sum())
    if not n:
        return 0.0
    key = convert_parcel_id(raw, pattern, conv)
    return float((key.notna() & populated).sum() / n)


def measure_parcel_id_fit(pc: pd.Series, admin_id) -> dict:
    """Report how a unit's bundled conversion behaves on one id column.

    The cross-dataset measure (:func:`measure_parcel_id_linkage`) needs a
    second source, and many places have only one: Maine ships a statewide
    parcel layer and no tax roll. A conversion can still be shown not to
    fit from one side alone, because a pattern written for differently
    formatted ids matches nothing at all. Measured 2026-08-25, the
    bundled rules convert 0% of Maine's 557,000 ingested parcel ids,
    whose shape is `S_S-S_S-S` where those rules expect `Sx-Sx_Sx-Sx`.

    Two numbers say it. `excess_loss` is how much more of the column the
    conversion fails to convert than a plain `simple` would, the same
    comparison `compute_parcel_id_local`'s `max_loss` guard makes at
    ingest time. `uniqueness` is the share of produced keys that are
    distinct, which catches the opposite failure: a conversion that
    matches but collapses distinct ids together.

    Parameters
    ----------
    pc : pd.Series
        Raw parcel-side ids for *admin_id*.
    admin_id : str or AdminId
        Unit whose bundled rule to apply.

    Returns
    -------
    dict
        `n`, `converted`, `converted_simple`, `excess_loss`, `uniqueness`.
    """
    raw = pc.astype('string').str.strip().str.upper()
    populated = raw.notna() & raw.ne('')
    n = int(populated.sum())
    if not n:
        return {
            'n': 0,
            'converted': 0.0,
            'converted_simple': 0.0,
            'excess_loss': 0.0,
            'uniqueness': 0.0,
        }
    pattern, conv, _ = _resolve_instruction(admin_id, None, 'parcel')
    key = convert_parcel_id(raw, pattern, conv)
    kept = key.notna() & populated
    n_kept = int(kept.sum())
    converted = n_kept / n
    simple = _converted_share(pc, None, 'simple')
    return {
        'n': n,
        'converted': converted,
        'converted_simple': simple,
        'excess_loss': simple - converted,
        'uniqueness': (key[kept].nunique() / n_kept) if n_kept else 0.0,
    }


def recheck_parcel_id_links(
    admin_ids: list[str],
    threshold: float = PARCEL_ID_RELINK_THRESHOLD,
    verbose: bool = True,
) -> pd.DataFrame:
    """Re-derive the parcel-id link for units whose bundled rule underperforms.

        The bundled conversions were measured once, on one vintage of one pair
        of sources. A county's newly ingested parcel or transaction data can
        carry a differently formatted id, and the join then quietly returns
        few rows rather than failing -- which is why this exists: run it when
        a unit's data is first ingested and linked, and whenever a link
        reports less than *threshold*.

    For a unit with both a parcel and a tax or transaction source
        ingested, it measures the cross-dataset linkage the bundled rules
        actually achieve (:func:`measure_parcel_id_linkage`) and, below
        *threshold*, re-runs the grid search (:func:`find_best_parcel_id_link`).

        For a unit with only a parcel source - Maine, where the state ships a
        parcel layer and no tax roll - there is no linkage to measure, but a
        rule written for another source's id format still shows itself by
        converting nothing (:func:`measure_parcel_id_fit`). Below *threshold*
        it proposes the simplest pattern that fits the ids in hand. That
        proposal is self-consistency only, never cross-validated, and
        `cross_validated` says which kind each row is.

        Either way a proposal is returned **only if it beats what the bundled
        rule achieved**, so a re-check can never make a unit worse. Units
        already at or above *threshold* are reported untouched.

        Nothing is written. The proposals go to
        `{country}_{entity_type}_id-overrides.csv` through
        :func:`propose_parcel_id_overrides`, or into the bundled table by
        hand, both of which are decisions for a person.

        Parameters
        ----------
        admin_ids : list of str
            Units to re-check. Not auto-discovered: pass the units whose data
            has actually been ingested.
        threshold : float, optional
            Achieved linkage below which the search is re-run.
        verbose : bool, optional
            Print one line per unit re-checked.

        Returns
        -------
        pd.DataFrame
            Columns `admin_id, cross_validated, achieved, best, improved,
            pattern_parcel, conv_parcel, pattern_tax, conv_tax, n_pc, n_za`.
            `achieved` is what the bundled rules manage today and `best` what
            the search found; `improved` marks the rows worth acting on.
            Empty when no unit has an ingested parcel source.
    """
    parcel_recipes = find_recipes('parcel', stage='ingest')
    tax_side_recipes = pd.concat(
        [
            find_recipes('transaction', stage='ingest'),
            find_recipes('property', stage='ingest'),
        ],
        ignore_index=True,
    )

    rows = []
    for admin_id_str in admin_ids:
        admin_id = AdminId(admin_id_str)
        pc_row = _most_specific_recipe(parcel_recipes, admin_id)
        if pc_row is None:
            continue
        pc = _load_assessor_ids(pc_row, admin_id)
        if pc is None or pc.empty:
            continue
        za_row = _most_specific_recipe(tax_side_recipes, admin_id)
        za = _load_assessor_ids(za_row, admin_id) if za_row is not None else None
        cross = za is not None and not za.empty

        row = {
            'admin_id': admin_id_str,
            'cross_validated': cross,
            'achieved': None,
            'best': None,
            'improved': False,
            'pattern_parcel': None,
            'conv_parcel': None,
            'pattern_tax': None,
            'conv_tax': None,
            'n_pc': len(pc),
            'n_za': len(za) if cross else 0,
        }

        if cross:
            achieved = measure_parcel_id_linkage(pc, za, admin_id_str)
            row.update(achieved=achieved, best=achieved)
            if achieved < threshold:
                best = find_best_parcel_id_link(pc, za)
                if best['success'] > achieved:
                    row.update(
                        best=best['success'],
                        improved=True,
                        pattern_parcel=best['pattern_pc'],
                        conv_parcel=best['conv_pc'],
                        pattern_tax=best['pattern_za'],
                        conv_tax=best['conv_za'],
                    )
            measure = 'linkage'
        else:
            # One side only. The conversion can still be shown not to
            # fit, and a pattern written for another source's format is
            # the common way that happens: it matches nothing. `achieved`
            # is the share of ids that convert at all, on the same 0-to-1
            # scale, so *threshold* means the same thing in both branches
            # -- but the proposal is self-consistency only, never
            # cross-validated, which `cross_validated` records.
            fit = measure_parcel_id_fit(pc, admin_id_str)
            row.update(achieved=fit['converted'], best=fit['converted'])
            if fit['converted'] < threshold and fit['excess_loss'] > 0:
                pattern = simplest_parcel_id_pattern(pc)
                proposed = _converted_share(pc, pattern, 'skip_empty: 1')
                if proposed > fit['converted']:
                    row.update(
                        best=proposed,
                        improved=True,
                        pattern_parcel=pattern,
                        conv_parcel='skip_empty: 1',
                    )
            measure = 'converts'

        rows.append(row)
        if verbose:
            verdict = (
                'ok'
                if row['achieved'] >= threshold
                else (
                    f'-> {row["best"]:.3f}'
                    if row['improved']
                    else 'no better rule found'
                )
            )
            note = '' if cross else ' (one source)'
            print(f'  {admin_id_str}: {measure} {row["achieved"]:.3f} {verdict}{note}')

    return pd.DataFrame(
        rows,
        columns=[
            'admin_id',
            'cross_validated',
            'achieved',
            'best',
            'improved',
            'pattern_parcel',
            'conv_parcel',
            'pattern_tax',
            'conv_tax',
            'n_pc',
            'n_za',
        ],
    )
