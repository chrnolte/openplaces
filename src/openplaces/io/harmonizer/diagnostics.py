"""Harmonize-stage conflict-inspection diagnostics, written to the cache.

Enabled by ``save_statistics`` on the harmonize entrypoint (or a recipe-level
``save_statistics: true``). Mirrors the conventions already established by
``openplaces.io.curator.diagnostics``: these functions never modify the
harmonized spine, write plain CSVs under ``cache_path(admin_id, entity,
filename=...)`` (the same regenerable, non-critical storage every other
cache artifact in this codebase uses), and a failure is warned about rather
than raised so diagnostics can never break a harmonize run.

Naming convention for any report added here: ``{topic}-agreement.csv``
(pairwise agreement rates), ``{topic}-conflicts.csv`` (a crosstab of
disagreeing value-pairs with counts), ``{topic}-conflict-cases.csv`` (a
bounded sample of raw conflicting rows).
"""

from __future__ import annotations

import warnings

import pandas as pd


def save_postal_code_conflicts(
    state,
    frame: pd.DataFrame,
    sources: list[str],
) -> None:
    """Write ZIP-code source agreement/conflict reports for one admin unit.

    Called by :func:`~openplaces.io.harmonizer.addresses.reconcile_postal_code`
    over the per-source normalized 5-digit ZIP values it already computed
    (*frame*, one column per entry in *sources*, in priority order). No-op
    unless ``state.save_statistics`` is set; never raises.

    Parameters
    ----------
    frame : pandas.DataFrame
        One column per source in *sources*, each holding that source's
        normalized 5-digit ZIP (or null).
    sources : list of str
        Source labels, in priority order, matching *frame*'s columns.
    """
    if not state.save_statistics:
        return
    from itertools import combinations

    from openplaces.path import cache_path

    try:
        entity = state.recipe.get('entity')
        agreement_rows = []
        conflict_frames = []
        cases = []
        for a, b in combinations(sources, 2):
            if a not in frame.columns or b not in frame.columns:
                continue
            both = frame[[a, b]].dropna()
            n = len(both)
            differ = both[a] != both[b]
            agree = int((~differ).sum())
            agreement_rows.append(
                {
                    'source_a': a,
                    'source_b': b,
                    'n_both_present': n,
                    'n_agree': agree,
                    'n_conflict': n - agree,
                    'agreement_rate': round(agree / n, 4) if n else float('nan'),
                }
            )
            diff = both[differ]
            if len(diff):
                pair = (
                    diff.groupby([a, b], dropna=False)
                    .size()
                    .rename('count')
                    .reset_index()
                    .rename(columns={a: 'value_a', b: 'value_b'})
                )
                pair.insert(0, 'source_b', b)
                pair.insert(0, 'source_a', a)
                conflict_frames.append(pair)
                cases.append(diff.reset_index())
    except Exception as exc:  # diagnostics must never break a harmonize run
        warnings.warn(f'save_postal_code_conflicts: skipped ({exc}).')
        return

    if not agreement_rows:
        return
    agree_path = cache_path(
        state.admin_id, entity, filename='postal-code-agreement.csv'
    )
    agree_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(agreement_rows).to_csv(agree_path, index=False)

    if conflict_frames:
        conflicts = pd.concat(conflict_frames, ignore_index=True).sort_values(
            'count', ascending=False, ignore_index=True
        )
        conflicts.to_csv(
            cache_path(state.admin_id, entity, filename='postal-code-conflicts.csv'),
            index=False,
        )
    if cases:
        pd.concat(cases, ignore_index=True).head(1000).to_csv(
            cache_path(
                state.admin_id, entity, filename='postal-code-conflict-cases.csv'
            ),
            index=False,
        )


def save_geographic_id_inheritance_conflicts(state) -> None:
    """Write geographic-id inheritance disagreements for one admin unit.

    Called by :func:`~openplaces.io.harmonizer.spine.link_geographic_ids`
    after an ``inherit_from`` rollup: reports parcels whose linked
    footprints disagreed on an output column (see
    :func:`~openplaces.io.harmonizer.spine._inherit_geographic_ids`), which
    is exactly the residual that fell back to a direct spatial join. No-op
    unless ``state.save_statistics`` is set, or there is nothing to report;
    never raises.
    """
    if not state.save_statistics:
        return

    from openplaces.path import cache_path

    conflicts = state.metadata.get('geographic_id_inheritance_conflicts')
    if not conflicts:
        return

    try:
        entity = state.recipe.get('entity')
        df = pd.DataFrame(conflicts)
        summary = (
            df.groupby('output_column')
            .size()
            .rename('n_disagreeing_groups')
            .reset_index()
            .sort_values('n_disagreeing_groups', ascending=False, ignore_index=True)
        )
        cases = df.copy()
        cases['values'] = cases['values'].map(lambda v: '; '.join(str(x) for x in v))
    except Exception as exc:  # diagnostics must never break a harmonize run
        warnings.warn(f'save_geographic_id_inheritance_conflicts: skipped ({exc}).')
        return

    conflicts_path = cache_path(
        state.admin_id, entity, filename='geographic-id-inheritance-conflicts.csv'
    )
    conflicts_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(conflicts_path, index=False)
    cases.head(1000).to_csv(
        cache_path(
            state.admin_id,
            entity,
            filename='geographic-id-inheritance-conflict-cases.csv',
        ),
        index=False,
    )
