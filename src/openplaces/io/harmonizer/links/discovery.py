"""Auto-discovery of the sources a link step reads for an admin unit,
and the remap sidecars applied to a discovered source.
"""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import (
    AdminId,
)
from openplaces.diagnostics import find_recipes
from openplaces.io.harmonizer import (
    HarmonizeState,
)
from openplaces.io.harmonizer.links.combine import (
    _write_prioritized,
)
from openplaces.recipe import (
    STACKED_UNITS_LAYER_KEY,
    get_supplements_key,
)


def _find_admin_scoped_recipe_ids(state: HarmonizeState, entity_type: str) -> list[str]:
    """Ingest recipes of *entity_type* whose admin scope covers ``state.admin_id``.

    One recipe id per (admin_id, source_id, filename_suffix), keeping the
    newest version when several exist for the same source (mirrors the
    specificity/version precedence ``find_entity_recipe_id`` uses,
    ``recipe.py:444-463``). ``filename_suffix`` keeps this a *competing-
    alternative* dedup, not a same-source-can-only-mean-one-recipe dedup:
    two recipe files sharing admin_id/source_id/version but distinguished
    by a filename suffix (e.g. a PACS roll's own APPRAISAL_INFO recipe
    alongside its ``_improvement-detail`` sibling, both ``source_id:
    victoriacad``) are genuinely different tables meant to coexist, not
    two versions of the same one competing to be "the" victoriacad
    recipe -- see :func:`_discover_link_sources`, which joins every one
    of them.

    Returned least-specific-admin-id-first, version ascending within a tier:
    :func:`link_by_id`'s auto-discover mode joins sources in this order, so
    the most admin-specific source's attributes are the ones applied last
    (see its column-priority rule) -- a county-scoped recipe's own values
    win over a statewide recipe's by default, not whichever happens to have
    the newer version string.

    Recipes with ``exclude_from_auto_discover: true`` are skipped -- for a
    parcel-entity ingest recipe that is a reference dataset consumed only via
    an explicit crosswalk (e.g. a legacy/external source's own attributes,
    not meant to auto-roll into the canonical spine).
    """
    if state.admin_id is None:
        return []
    best: dict[tuple[str, str, str], tuple[str, str, int]] = {}
    for _, row in find_recipes(entity_type, stage='ingest').iterrows():
        if row['exclude_from_auto_discover']:
            continue
        admin_id_str = row['admin_id']
        if not admin_id_str or not AdminId(admin_id_str).is_parent_or_equal_of(
            state.admin_id
        ):
            continue
        key = (admin_id_str, row['source_id'], row['filename_suffix'])
        recipe_id = row['recipe_id']
        specificity = admin_id_str.count('-') + 1
        if key not in best or row['version'] > best[key][0]:
            best[key] = (row['version'], recipe_id, specificity)
    return [
        recipe_id
        for _version, recipe_id, _specificity in sorted(
            best.values(), key=lambda vrs: (vrs[2], vrs[0])
        )
    ]


def _discover_link_sources(state: HarmonizeState, entity_type: str) -> list[dict]:
    """Find ingest sources covering ``state.admin_id`` and how to join each.

    A standalone roll (the candidate's own primary entity) joins on the
    standardized cross-source key ``parcel_id_local``. A bundled
    ``additional_layers`` entry joins on its declared ``layer_key`` if
    present (a same-source key shared with its primary entity, e.g.
    MassGIS's ``parcel_id_admin2``), else also falls back to
    ``parcel_id_local``.

    Ordered least-specific-admin-id-first, version ascending within a tier
    (see :func:`_find_admin_scoped_recipe_ids`): :func:`link_by_id` joins
    matches in this order and, for any column covered by more than one
    match, prefers whichever source is applied last as long as it covers a
    majority of parcels — so the most admin-specific source's attributes
    win by default, falling back to version only among equally-specific
    sources.

    Each match also carries its own ``aggregation_function`` (a recipe's or
    ``additional_layers`` entry's own top-level ``aggregation_function``
    key, or ``None``), letting one specific ingest recipe declare that its
    rows are structurally 1:many for a reason the attribute registry's
    global default does not anticipate -- e.g. a PACS
    ``APPRAISAL_IMPROVEMENT_DETAIL`` roll, one row per building component,
    where ``year_built`` needs the earliest component's year rather than
    the registry's ``mean``. Scoped to the recipe that declares it: unlike a
    caller-supplied override on the :func:`link_by_id` step itself, it
    never reaches a sibling match's columns.

    A supplement declaring ``supplements_key`` joins on that column
    instead of ``parcel_id_local`` (``key`` and ``supplements_key`` both
    carry it), validated against its roll by
    :func:`~openplaces.recipe.get_supplements_key`, which raises on a
    column either table does not produce.
    """
    # Imported at call time on purpose: the supplements tests patch
    # `openplaces.recipe.get_recipe_by_id` with fabricated recipes, and a
    # module-level binding would not see the patch (the linter removed
    # this once, 2026-09-29, and three tests died).
    from openplaces.recipe import get_recipe_by_id  # noqa: F811, PLC0415

    matches = []
    for recipe_id in _find_admin_scoped_recipe_ids(state, entity_type):
        recipe = get_recipe_by_id(recipe_id)
        supplements_key = get_supplements_key(recipe)
        matches.append(
            {
                'recipe_id': recipe_id,
                'layer': None,
                'key': supplements_key or 'parcel_id_local',
                'aggregation_function': recipe.get('aggregation_function'),
                'supplements': recipe.get('supplements'),
                'supplements_key': supplements_key,
            }
        )
        for layer_spec in recipe.get('additional_layers') or []:
            if 'entity' not in layer_spec:
                continue
            matches.append(
                {
                    'recipe_id': recipe_id,
                    'layer': str(layer_spec['entity'].entity_type),
                    'key': layer_spec.get('layer_key', 'parcel_id_local'),
                    'aggregation_function': layer_spec.get('aggregation_function'),
                    'supplements': None,
                    'stacked_units_layer': bool(
                        layer_spec.get(STACKED_UNITS_LAYER_KEY)
                    ),
                }
            )

    from openplaces.recipe import get_supplemented_table

    # Raises on a supplement whose roll, layer, entity type or scope
    # is wrong. Left unchecked, _select_supplements would still match
    # it on the host id alone and join it onto rows it does not
    # describe.
    for match in matches:
        if match['supplements']:
            get_supplemented_table(get_recipe_by_id(match['recipe_id']))
    return matches


def _select_supplements(matches: list[dict], spine_source_ids: set[str]) -> list[dict]:
    """Keep the matches that supplement one of the spine's own sources.

    A supplement (a recipe declaring ``supplements: <recipe_id>``) details
    the entities of the recipe it names, e.g. a roll's improvement-detail
    member. Joined onto a spine built from that roll, its columns land on
    the entity they describe. A supplement of a roll this spine did not
    load has nothing to attach to here, and a roll itself is already the
    spine's rows, so neither is joined.

    A supplement of an ``additional_layers`` table names the layer's
    host recipe (``supplements`` plus ``supplements_layer``, see
    :func:`~openplaces.recipe.get_supplemented_table`), and
    ``union_spine_sources`` records a layer source under its host's id,
    so the same comparison matches it. The layer, entity type and scope
    are checked when the match is discovered, not here.
    """
    return [
        match
        for match in matches
        if match.get('supplements') and match['supplements'] in spine_source_ids
    ]


def _apply_remap_csvs(state: HarmonizeState, recipe_id: str) -> HarmonizeState:
    """Auto-apply any ``{recipe_id}_*-remap.csv`` crosswalk found beside *recipe_id*.

    Each match's filename (the part between ``{recipe_id}_`` and
    ``-remap.csv``, dashes replaced by underscores) names the spine column it
    remaps; applied only when that column is present. Output columns are
    whichever non-key columns the crosswalk's own header defines (e.g.
    ``use_group``/``use_subgroup`` for a use-code crosswalk). The crosswalk's
    own key length determines how much to truncate codes before lookup
    (handles e.g. 3- vs 4-digit codes sharing one 3-digit crosswalk).

    An optional ``admin_id`` column scopes a row to one admin unit and
    its descendants; a scoped row overrides the unscoped row for the
    same key. This is how a single deviant source jurisdiction (Kleberg
    County's use codes do not follow the statewide category scheme)
    gets corrected in data, without forking the recipe or teaching the
    code any geography.
    """
    if state.spine is None:
        return state
    from openplaces.path import recipe_path
    from openplaces.recipe import get_recipe_by_id  # noqa: F811, PLC0415

    recipe = get_recipe_by_id(recipe_id)
    recipe_dir = recipe_path(recipe['admin_id'], recipe['entity'], as_dir=True)
    if not recipe_dir.exists():
        return state

    spine = state.spine
    prefix = f'{recipe_id}_'
    for csv_path in sorted(recipe_dir.glob(f'{prefix}*-remap.csv')):
        stem = csv_path.stem
        if not stem.endswith('-remap'):
            continue
        column = stem[len(prefix) : -len('-remap')].replace('-', '_')
        if column not in spine.columns:
            continue
        table = pd.read_csv(csv_path, dtype=str)
        key_col = table.columns[0]
        if 'admin_id' in table.columns:
            admin = str(state.admin_id) if state.admin_id is not None else ''
            scope = table['admin_id'].fillna('')
            applies = (scope == '') | scope.map(
                lambda s: bool(s) and (admin == s or admin.startswith(s + '-'))
            )
            table = (
                table[applies]
                .assign(_depth=scope[applies].str.len())
                .sort_values('_depth', kind='stable')
                .drop_duplicates(subset=[key_col], keep='last')
                .drop(columns=['_depth', 'admin_id'])
            )
        table = table.drop_duplicates(subset=[key_col]).set_index(key_col)
        key_lengths = table.index.to_series().astype(str).str.len()
        key_length = int(key_lengths.mode().iat[0])
        codes = spine[column].astype('string').str.slice(0, key_length)
        for target in table.columns:
            # Gap-fill, never wholesale replace: auto-discovery
            # calls this once per matched source, and a county
            # crosswalk covering only part of the roll would
            # otherwise null out every value a statewide crosswalk
            # already resolved. Same rule as the value columns
            # joined beside it.
            _write_prioritized(spine, target, codes.map(table[target]))
        if state.verbose:
            matched = codes.isin(table.index).sum()
            print(
                f'  link_by_id: applied {csv_path.name} '
                f'({matched:,d}/{len(spine):,d} {column!r} matched)'
            )

    state.spine = spine
    return state
