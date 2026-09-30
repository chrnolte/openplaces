"""Local parcel ids and the stacked-units split at ingest."""

from __future__ import annotations

import warnings

import geopandas as gpd
import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.geo.ids import add_parcel_id_alnum
from openplaces.recipe import (
    STACKED_UNITS_LAYER_KEY,
    build_table_recipe,
    get_process_admin_level,
    get_recipe,
    get_save_admin_level,
)


class _TableIdMixin:
    """Methods of :class:`~openplaces.io.ingester.table_ingester.TableIngester`
    (table_ids).

    State lives on the TableIngester instance.
    """

    def _load_parcel_id_overrides(self, kind: str) -> dict | None:
        """Load the recipe-tree id-conversion override table, if present.

        Returns an ``{admin_id: {pattern, conv, [tolerance], [source]}}`` dict
        in the shape :func:`~openplaces.geo.ids.compute_parcel_id_local`'s
        ``instruction`` parameter expects (plus the ``source`` key this
        method also resolves for -- see :meth:`_resolve_parcel_id_source`),
        built from rows of ``{country}_{entity_type}_id-overrides.csv``
        (``recipes/{country}/_all/{entity_type}/_all/``) matching *kind* and
        this recipe's ``source_id`` (a blank ``source_id`` row matches any
        source at that ``admin_id``; an exact-source row at the same
        ``admin_id`` takes precedence). A row's optional ``tolerance`` column
        overrides the duplicate-guard tolerance for that admin unit (see
        ``compute_parcel_id_local``); left blank, the caller's default
        applies. A row's optional ``source`` column overrides which raw
        column feeds ``parcel_id_local`` for that admin unit (e.g. a county
        whose usual id column is itself truncated/degenerate at the source --
        confirmed for Carteret County, NC's ``ALTPARNO`` field, block-level
        truncated to 8 of the real 15-digit PIN's digits for ~99% of its
        rows: no amount of pattern/conv tuning can recover precision a
        source field never had, so a different column must be chosen
        instead). Admin-hierarchy walking from a specific admin id to a
        broader one is handled by
        :func:`~openplaces.geo.ids._resolve_instruction` for
        pattern/conv/tolerance and by :meth:`_resolve_parcel_id_source` for
        ``source`` -- this only builds the dict. Returns ``None`` when no
        override table exists for this recipe's country/entity_type (the
        common case today).
        """
        entity = self.recipe.get('entity')
        admin_id = self.recipe.get('admin_id')
        if entity is None or not admin_id or not admin_id.levels:
            return None
        country = AdminId(admin_id.levels[0])
        try:
            table = get_recipe(
                country,
                str(entity.entity_type),
                filename='id-overrides',
                dtype=str,
                keep_default_na=False,
            )
        except OSError:
            return None

        table = table[table['kind'] == kind]
        source_id = str(entity.source) if entity.source else ''

        def _entry(row):
            entry = {'pattern': row['pattern'], 'conv': row['conv']}
            tolerance = row.get('tolerance', '')
            if tolerance:
                entry['tolerance'] = tolerance
            source = row.get('source', '')
            if source:
                entry['source'] = source
            return entry

        overrides: dict[str, dict] = {}
        for _, row in table[table['source_id'] == ''].iterrows():
            if row['admin_id']:
                overrides[row['admin_id']] = _entry(row)
        for _, row in table[table['source_id'] == source_id].iterrows():
            if row['admin_id']:
                overrides[row['admin_id']] = _entry(row)
        return overrides or None

    @staticmethod
    def _resolve_parcel_id_source(
        admin_unit_id, instruction: dict | None, default: str
    ) -> str:
        """Return the per-admin-unit ``source`` column override, if any.

        Walks from *admin_unit_id* up to broader admin units (same
        dash-truncation walk :func:`~openplaces.geo.ids._resolve_instruction`
        uses for ``pattern``/``conv``/``tolerance``), returning the first
        ``instruction`` entry that declares a ``source``. *instruction* is
        not itself admin-hierarchy-aware for this key (`_resolve_instruction`
        only resolves pattern/conv/tolerance), so this is a distinct, small
        walk rather than a call into that function.
        """
        if instruction:
            aid = str(admin_unit_id) if admin_unit_id is not None else None
            while aid:
                entry = instruction.get(aid)
                if entry and entry.get('source'):
                    return entry['source']
                aid = aid.rsplit('-', 1)[0] if '-' in aid else None
        return default

    def _add_parcel_id_local(self, df):
        """Add `parcel_id_local` from `parcel_id_assessor` per the recipe directive.

        Recipe directive::

            parcel_id_local:
              source: parcel_id_assessor   # default raw column to standardize
              kind: parcel                 # parcel | tax (selects default conv)
              admin_id_column: admin4_id   # optional: per-row admin unit (MA towns)
              # optional per-admin-unit override:
              instruction: {<admin_id>: {pattern: ..., conv: ..., source: ...}}

        An ``instruction`` entry's ``source`` (recipe-inline or from the
        recipe-tree override table, :meth:`_load_parcel_id_overrides`)
        overrides which raw column feeds ``parcel_id_local`` for that admin
        unit specifically, resolved by :meth:`_resolve_parcel_id_source` --
        for an admin unit whose usual *source* column is itself
        truncated/non-unique at the source (no ``pattern``/``conv`` tuning
        can recover precision a field never had).

        The conversion is admin-unit-specific: a recipe-inline `instruction`
        wins, then the recipe-tree override table
        (:meth:`_load_parcel_id_overrides`), then the bundled default table
        (see :func:`openplaces.geo.ids.compute_parcel_id_local`), and is
        hardened so it never adds duplicates beyond those already in
        `parcel_id_assessor` -- but only *within* the admin unit a given
        ingest chunk covers. When a recipe processes at a finer admin level
        than it saves at (e.g. MassGIS: per-town `process_by`, per-county
        `save_to`), multiple chunks' outputs are later merged, and nothing
        has checked uniqueness *across* those chunks. Two different towns'
        raw ids that happen to be identical (each perfectly valid, since a
        raw MassGIS map-parcel id is only documented as unique within its own
        town) would otherwise collapse into the same `parcel_id_local` once
        merged -- confirmed empirically on real Middlesex County data: 29% of
        parcels gained a "duplicate" this way, 99.9% of the colliding groups
        spanning more than one town. Prefixing with the chunk's own admin id
        whenever process level > save level closes this structurally, without
        touching `compute_parcel_id_local`'s own (still correct, per-chunk)
        duplicate guard.
        """
        spec = self.recipe.get('parcel_id_local')
        if not spec:
            return df
        from openplaces.geo.ids import compute_parcel_id_local

        default_source = spec.get('source', 'parcel_id_assessor')
        kind = spec.get('kind', 'parcel')
        instruction = {
            **(self._load_parcel_id_overrides(kind) or {}),
            **(spec.get('instruction') or {}),
        } or None
        admin_col = spec.get('admin_id_column')
        scope_across_chunks = get_process_admin_level(
            self.recipe
        ) > get_save_admin_level(self.recipe)

        if admin_col and admin_col in df.columns:
            # Per-row admin-unit-specific conversion (e.g. Massachusetts towns).
            result = pd.Series(pd.NA, index=df.index, dtype='string')
            for admin_id, group in df.groupby(admin_col):
                source = self._resolve_parcel_id_source(
                    admin_id, instruction, default_source
                )
                if source not in group.columns:
                    warnings.warn(
                        f"parcel_id_local: source column '{source}' not found "
                        f'for admin {admin_id}; skipping.',
                        stacklevel=2,
                    )
                    continue
                key = compute_parcel_id_local(
                    group[source],
                    admin_unit_id=admin_id,
                    instruction=instruction,
                    kind=kind,
                )
                if scope_across_chunks:
                    key = str(admin_id) + '|' + key
                result.loc[group.index] = key
            df['parcel_id_local'] = result
        else:
            admin_id = self.processing_chunk.get('admin_id_to_process')
            source = self._resolve_parcel_id_source(
                admin_id, instruction, default_source
            )
            if source not in df.columns:
                warnings.warn(
                    f"parcel_id_local: source column '{source}' not found; skipping.",
                    stacklevel=2,
                )
                return df
            key = compute_parcel_id_local(
                df[source],
                admin_unit_id=admin_id,
                instruction=instruction,
                kind=kind,
            )
            if scope_across_chunks and admin_id is not None:
                key = str(admin_id) + '|' + key
            df['parcel_id_local'] = key
        return add_parcel_id_alnum(df)

    # Raw id columns the fallback match key may be built from, best first. The
    # order matters: a source's own assessor id is the most specific, but a
    # statewide layer often leaves it blank or zero-filled while carrying the
    # real county PIN under parcel_id_admin3 (NC OneMap's altparno/parno pair is
    # the measured case).
    # Save

    def _split_stacked_units(self, gdf, suffix=''):
        """Split a parcel table's stacked units off into its property layer.

        Returns the parcel table (unchanged when the split does not apply
        or found nothing stacked) and the property table or None.
        """
        from openplaces.io.stacked_units import (
            is_enabled,
            lot_key_of,
            split_stacked_units,
            unit_key_of,
        )

        if not is_enabled(self.recipe) or not isinstance(gdf, gpd.GeoDataFrame):
            return gdf, None
        lot_key = lot_key_of(self.recipe)
        if lot_key not in gdf.columns:
            if lot_key == 'geo_id':
                # A parcel recipe with its own index has no geo_id column;
                # nothing to group on, so nothing to split.
                return gdf, None
            raise KeyError(
                f'lot_key {lot_key!r} names a column the table does not carry '
                f'after column mapping for {self.table_name}.'
            )
        result = split_stacked_units(
            gdf, lot_key=lot_key, unit_key=unit_key_of(self.recipe)
        )
        if self.verbose or result.n_stacks:
            print(f'  stacked units{suffix}: {result.summary()}')
        self.timer.mark(f'Split stacked units{suffix}')
        return result.parcels, result.properties

    def _stacked_units_recipe(self) -> dict:
        """The table recipe of this parcel recipe's implicit property layer.

        Built from `self.recipe` so that a temporary aggregate-mode recipe
        (save level stripped) carries over to the layer's output path.
        """
        spec = next(
            spec
            for spec in self.recipe.get('additional_layers', [])
            if spec.get(STACKED_UNITS_LAYER_KEY)
        )
        table_recipe = build_table_recipe(self.recipe, spec)
        if self.recipe.get('save_to') is not None:
            table_recipe['save_to'] = self.recipe['save_to']
        else:
            table_recipe.pop('save_to', None)
        return table_recipe
