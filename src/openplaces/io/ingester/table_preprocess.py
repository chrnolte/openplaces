"""Preprocessing a read table: renames, casts, transformations,
geometry, admin ids and indexing, and the helpers it calls.
"""

from __future__ import annotations

import importlib
import warnings
from itertools import product

import geopandas as gpd
import pandas as pd

from openplaces.core.attribute_registry import get_null_placeholder
from openplaces.geo.ids import get_geo_ids
from openplaces.geo.overlay import overlay_admin_ids, prefer_located_admin_ids
from openplaces.geo.polygon import (
    clean_polygons,
    fix_polygons,
    resolve_overlapping_polygons,
)
from openplaces.io.transform import (
    add_unique_suffix,
    apply_transformations,
    get_crosswalk,
)
from openplaces.path import recipe_path
from openplaces.recipe import (
    resolve_attribute_name,
)


class _TablePreprocessMixin:
    """Methods of :class:`~openplaces.io.ingester.table_ingester.TableIngester`
    (table_preprocess).

    State lives on the TableIngester instance.
    """

    def _preprocess_recipe_data(self, df):
        """Rename columns, filter rows, apply transformations, set index.

        Parameters
        ----------
        df : DataFrame or GeoDataFrame
            Raw data read from the source file.
        """
        # Rename columns
        if 'columns' in self.recipe:
            df = df.rename(columns={v: k for k, v in self.recipe['columns'].items()})

        # Null out registry-declared placeholder sentinels (e.g. year_built's
        # 0 for "never recorded") -- applies to every recipe that maps a
        # column to this attribute, current or future, without per-recipe
        # opt-in. See attribute_registry.csv's null_placeholder column.
        for col in df.columns:
            placeholder = get_null_placeholder(resolve_attribute_name(col))
            if placeholder is None:
                continue
            numeric = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].mask(numeric == float(placeholder))

        # Replace known NA value strings with None
        if 'null_value_strings' in self.recipe:
            columns_to_convert = [
                v for v in df.columns if not v.startswith('admin1_id')
            ]
            for col, na_value in product(
                columns_to_convert, self.recipe['null_value_strings']
            ):
                i_has_na_value = df[col].eq(na_value)
                if i_has_na_value.sum():
                    df.loc[i_has_na_value, col] = None

        # Filter rows
        if 'query' in self.recipe:
            df = df.query(self.recipe['query'])
            if df.empty:
                # Nothing survived, so there is nothing to attribute, index
                # or save for this partition. Returning here rather than
                # letting an empty frame fall through the crosswalk and
                # `create_index`, both of which need columns the filter
                # just removed every row of. A recipe that deliberately
                # excludes a whole admin unit -- New England in
                # `US_admin-census-2025_admin4`, whose level 3 is towns and
                # which therefore has no level-4 subdivisions -- empties
                # that unit's partition by design.
                # Carry the index name `create_index` would have set.
                # An unnamed empty frame concatenated with properly
                # indexed partitions drops the name for the whole file,
                # which lands on disk as `__index_level_0__` and makes
                # every later read fail with
                # `No match for FieldRef.Name(admin4_id)`.
                # Every way a recipe can name its index, not just one
                # indexer's own kwarg: `create_index: {method: prefix}`
                # carries `name`, and a function indexer takes it in
                # `args` under either `name` or `new_admin_id_col`.
                create_index = self.recipe.get('create_index') or {}
                index_args = create_index.get('args') or {}
                index_name = (
                    index_args.get('new_admin_id_col')
                    or index_args.get('name')
                    or create_index.get('name')
                    or self.recipe.get('set_index')
                )
                if isinstance(index_name, str) and df.index.name != index_name:
                    df.index.name = index_name
                # Prune to the columns the recipe names, which the reorder
                # step below would have done. Returning here with the raw
                # read's own columns gave the aggregate a set of stray
                # all-null source columns that no populated chunk carries.
                if 'columns' in self.recipe and not self.recipe.get(
                    'keep_unnamed_columns'
                ):
                    kept = [c for c in self.recipe['columns'] if c in df]
                    kept += [c for c in ('geo_id', 'geometry') if c in df]
                    df = df[kept]
                if self.verbose:
                    print(f'  query left no rows for {self.table_name}; skipping.')
                return df

        # Drop duplicate rows (some source dumps repeat exact rows). `true`
        # drops full-row duplicates; a list of column names dedupes on a subset.
        if self.recipe.get('drop_duplicates'):
            subset = self.recipe['drop_duplicates']
            df = df.drop_duplicates(subset=None if subset is True else subset)

        # Apply variable transformations
        # (Before categorical casting and crosswalks, so that derived columns
        # can be cast to categorical and used in overlap resolution).
        # Either key alone must trigger the call: apply_transformations
        # handles both, but a recipe declaring only transformation_patterns
        # (e.g. a bare to_numeric cast) was silently skipped by the
        # single-key guard -- the txgio value columns stayed text through
        # an entire re-ingest before anyone noticed.
        if 'transformations' in self.recipe or 'transformation_patterns' in self.recipe:
            cols_before = list(df)
            # getattr, because a TableIngester can be constructed without
            # a download partition (several tests drive `process` directly
            # on a prepared frame).
            partition = getattr(self, 'download_partition', None) or {}
            partition_admin_id = partition.get(
                'admin_id_to_download'
            ) or self.recipe.get('admin_id')
            # The chunk's own admin unit decides an entry's `admin_ids`
            # scope: a statewide file split per county by `process_by`
            # downloads as the state, and only the chunk knows the
            # county. Without a process chunk the frame covers the
            # partition, so that is its unit.
            chunk = getattr(self, 'processing_chunk', None) or {}
            df = apply_transformations(
                df,
                self.recipe,
                admin_id=partition_admin_id,
                process_admin_id=chunk.get('admin_id_to_process') or partition_admin_id,
            )
            cols_added = [v for v in df if v not in cols_before]
        else:
            cols_added = []

        # Discard scratch columns a transformation needed only as an
        # intermediate step (e.g. a zero-padded value before a prefix is
        # added), so they never reach the saved output.
        if 'drop_columns' in self.recipe:
            to_drop = [c for c in self.recipe['drop_columns'] if c in df]
            df = df.drop(columns=to_drop)
            cols_added = [c for c in cols_added if c not in to_drop]

        # Cast columns to categorical
        # Each item is either a plain column name (unordered) or a single-key
        # dict {column: [cat1, cat2, ...]} for an inline ordered categorical.
        if 'columns_to_categorical' in self.recipe:
            for item in self.recipe['columns_to_categorical']:
                if isinstance(item, dict):
                    column_to_cast, inline_categories = next(iter(item.items()))
                else:
                    column_to_cast, inline_categories = item, None

                if column_to_cast not in df:
                    continue

                if inline_categories is not None:
                    labels = self._get_labels(column_to_cast)
                    if labels is not None and all(
                        c in labels for c in inline_categories
                    ):
                        # inline_categories are raw codes → remap and preserve order
                        values = df[column_to_cast].replace(labels)
                        categories = [labels[c] for c in inline_categories]
                    else:
                        # inline_categories are already the final label strings
                        values = df[column_to_cast]
                        categories = inline_categories
                    ordered = True
                else:
                    labels = self._get_labels(column_to_cast)
                    if labels is not None:
                        values = df[column_to_cast].replace(labels)
                        categories, ordered = labels.values(), True
                    else:
                        values, categories, ordered = df[column_to_cast], None, False

                df[column_to_cast] = pd.Series(
                    pd.Categorical(values, categories, ordered),
                    index=values.index,
                )

        admin_id_to_process = self.processing_chunk.get('admin_id_to_process')
        self.timer.mark('Transform' + self._mark_suffix(admin_id_to_process))

        # Clean geometries and resolve overlapping polygons (parcels, buildings).
        # Cleaning must precede the overlap test: invalid geometries cause
        # TopologyExceptions in shapely intersection.
        # Runs after transformations and categorical casting so that
        # 'prefer_higher' can reference a transformed or categorical column.
        if isinstance(df, gpd.GeoDataFrame):
            _suffix = self._mark_suffix(admin_id_to_process)

            if self.recipe.get('force_2d', False):
                import shapely

                df['geometry'] = shapely.force_2d(df['geometry'])

            if self.recipe.get('add_geometry_derivatives', False):
                from openplaces.geo.polygon import add_geometry_derivatives

                df = add_geometry_derivatives(df, self.timer)

            if self.recipe.get('add_tile_utm_derivatives', False):
                from openplaces.geo.tiles import add_tile_utm_derivatives

                cfg_utm = self.recipe['add_tile_utm_derivatives']
                df = add_tile_utm_derivatives(
                    df,
                    tile_id_col=cfg_utm.get('tile_id_col', 'tile_id'),
                    tile_type=cfg_utm.get('tile_type'),
                )

            if (~df.geometry.is_valid).any():
                df = fix_polygons(df)
            self.timer.mark(f'Clean geometries{_suffix}')

            if self.recipe.get('resolve_overlaps', False):
                df = clean_polygons(df)
                keep = self.recipe.get('keep_overlapping_polygons', None)
                recipe_col_names = list(self.recipe.get('columns', {}) or {})
                skip = {c for c in df.columns if '_id' in c} | {'geometry'}
                compare_cols = [c for c in df.columns if c not in skip]
                snippet_cols = (
                    [c for c in recipe_col_names if c in compare_cols]
                    + [c for c in compare_cols if c not in set(recipe_col_names)]
                )[:5]
                df = resolve_overlapping_polygons(
                    df,
                    keep=keep,
                    compare_cols=compare_cols,
                    snippet_cols=snippet_cols,
                )
                self.timer.mark(f'Resolve overlaps{_suffix}')

        # Attribute entities to administrative unit IDs via crosswalk
        # (Before admin ID index creation, which needs parent Admin ID)
        use_spatial_mask = (
            'process_by' in self.recipe
            and 'use_spatial_mask' in self.recipe['process_by']
            and self.recipe['process_by']['use_spatial_mask']
        )
        # An empty frame has nothing to attribute, and building the
        # crosswalk for one can fail outright: the recipe's `query` runs
        # first, so a state the recipe deliberately excludes arrives here
        # with zero rows, and asking `get_admin` for a level the excluded
        # state does not populate returns a frame without the crosswalk's
        # own column. New England is the live case -- its level 3 is
        # towns, so it has no level-4 subdivisions and
        # `US_admin-census-2025_admin4` filters it out, then died on the
        # crosswalk for the very partitions it had just emptied.
        if 'admin_id_crosswalk' in self.recipe and len(df):
            admin_id_crosswalk_dict = self.recipe['admin_id_crosswalk']
            admin_id_crosswalk_dict['admin_id'] = self.processing_chunk[
                'admin_id_to_process'
            ]
            admin_id_crosswalk = get_crosswalk(admin_id_crosswalk_dict, flip=True)

            missing_crosswalk_ids = set(df[admin_id_crosswalk.index.name]) - set(
                admin_id_crosswalk.index
            )
            if missing_crosswalk_ids:
                mask_unmatched = df[admin_id_crosswalk.index.name].isin(
                    missing_crosswalk_ids
                )
                # The join below drops these rows unconditionally, so the
                # warning has to be unconditional too. It was previously
                # gated behind `verbose`, which hid a total loss: every
                # Connecticut tract vanished from the census tile data
                # because the recipe crosswalked 2025 TIGER GEOIDs (which
                # carry the nine planning-region codes) against the 2021
                # admin recipe (which still had the eight legacy
                # counties). Nothing surfaced it. Keep the message short
                # so it stays usable at this volume; the full row dump it
                # used to build is what made gating tempting.
                n_dropped = int(mask_unmatched.sum())
                share = n_dropped / len(df) if len(df) else 0.0
                sample = sorted(str(v) for v in missing_crosswalk_ids)[:5]
                severity = 'ALL rows' if share == 1.0 else f'{share:.1%} of rows'
                recipe_id = self.recipe.get('recipe_id') or self.table_name
                warnings.warn(
                    f'Imperfect crosswalk in {recipe_id}: {n_dropped:,d} '
                    f'row(s) ({severity}) have an unmatched '
                    f'{admin_id_crosswalk.index.name} and will be dropped. '
                    f'Unmatched ids include: {", ".join(sample)}. '
                    'A large share usually means the recipe points at the '
                    'wrong vintage of the admin recipe.',
                    stacklevel=2,
                )
            df = df.join(
                admin_id_crosswalk, on=admin_id_crosswalk.index.name, how='inner'
            )
            cols_added += (
                [admin_id_crosswalk.name]
                if isinstance(admin_id_crosswalk, pd.Series)
                else list(admin_id_crosswalk)
            )
            self.timer.mark('Attribute admin IDs: crosswalk join')

        elif use_spatial_mask or ('overlay_admin_ids' in self.recipe):
            if self.verbose:
                print(
                    'Overlaying polygons with administrative boundaries. '
                    'This can take a while.'
                )
            if 'admin_geometries' not in self.download_partition:
                self._load_admin_geometries()
            if use_spatial_mask:
                admin_specs = self.recipe['process_by']
                admin_geometries = (
                    self.download_partition['admin_geometries']
                    .loc[[self.processing_chunk['admin_id_to_process']]]
                    .copy()
                )
            else:
                admin_specs = self.recipe['overlay_admin_ids']
                admin_geometries = self.download_partition['admin_geometries']

            # `admin_specs` is the whole `process_by` block in the
            # spatial-mask case, so it carries chunking keys that are not
            # overlay parameters. Allowlist rather than exclude, so a new
            # `process_by` key cannot break the call the way
            # `use_spatial_mask` did.
            _overlay_keys = {'admin_level', 'admin_id', 'include_overlays'}
            kwargs_overlay = {
                k: v for k, v in admin_specs.items() if k in _overlay_keys
            }
            cols_before = set(df.columns)
            # `fallback_to_existing: true` keeps an admin id the recipe
            # already derived (e.g. from a county-name column) for rows
            # whose location resolves no unit, and reports how often the
            # two disagree. Without it the overlay overwrites the column
            # in place and a row without coordinates loses its unit.
            _admin_col = admin_geometries.index.name
            existing_ids = None
            if (
                not use_spatial_mask
                and admin_specs.get('fallback_to_existing')
                and _admin_col in df.columns
            ):
                existing_ids = df[_admin_col].copy()
            df = overlay_admin_ids(
                df,
                admin_geometries=admin_geometries,
                timer=self.timer,
                **kwargs_overlay,
            )
            if existing_ids is not None:
                df[_admin_col], counts = prefer_located_admin_ids(
                    df[_admin_col], existing_ids
                )
                self._report_located_admin_ids(_admin_col, counts)
            _new_cols = [v for v in df.columns if v not in cols_before]
            cols_added += _new_cols

            # A spatial mask reads by the admin unit's *bounding box*, so a
            # non-rectangular unit picks up its neighbors' rows. The overlay
            # above is given only this unit's geometry, so those rows come
            # back with a null admin id: drop them, or they are written into
            # this unit's output file (31k of 50k rows for a coastal county).
            # Keyed on the admin layer's own index name, not on whichever
            # columns the overlay added: `overlay_admin_ids` writes that
            # column in place, so a recipe that already maps or derives it
            # added no column at all and the drop never ran.
            if use_spatial_mask and _admin_col in df.columns:
                df = df[df[_admin_col].notna()]

        # Set index
        _entity = self.recipe.get('entity')
        _has_custom_index = (
            'set_index' in self.recipe
            or 'create_index' in self.recipe
            or 'index_function' in self.recipe
        )
        if (
            _entity is not None
            and str(_entity.entity_type) == 'parcel'
            and isinstance(df, gpd.GeoDataFrame)
            and not _has_custom_index
            and df.index.name != 'geo_id'
        ):
            df['geo_id'] = get_geo_ids(df, handle_duplicates=False)
            df.index = pd.Index(add_unique_suffix(df['geo_id']), name='parcel_id')
        elif 'set_index' in self.recipe:
            if self.recipe['set_index'] not in df:
                raise ValueError(
                    'Column not found to use as index: ' + str(self.recipe['set_index'])
                )
            if df[self.recipe['set_index']].duplicated().any():
                raise ValueError(
                    f"Duplicates found in '{self.recipe['set_index']}'. "
                    'Choose other index.\n\n'
                    + str(
                        df[df[self.recipe['set_index']].duplicated(keep=False)][
                            self.recipe['set_index']
                        ]
                        .sort_values()
                        .head(5)
                    )
                )
            df = df.set_index(self.recipe['set_index'])
        elif 'create_index' in self.recipe:
            if 'function' in self.recipe['create_index']:
                if not self.recipe['create_index']['function'].startswith(
                    'openplaces.'
                ):
                    raise ValueError(
                        'Function in `create_index` must start with `openplaces.`\n'
                        'Changing this would create a security risk (run any function).'
                    )
                index_function = self._load_function(
                    self.recipe['create_index']['function']
                )
                index_function_kwargs = self.recipe['create_index'].get('args', {})
                df = index_function(df, **index_function_kwargs)
            elif 'method' in self.recipe['create_index']:
                if self.recipe['create_index']['method'] == 'prefix':
                    df.index = pd.Index(
                        self.recipe['create_index']['prefix']
                        + df[self.recipe['create_index']['column']],
                        name=self.recipe['create_index']['name'],
                    )
        elif 'index_function' in self.recipe:
            if not self.recipe['index_function'].startswith('openplaces.'):
                raise ValueError(
                    'Function in `index_function` must start with `openplaces.`\n'
                    'Changing this would create a security risk (run any function).'
                )
            index_function = self._load_function(self.recipe['index_function'])
            df = index_function(df)
            self.timer.mark('Generate indices')

        # Drop observations by index
        if 'drop' in self.recipe:
            df = df.drop(self.recipe['drop'])

        # Double-check that the index has no duplicates
        if df.index.duplicated().any():
            raise ValueError(
                'Duplicated indices are not allowed in imported data.\n'
                'Change `index_function`, `create_index` or `set_index` column:\n'
                + str(df[df.index.duplicated(keep=False)].sort_index().head())
            )

        # Reorder columns
        if 'columns' in self.recipe:
            named_cols = [c for c in list(self.recipe['columns']) if c in df]
            cols_order = (
                named_cols
                + [
                    c
                    for c in df
                    if c.startswith('admin')
                    and c.endswith('_id_source')
                    and c not in named_cols
                ]
                + cols_added
            )
            if self.recipe.get('keep_unnamed_columns'):
                cols_order += [
                    c for c in df if c not in cols_order + ['geo_id', 'geometry']
                ]
            for geo_col in ['geo_id', 'geometry']:
                if geo_col in df:
                    cols_order += [geo_col]
            df = df[cols_order]

        # Standardized cross-comparable parcel matching key (after reorder so it
        # is always retained).
        df = self._add_parcel_id_local(df)

        return df

    def _report_located_admin_ids(self, column, counts):
        """Report how location-derived admin ids compared with the recipe's.

        Printed unconditionally, like the save messages: a share of rows
        moving to a neighboring unit changes what every downstream file
        of that unit holds, so it should not depend on `verbose`. The
        counts are kept on the instance (summed over chunks) for callers
        and tests.

        Parameters
        ----------
        column : str
            Admin id column that was assigned.
        counts : dict of str to int
            Output of `prefer_located_admin_ids`.
        """
        totals = getattr(self, 'located_admin_id_counts', None) or {}
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
        self.located_admin_id_counts = totals
        n_rows = sum(counts[k] for k in ('located', 'fallback', 'unresolved'))
        share = counts['disagree'] / counts['located'] if counts['located'] else 0.0
        print(
            f'  {column} by location: {counts["located"]:,d} of {n_rows:,d} '
            f'rows; {counts["disagree"]:,d} ({share:.1%}) moved from the '
            f'recipe-derived unit; {counts["fallback"]:,d} kept the '
            f'recipe-derived unit (no location match); '
            f'{counts["unresolved"]:,d} unresolved.'
        )

    def _get_labels(self, column):
        """Get code → label dict for a categorical column.

        Looks for a CSV file named '<column-with-dashes>-labels.csv' next
        to the recipe file.
        """
        labels_recipe_path = recipe_path(
            self.recipe['admin_id'],
            self.recipe.get('entity') or self.recipe.get('dataset'),
            filename=column.replace('_', '-') + '-labels.csv',
        )
        if labels_recipe_path.exists():
            labels = pd.read_csv(labels_recipe_path)
            return labels.set_index(labels.columns[0])[labels.columns[1]].to_dict()
        return None

    def _load_function(self, path):
        """Import and return a function by dotted module path."""
        module, name = path.rsplit('.', 1)
        return getattr(importlib.import_module(module), name)
