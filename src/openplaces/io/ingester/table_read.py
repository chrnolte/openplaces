"""Reading a source table: the recipe data, flat tables and
fixed-width files.
"""

from __future__ import annotations

import json
import shutil
import warnings

import geopandas as gpd
import pandas as pd
from pyogrio.errors import DataSourceError

from openplaces.config import can_prompt
from openplaces.core.constants import (
    ACCESS_EXTENSIONS,
    GEOPANDAS_EXTENSIONS,
    PANDAS_EXTENSIONS,
    ZIP_EXTENSIONS,
)
from openplaces.io import (
    find_latest_file_or_gdb,
    read_gdb_with_domains,
    unzip,
)
from openplaces.io.ingester.access import read_access_table


class _TableReadMixin:
    """Methods of :class:`~openplaces.io.ingester.table_ingester.TableIngester`
    (table_read).

    State lives on the TableIngester instance.
    """

    def _read_recipe_data(self, columns=None, data_path_override=None, **kwargs):
        """Read data from the resolved data path for this table's layer.

        Parameters
        ----------
        columns : list, optional
            Column names to read. Enables lightweight reads for FID prep.
        data_path_override : Path, optional
            Read from this path instead of ``download_partition['data_path']``,
            without mutating the shared partition state. Used by
            ``process_by.file_pattern`` (see :meth:`_resolve_file_pattern_path`),
            where each admin unit within the same download partition reads a
            different physical file.
        kwargs : dict
            Passed to the underlying reader (e.g. fids, bbox,
            read_geometry, fid_as_index).
        """
        warnings.filterwarnings('ignore', 'received a polygon with more than 100 parts')

        if 'encoding' in self.recipe:
            kwargs['encoding'] = self.recipe['encoding']
        # A single malformed shape (a ring with fewer than four points,
        # as in one of Montgomery County TX's parcels) otherwise makes
        # pyogrio raise on the whole layer. `on_invalid: warn` or
        # `ignore` keeps the row with an empty geometry instead;
        # `clean_polygons` then drops it like any other empty shape.
        if 'on_invalid' in self.recipe:
            kwargs['on_invalid'] = self.recipe['on_invalid']

        if columns:
            timer_suffix = (
                ', '
                + str(len(columns))
                + ' column'
                + ('(s)' if len(columns) > 1 else '')
            )
        else:
            timer_suffix = ''

        layer = self.recipe.get('layer')
        data_path = data_path_override or self.download_partition['data_path']

        # Match the extension case-insensitively: the *_EXTENSIONS sets are
        # lowercase, but agency exports routinely ship uppercase names --
        # the PACS appraisal roll members are all `.TXT`, which otherwise
        # falls through every branch to "suffix not yet interpreted".
        suffix = data_path.suffix.lower()

        if suffix == '.parquet':
            if 'fids' in kwargs:
                raise ValueError('`fid`-based selection might not work with `parquet`.')
            try:
                gdf = gpd.read_parquet(data_path, columns=columns, **kwargs)
            except ValueError as e:
                # A plain (non-geo) parquet partition -- e.g. one of several
                # per-admin-unit tables meant to be joined later, only one of
                # which carries geometry (see io.aggregate.join_partitions_by_index).
                if 'Missing geo metadata' not in str(e):
                    raise
                gdf = pd.read_parquet(data_path, columns=columns)
            self.timer.mark('Read parquet file' + timer_suffix, path=data_path)
        elif suffix == '.gdb':
            try:
                gdf = read_gdb_with_domains(
                    data_path, columns=columns, layer=layer, **kwargs
                )
            except DataSourceError as e:
                if 'Permission denied' in str(e):
                    print(
                        f'\n\033[33mPermission denied reading:\033[0m\n'
                        f'  {data_path}\n\n'
                        'This usually means the .gdb folder is locked by a file sync '
                        'app (e.g. Dropbox) or was only partially deleted.\n'
                    )
                    # An orchestrator job has no one to answer, and a
                    # bare input() there dies with an EOFError that
                    # names none of this. `can_prompt` is the same test
                    # the terms-consent gate uses.
                    if not can_prompt():
                        raise RuntimeError(
                            f'Cannot read {data_path}: permission denied, and '
                            'this run cannot ask whether to delete it. Remove '
                            'the folder manually (or run once interactively) '
                            'and re-run.'
                        ) from e
                    answer = input('Try to delete it now? [y/n] ').strip().lower()
                    if answer == 'y':
                        try:
                            shutil.rmtree(data_path)
                            raise RuntimeError('Deleted. Please re-run the ingestion.')
                        except Exception as del_err:
                            raise RuntimeError(
                                f'Could not delete: {del_err}\n'
                                'Remove it manually and re-run.'
                            )
                    raise RuntimeError('Remove the folder manually and re-run.')
                raise
            self.timer.mark('Read GDB file' + timer_suffix, path=data_path)
        elif suffix in GEOPANDAS_EXTENSIONS:
            try:
                gdf = gpd.read_file(data_path, layer=layer, columns=columns, **kwargs)
            except DataSourceError:
                raise OSError(
                    f'Failed to read data file:\n\n{data_path}\n\n'
                    'Possibly an incompletely unzipped file? '
                    'If so, delete manually, and re-run unzipping.'
                )
            self.timer.mark(
                f'Read vector file ({data_path.suffix})' + timer_suffix,
                path=data_path,
            )
        elif suffix in PANDAS_EXTENSIONS or suffix in ACCESS_EXTENSIONS:
            if 'fids' in kwargs:
                # A `process_by.admin_id_column` recipe re-enters this branch
                # once per admin unit with the same file and the same
                # columns, differing only in which rows it keeps. Parsing the
                # whole file every time made a statewide roll (New York
                # ORPTS: 62 counties over a multi-GB file) pay the parse 62
                # times. Cache the parsed frame on the download partition and
                # slice it instead; only one entry is kept, so moving on to
                # another file or table releases the previous frame.
                cache_key = (self.table_name, str(data_path), tuple(columns or ()))
                cached = self.download_partition.get('flat_table_cache')
                if cached is None or cached[0] != cache_key:
                    cached = (
                        cache_key,
                        self._read_flat_table(
                            data_path, columns, kwargs.get('encoding')
                        ),
                    )
                    self.download_partition['flat_table_cache'] = cached
                    self.timer.mark('Read data table' + timer_suffix, path=data_path)
                # `fids` (built by `_prepare_table_fid_filter` from a plain
                # read of this same file, so its index is this file's own
                # 0-based row order) is applied positionally: row order is
                # unchanged by `usecols`/dtype selection, so `.iloc` selects
                # exactly the rows the crosswalk built the filter from.
                # De-duplicated because a crosswalk mapping one raw code to
                # two admin units repeats a position, which would otherwise
                # write the same source row twice.
                fids = list(dict.fromkeys(kwargs['fids']))
                gdf = cached[1].iloc[fids].copy()
            else:
                gdf = self._read_flat_table(data_path, columns, kwargs.get('encoding'))
                self.timer.mark('Read data table' + timer_suffix, path=data_path)
        elif suffix in ZIP_EXTENSIONS:
            try:
                gdf = gpd.read_file(data_path, layer=layer, columns=columns, **kwargs)
                self.timer.mark('Read compressed file' + timer_suffix, path=data_path)
            except Exception:
                # `geopandas` cannot read this archive in place, so extract it
                # and read what came out. The extracted path is cached under
                # its own key rather than written back over
                # `download_partition['data_path']`: that entry is shared by
                # every table in the download partition (and is the sentinel
                # the "already unzipped" check and the heap cleanup use), so
                # overwriting it made every later table in the partition read
                # whatever file this one happened to extract.
                extracted_paths = self.download_partition.setdefault(
                    'extracted_data_paths', {}
                )
                extracted_path = extracted_paths.get(str(data_path))
                if extracted_path is None:
                    unzip(
                        data_path,
                        self.recipe_heap_dir,
                        members=self.recipe.get('extract_members'),
                        verbose=self.verbose,
                    )
                    extracted_path = find_latest_file_or_gdb(self.recipe_heap_dir)
                    if extracted_path is None:
                        raise OSError(
                            '`geopandas` could not read compressed file:'
                            f'\n\n{data_path}.\n\n'
                            'Could not find a dataset after unzipping to:\n\n'
                            f'{self.recipe_heap_dir}'
                        )
                    extracted_paths[str(data_path)] = extracted_path
                gdf = gpd.read_file(
                    extracted_path, layer=layer, columns=columns, **kwargs
                )
                self.timer.mark(
                    'Read unzipped file' + timer_suffix, path=extracted_path
                )
        else:
            raise ValueError(f'Filepath suffix not yet interpreted: {data_path.suffix}')

        warnings.filterwarnings(
            'default', 'received a polygon with more than 100 parts'
        )
        return gdf

    def _read_flat_table(self, data_path, columns, encoding):
        """Read one flat (non-spatial) source file in full.

        Covers the fixed-width, JSON, flat XML, spreadsheet,
        delimited-text and Microsoft Access layouts a recipe can
        declare. Row filtering is the caller's job: this returns every
        row of the file, so a chunked recipe can parse once per
        download partition and slice per admin unit.

        Parameters
        ----------
        data_path : Path
            File to read.
        columns : list, optional
            Column names to read, where the layout supports selection.
        encoding : str, optional
            From the recipe's ``encoding`` key.
        """
        # `csv_dtype: str` reads every column as text, the robust choice
        # for messy flat dumps where a column mixes ints and strings.
        # A dict maps specific columns to dtypes.
        csv_dtype = self.recipe.get('csv_dtype')
        if csv_dtype == 'str':
            dtype = str
        elif isinstance(csv_dtype, dict):
            dtype = csv_dtype
        else:
            dtype = None

        suffix = data_path.suffix.lower()

        if suffix in ACCESS_EXTENSIONS:
            # One named table of the database (the recipe's `layer`),
            # reading only the requested columns: county databases run
            # to 0.5-1.5 GB. Typed at the source, so `csv_dtype` is
            # applied only when a recipe asks for it.
            df = read_access_table(data_path, self.recipe.get('layer'), columns)
            return df.astype(dtype) if dtype is not None else df

        if self.recipe.get('fixed_width'):
            return self._read_fixed_width(data_path, dtype, encoding=encoding)

        if suffix == '.json':
            # Flattened rather than read directly, because these APIs
            # nest: IBGE returns a municipality's state four levels
            # down at `microrregiao.mesorregiao.UF.sigla`.
            # `json_normalize` turns that into a dotted column name a
            # recipe can map like any other. `record_path` selects the
            # list of records when it is not the top-level object.
            with open(data_path, encoding='utf-8') as handle:
                payload = json.load(handle)
            df = pd.json_normalize(payload, record_path=self.recipe.get('record_path'))
            return df.astype(dtype) if dtype is not None else df

        if suffix == '.xml':
            # A flat table: one element per row under the root, fields
            # as attributes or child elements. `xml_xpath` selects the
            # row elements when they are not the root's children. The
            # standard-library parser avoids an lxml dependency (no
            # lxml in the environment); it reads the whole file, which
            # suits county extracts (tens of MB per table). XML holds
            # every value as text, so text is the default: inferring
            # types read 12-digit parcel ids as integers and dropped
            # their leading zeros. Recipes cast counts with to_numeric.
            df = pd.read_xml(
                data_path,
                xpath=self.recipe.get('xml_xpath', './*'),
                parser='etree',
                dtype=dtype if dtype is not None else str,
                encoding=encoding or 'utf-8',
            )
            if columns:
                df = df[[c for c in columns if c in df.columns]]
            return df

        if suffix in {'.xlsx', '.xls'}:
            header = self.recipe.get('header', 'infer')
            # 'infer' is read_csv's vocabulary; read_excel rejects it, so
            # an Excel recipe without a `header` key failed outright.
            # read_excel's own default is the first row.
            if header == 'infer':
                header = 0
            return pd.read_excel(
                data_path,
                sheet_name=self.recipe.get('sheet_name', 0),
                header=None if header in (None, 'none') else header,
                names=self.recipe.get('names'),
                dtype=dtype,
            )

        # low_memory=False avoids per-chunk dtype inference (the source
        # of mixed-type object columns that then fail Parquet writes).
        read_kwargs = {
            'delimiter': self.recipe.get('delimiter', ','),
            'low_memory': False,
        }
        if dtype is not None:
            read_kwargs['dtype'] = dtype
        # The recipe's `encoding` key is used directly by the `.gdb`,
        # geopandas and fixed-width branches but was not previously
        # forwarded here, so a plain non-UTF-8 flat file (e.g. a UTF-16
        # export) failed to decode regardless of a declared `encoding:`.
        if encoding:
            read_kwargs['encoding'] = encoding
        # pandas reads "NA" as missing by default, which erases Namibia's
        # country code and any source whose codes look like null
        # markers. A recipe that declares `keep_default_na: false` keeps
        # every string, with only an empty cell read as missing.
        if self.recipe.get('keep_default_na') is False:
            read_kwargs['keep_default_na'] = False
            read_kwargs['na_values'] = ['']
        # A source with a known malformed line (Harris County TX's
        # account roll carries one row with a stray tab) would otherwise
        # stop the whole read. The recipe opts in and names the rule
        # ('warn' drops the line and says so; 'skip' drops it silently);
        # the default stays pandas' 'error', so nothing is lost unasked.
        if self.recipe.get('on_bad_lines'):
            read_kwargs['on_bad_lines'] = self.recipe['on_bad_lines']
        return pd.read_csv(data_path, usecols=columns, **read_kwargs)

    def _read_fixed_width(self, data_path, dtype, encoding=None):
        """Read a fixed-width flat file using the recipe's ``fixed_width`` layout.

        ``fixed_width`` is an ordered list of ``[field_name, width]`` covering the
        whole record. Fields named ``filler`` are padding and are dropped after
        reading. Field values are stripped of their fixed-width padding.

        The layout need not cover the whole record: reading stops after the
        last declared field, so a recipe that wants six columns out of a
        9,714-character PACS appraisal row declares only as far as it needs.

        Parameters
        ----------
        data_path : Path
            Path to the flat file.
        dtype : type, dict, or None
            Passed to :func:`pandas.read_fwf` (defaults to ``str`` so zero-padded
            identifiers keep their leading zeros).
        encoding : str, optional
            From the recipe's ``encoding`` key. Legacy mainframe-style exports
            are rarely UTF-8 -- the Texas PACS appraisal roll is latin-1, and
            without this it dies on the first accented owner name.
        """
        spec = self.recipe['fixed_width']
        names, widths = [], []
        for i, field in enumerate(spec):
            name, width = field[0], int(field[1])
            names.append(f'filler_{i}' if name == 'filler' else name)
            widths.append(width)

        df = pd.read_fwf(
            data_path,
            widths=widths,
            names=names,
            dtype=dtype or str,
            encoding=encoding,
        )
        df = df.drop(columns=[c for c in df.columns if c.startswith('filler_')])
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.strip()
        return df

    # Preprocess
