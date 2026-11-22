"""Reading a source table: the recipe data, flat tables and
fixed-width files.
"""

from __future__ import annotations

import codecs
import io
import json
import shutil
import warnings
from pathlib import Path

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

_UTF16_NAMES = {'utf-16', 'utf16', 'utf_16'}


def _is_valid_utf8(path, chunk_bytes=1 << 20):
    """Whether the whole file at *path* decodes as UTF-8."""
    decoder = codecs.getincrementaldecoder('utf-8')()
    try:
        with open(path, 'rb') as handle:
            while chunk := handle.read(chunk_bytes):
                decoder.decode(chunk)
        decoder.decode(b'', final=True)
    except UnicodeDecodeError:
        return False
    return True


def resolve_utf16_byte_order(path, encoding, fallback=None, sample_bytes=4096):
    """Name the encoding of a file declared UTF-16 that carries no
    byte-order mark.

    Python's 'utf-16' codec reads the byte order from the mark at the
    start of the file and raises "UTF-16 stream does not start with
    BOM" without it. A file without the mark is still unambiguous when
    its text is mostly ASCII: every second byte is zero, at odd offsets
    for little-endian and at even ones for big-endian. That pattern is
    read from the first *sample_bytes*.

    A file that shows neither is 8-bit text, as when one vintage of a
    multi-file source was exported differently. It is read only where
    the recipe names a *fallback* (its encoding_fallback key): as UTF-8
    when the whole file is valid UTF-8, which 8-bit text with non-ASCII
    characters almost never is by accident, and as *fallback* otherwise.
    Without a fallback the error names the file and its first bytes.

    Parameters
    ----------
    path : Path
        File the recipe reads.
    encoding : str or None
        From the recipe's encoding key. Anything other than 'utf-16'
        is returned unchanged, as is 'utf-16' on a file with the mark.
    fallback : str, optional
        8-bit encoding of a file that is not UTF-16, e.g. 'cp1252'.
    sample_bytes : int, optional
        How many leading bytes to test.

    Returns
    -------
    str or None
        *encoding*; 'utf-16-le' or 'utf-16-be' for a UTF-16 file
        without the mark; 'utf-8' or *fallback* for an 8-bit file.

    Raises
    ------
    UnicodeError
        When a file declared 'utf-16' has no mark and no UTF-16 byte
        pattern, and no fallback is given.
    """
    if not encoding or str(encoding).lower() not in _UTF16_NAMES:
        return encoding
    path = Path(path)
    if not path.is_file():
        return encoding
    with open(path, 'rb') as handle:
        head = handle.read(sample_bytes)
    if len(head) < 2 or head[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return encoding
    pairs = len(head) // 2
    zero_odd = sum(1 for i in range(1, 2 * pairs, 2) if head[i] == 0)
    zero_even = sum(1 for i in range(0, 2 * pairs, 2) if head[i] == 0)
    # Mostly-ASCII text puts a zero in nearly every high byte; 90% of
    # pairs leaves room for accented or other non-ASCII characters.
    if zero_odd >= 0.9 * pairs and zero_even == 0:
        resolved = 'utf-16-le'
    elif zero_even >= 0.9 * pairs and zero_odd == 0:
        resolved = 'utf-16-be'
    elif fallback:
        resolved = 'utf-8' if _is_valid_utf8(path) else fallback
    else:
        raise UnicodeError(
            f'{path} is declared utf-16 but starts with no byte-order mark '
            f'and its bytes are not UTF-16 (first bytes: {head[:8].hex(" ")}). '
            'Check whether this file was exported in another encoding; '
            'encoding_fallback in the recipe names it.'
        )
    print(f'{path.name}: no byte-order mark, read as {resolved}.')
    return resolved


def requote_all_quoted_fields(path, delimiter, encoding):
    """Escape a file whose every field is quoted but whose quotes are not.

    Some exports enclose every field in double quotes and write a quote
    or delimiter inside a field as is, never doubled. A CSV parser then
    ends the field at the inner quote and either finds too many fields
    or, without a word, shifts the rest of the row into the wrong
    columns. In such a file, a field boundary is exactly
    quote-delimiter-quote, so each line is split there, its outer
    quotes removed, and the fields written back with their quotes
    doubled, which any CSV parser reads correctly.

    Parameters
    ----------
    path : Path
        File to read.
    delimiter : str
        Field delimiter.
    encoding : str or None
        Text encoding of the file; UTF-8 when None.

    Returns
    -------
    tuple of (io.StringIO, int)
        The escaped text, and how many lines held a quote or delimiter
        inside a field.

    Raises
    ------
    ValueError
        When a nonempty line does not start and end with a quote: the
        file is not quoted throughout, or a field spans lines.
    """
    boundary = '"' + delimiter + '"'
    with open(path, encoding=encoding or 'utf-8', newline='') as handle:
        lines = handle.read().split('\n')
    out = []
    n_repaired = 0
    for number, line in enumerate(lines, start=1):
        line = line.removesuffix('\r')
        if not line:
            continue
        if len(line) < 2 or line[0] != '"' or line[-1] != '"':
            raise ValueError(
                f'{Path(path).name}, line {number}: all_fields_quoted expects '
                'every line to start and end with a quote.'
            )
        fields = line[1:-1].split(boundary)
        if any('"' in f or delimiter in f for f in fields):
            n_repaired += 1
        out.append('"' + boundary.join(f.replace('"', '""') for f in fields) + '"')
    return io.StringIO('\n'.join(out) + '\n'), n_repaired


class _TableReadMixin:
    """Methods of :class:`~openplaces.io.ingester.table_ingester.TableIngester`
    (table_read).

    State lives on the TableIngester instance.
    """

    def _read_recipe_data(
        self, columns=None, data_path_override=None, layer_override=None, **kwargs
    ):
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
        layer_override : str, optional
            Read this layer instead of the recipe's ``layer``. Used by
            ``process_by.layer_pattern`` (see :meth:`_resolve_layer_pattern`),
            where each admin unit reads its own layer of one file.
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

        layer = layer_override or self.recipe.get('layer')
        data_path = data_path_override or self.download_partition['data_path']

        joined = (getattr(self, 'download_partition', None) or {}).get('join_members')
        if joined and data_path_override is None:
            if 'fids' in kwargs:
                raise ValueError('`join_members` does not support `fid` selection.')
            df = self._read_joined_members(
                joined['members'], joined['key'], kwargs.get('encoding')
            )
            self.timer.mark('Read joined members' + timer_suffix, path=data_path)
            if columns:
                df = df[[c for c in columns if c in df.columns]]
            warnings.filterwarnings(
                'default', 'received a polygon with more than 100 parts'
            )
            return df

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
                # `gdb_domain_labels: false` keeps a coded field's codes
                # (PAD-US 'FED', 'OA'), where the source documents its
                # vocabulary by code and some fields carry no domain.
                gdf = read_gdb_with_domains(
                    data_path,
                    columns=columns,
                    layer=layer,
                    resolve_domains=self.recipe.get('gdb_domain_labels', True),
                    **kwargs,
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

    def _read_flat_table(self, data_path, columns, encoding, member=False):
        """Read one flat (non-spatial) source file in full.

        Covers the fixed-width, JSON, flat XML, spreadsheet,
        delimited-text and Microsoft Access layouts a recipe can
        declare. Row filtering is the caller's job: this returns every
        row of the file, so a chunked recipe can parse once per
        download partition and slice per admin unit. A recipe with
        ``read_members`` reads every extracted file matching that
        pattern beside *data_path*, in name order, as one table.

        Parameters
        ----------
        data_path : Path
            File to read.
        columns : list, optional
            Column names to read, where the layout supports selection.
        encoding : str, optional
            From the recipe's ``encoding`` key.
        member : bool, optional
            Read *data_path* alone, as one member of a ``read_members``
            set.
        """
        partition = getattr(self, 'download_partition', None) or {}
        members_glob = partition.get('read_members_glob')
        if members_glob and not member:
            files = sorted(
                p for p in Path(data_path).parent.glob(members_glob) if p.is_file()
            )
            return pd.concat(
                [
                    self._read_flat_table(p, columns, encoding, member=True)
                    for p in files
                ],
                ignore_index=True,
            )

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
        encoding = resolve_utf16_byte_order(
            data_path, encoding, self.recipe.get('encoding_fallback')
        )

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
        # Indiana's sales disclosure files quote every field but never
        # double a quote inside one, and one field even holds a literal
        # tab. pandas then splits that row in the wrong place (a
        # ParserError) or, worse, shifts values into the neighboring
        # columns without a word. `all_fields_quoted: true` splits on
        # quote-delimiter-quote instead and hands pandas a properly
        # escaped copy, so every other read option still applies.
        source = data_path
        if self.recipe.get('all_fields_quoted'):
            source, n_repaired = requote_all_quoted_fields(
                data_path, read_kwargs['delimiter'], encoding
            )
            read_kwargs.pop('encoding', None)
            print(
                f'{Path(data_path).name}: {n_repaired} line(s) hold a quote or '
                'delimiter inside a field, read as text (all_fields_quoted).'
            )
        # The Census Gazetteer pads its last header with spaces, so the
        # name never matches a mapping. A recipe that declares
        # `strip_column_names: true` reads its headers trimmed.
        if self.recipe.get('strip_column_names'):
            wanted = None if columns is None else {str(c).strip() for c in columns}
            df = pd.read_csv(
                source,
                usecols=None if wanted is None else lambda c: c.strip() in wanted,
                **read_kwargs,
            )
            df.columns = df.columns.str.strip()
            # A callable usecols skips a missing header silently, where
            # the list form raises; keep the list form's contract.
            absent = sorted((wanted or set()) - set(df.columns))
            if absent:
                raise ValueError(
                    f'Usecols do not match columns, columns expected but '
                    f'not found: {absent}'
                )
            return df
        return pd.read_csv(source, usecols=columns, **read_kwargs)

    def _read_joined_members(self, members, key, encoding=None):
        """Read several headerless delimited files and join them on a key.

        Declared as ``join_members: {key, members: [{file, fields}]}``.
        Each member is read without a header row, keeping only the
        columns its ``fields`` map names (0-based position to name), and
        the members are joined side by side on *key*, which every member
        must name. The key is read as text in every member, so a
        zero-padded record number matches across files. Other columns
        follow the recipe's ``csv_dtype`` like a flat table's.

        The join is strict: a key repeated within a member, or present in
        one member and missing from another, raises rather than dropping
        or duplicating rows, because such files are meant to be parallel
        record for record.

        Parameters
        ----------
        members : list of tuple
            ``(path, fields)`` per member, in join order.
        key : str
            Column name, present in every member's ``fields``, to join on.
        encoding : str, optional
            From the recipe's ``encoding`` key.

        Raises
        ------
        ValueError
            When a member does not name *key*, repeats a key value, or the
            members do not hold the same set of keys.
        """
        csv_dtype = self.recipe.get('csv_dtype')
        delimiter = self.recipe.get('delimiter', ',')
        joined = None
        for path, fields in members:
            names = {int(position): name for position, name in fields.items()}
            if key not in names.values():
                raise ValueError(f'join_members: {path.name} does not name {key!r}.')
            if csv_dtype == 'str':
                dtype = dict.fromkeys(names, str)
            else:
                lookup = csv_dtype if isinstance(csv_dtype, dict) else {}
                dtype = {p: lookup[n] for p, n in names.items() if n in lookup}
            dtype.update({p: str for p, n in names.items() if n == key})
            df = pd.read_csv(
                path,
                sep=delimiter,
                header=None,
                usecols=sorted(names),
                dtype=dtype,
                encoding=encoding,
                low_memory=False,
            ).rename(columns=names)
            if df[key].duplicated().any():
                raise ValueError(
                    f'join_members: {path.name} repeats {key} values '
                    f'({int(df[key].duplicated().sum())} rows).'
                )
            if joined is None:
                joined = df
                continue
            merged = joined.merge(df, on=key, how='outer', indicator=True)
            unmatched = int((merged['_merge'] != 'both').sum())
            if unmatched:
                raise ValueError(
                    f'join_members: {unmatched} {key} values are not in every '
                    f'member (last joined: {path.name}).'
                )
            joined = merged.drop(columns='_merge')
        return joined

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
