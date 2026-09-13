"""
Read tables from Microsoft Access databases (`.mdb`, `.accdb`).

Some county assessors publish their appraisal data only as an Access
database (Lucas County, OH; Collin County, TX; several Florida property
appraisers). Two routes read one, tried in this order:

1. Jackcess (Java, Apache-2.0) through JPype, on every platform, when
   `jpype` imports and a Java runtime is found. The jar is not
   redistributed: it is downloaded once from Maven Central into the
   models directory and checked against a pinned sha256.
2. GDAL's ODBC driver through pyogrio, on Windows with the Microsoft
   Access ODBC driver installed (it ships with Office and with the free
   Access Database Engine redistributable).

Otherwise the read raises, naming both routes. Pure-Python
`access-parser` was measured and rejected: it returns NUMERIC columns
as raw bytes and fails outright on some tables (see
plans/florida-county-property-appraiser-bedrooms-bathrooms.md,
decision B). `mdbtools` has a conda-forge build for linux-64 only.
"""

import hashlib
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import requests

from openplaces.config import cfg
from openplaces.io import request_headers

JACKCESS_VERSION = '5.0.0'
JACKCESS_URL = (
    'https://repo1.maven.org/maven2/com/healthmarketscience/jackcess/jackcess/'
    f'{JACKCESS_VERSION}/jackcess-{JACKCESS_VERSION}.jar'
)
# Pinned so a changed or substituted jar is refused, not run.
JACKCESS_SHA256 = 'dd05c20e6123ba088b3b4360bfab62a18af5a51545c919071e4fc555809b9ab0'


class AccessReaderUnavailableError(RuntimeError):
    """Neither route to an Access database is available on this machine."""


def _jar_path() -> Path:
    name = f'jackcess-{JACKCESS_VERSION}.jar'
    return Path(cfg.models_dir) / 'external' / 'jackcess' / name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def jackcess_jar() -> Path:
    """Return the local Jackcess jar, downloading and verifying it once.

    Returns
    -------
    pathlib.Path
        The jar in the models directory, matching `JACKCESS_SHA256`.

    Raises
    ------
    OSError
        If the downloaded jar does not match the pinned sha256; nothing
        is kept in that case.
    """
    destination = _jar_path()
    if destination.exists() and _sha256(destination) == JACKCESS_SHA256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix='.jackcess.', suffix='.tmp', delete=False
    ) as handle:
        temp_path = Path(handle.name)
        with requests.get(
            JACKCESS_URL, stream=True, timeout=120, headers=request_headers()
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=1 << 16):
                if chunk:
                    handle.write(chunk)
    if _sha256(temp_path) != JACKCESS_SHA256:
        temp_path.unlink(missing_ok=True)
        raise OSError(
            f'The Jackcess jar downloaded from {JACKCESS_URL} does not match '
            f'its pinned sha256 ({JACKCESS_SHA256}); it was not kept.'
        )
    temp_path.replace(destination)
    return destination


def _jvm_available() -> bool:
    """True when JPype imports and can find a Java runtime."""
    try:
        import jpype
    except ImportError:
        return False
    try:
        jpype.getDefaultJVMPath()
    except Exception:  # noqa: BLE001 - JVMNotFoundException and kin
        return False
    return True


def _odbc_available() -> bool:
    """True on Windows when GDAL offers a driver that opens Access files.

    Whether the Microsoft Access ODBC driver itself is installed is only
    known when GDAL opens the file; a missing driver fails there.
    """
    if sys.platform != 'win32':
        return False
    try:
        import pyogrio
    except ImportError:
        return False
    drivers = pyogrio.list_drivers()
    return 'PGeo' in drivers or 'ODBC' in drivers


# Held so the JVM cannot collect the configured logger (java.util.logging
# keeps only weak references to loggers).
_JACKCESS_LOGGER = None


def _quiet_jackcess_logging() -> None:
    """Show Jackcess's own log records only at SEVERE.

    Jackcess warns when it opens an index read-only because the file
    uses a text sort order it cannot write (Lucas County's condo table,
    for one). This reader never writes, so the warning is noise.
    """
    import jpype

    global _JACKCESS_LOGGER
    logger = jpype.JClass('java.util.logging.Logger').getLogger(
        'com.healthmarketscience.jackcess'
    )
    logger.setLevel(jpype.JClass('java.util.logging.Level').SEVERE)
    _JACKCESS_LOGGER = logger


def _start_jvm() -> None:
    import jpype

    jar = str(jackcess_jar())
    if not jpype.isJVMStarted():
        # JPype loads its native bridge through System.load, which JDK 24+
        # warns about unless native access is granted. The option exists
        # since JDK 17, the minimum the `access` extra declares.
        jpype.startJVM(
            '--enable-native-access=ALL-UNNAMED',
            classpath=[jar],
            convertStrings=True,
        )
        _quiet_jackcess_logging()
        return
    # A JVM started elsewhere cannot take a classpath entry afterwards.
    try:
        jpype.JClass('com.healthmarketscience.jackcess.DatabaseBuilder')
    except TypeError as exc:
        raise AccessReaderUnavailableError(
            'A Java VM is already running without Jackcess on its classpath; '
            'read Access files before starting Java for anything else.'
        ) from exc


def _to_python(value):
    """Convert one Jackcess cell to a plain Python value."""
    if value is None or isinstance(value, str | bool | int | float):
        return value
    name = str(value.getClass().getName())
    if name in ('java.math.BigDecimal', 'java.lang.Double', 'java.lang.Float'):
        # Access NUMERIC and currency columns are BigDecimal: read them
        # as floats, never as text.
        return float(value.doubleValue())
    if name in ('java.lang.Integer', 'java.lang.Short', 'java.lang.Byte'):
        return int(value.longValue())
    if name == 'java.lang.Long':
        return int(value.longValue())
    if name == 'java.lang.Boolean':
        return bool(value.booleanValue())
    if name.startswith('java.time.'):
        return pd.Timestamp(str(value.toString()))
    return str(value.toString())


def _read_with_jackcess(path: Path, table: str, columns) -> pd.DataFrame:
    import jpype

    _start_jvm()
    database_builder = jpype.JClass('com.healthmarketscience.jackcess.DatabaseBuilder')
    java_file = jpype.JClass('java.io.File')
    database = database_builder(java_file(str(path))).setReadOnly(True).open()
    try:
        source = database.getTable(table)
        if source is None:
            names = sorted(str(name) for name in database.getTableNames())
            raise KeyError(f'No table {table!r} in {path.name}; tables: {names}')
        names = [str(column.getName()) for column in source.getColumns()]
        wanted = _selected(columns, names, table)
        values = {name: [] for name in wanted}
        for row in source:
            for name in wanted:
                values[name].append(_to_python(row.get(name)))
    finally:
        database.close()
    return pd.DataFrame(values, columns=wanted)


def _read_with_odbc(path: Path, table: str, columns) -> pd.DataFrame:
    import pyogrio

    layers = [str(name) for name, _ in pyogrio.list_layers(path)]
    if table not in layers:
        raise KeyError(f'No table {table!r} in {path.name}; tables: {sorted(layers)}')
    names = list(pyogrio.read_info(path, layer=table)['fields'])
    wanted = _selected(columns, names, table)
    frame = pyogrio.read_dataframe(
        path, layer=table, columns=wanted, read_geometry=False
    )
    return pd.DataFrame(frame)[wanted]


def _selected(columns, names, table) -> list[str]:
    if not columns:
        return list(names)
    missing = [column for column in columns if column not in names]
    if missing:
        raise KeyError(f'Table {table!r} has no column(s) {missing}; it has {names}')
    return list(columns)


def read_access_table(path, table, columns=None) -> pd.DataFrame:
    """Read one table of a Microsoft Access database into a DataFrame.

    Parameters
    ----------
    path : str or pathlib.Path
        The `.mdb` or `.accdb` file.
    table : str
        Table to read. An Access database holds many tables, so a
        recipe names one with its `layer` key.
    columns : list of str, optional
        Columns to read; all columns when omitted. Reading only the
        mapped columns matters: county databases run to 0.5-1.5 GB.

    Returns
    -------
    pandas.DataFrame
        One row per table row. NUMERIC and currency columns arrive as
        floats, text as strings, dates as timestamps.

    Raises
    ------
    ValueError
        If no table is named.
    AccessReaderUnavailableError
        If neither Jackcess (JPype plus a Java runtime) nor, on Windows,
        GDAL's Access ODBC driver is available.
    """
    if not table:
        raise ValueError(
            'An Access database holds several tables; name the one to read '
            'with the recipe key `layer`.'
        )
    path = Path(path)
    if _jvm_available():
        try:
            return _read_with_jackcess(path, table, columns)
        except KeyError:
            raise
        except Exception as exc:  # noqa: BLE001 - fall back, then re-raise
            if not _odbc_available():
                raise
            warnings.warn(
                f'Jackcess could not read {path.name} ({exc}); '
                'falling back to the Windows ODBC driver.',
                stacklevel=2,
            )
    if _odbc_available():
        return _read_with_odbc(path, table, columns)
    raise AccessReaderUnavailableError(
        f'Cannot read {path.name}: no Access reader is available. Either '
        'install JPype and a Java runtime (`conda install -c conda-forge '
        'jpype1 openjdk`; Jackcess is then downloaded on first use), or on '
        'Windows install the Microsoft Access ODBC driver (Microsoft Access '
        'Database Engine), which GDAL uses through pyogrio.'
    )
