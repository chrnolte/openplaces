"""
Standardized path generation for `openplaces` data files.

Implements consistent naming conventions:
- AdminId_Entity-SourceId-Timestamp_Theme-Attribute-SourceId-Timestamp_filename.ext
- Entity examples: human, parcel, property, transaction, building
- Timestamps: flexible format (YYYY, YYYYMMDD, YYYYMMDDHHmmss, etc.)

Administrative Referencing:
- Directories with admin referencing use structure
  - Global data uses "_" placeholder: _/building/filename
- Directories without admin referencing use flat structure
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from openplaces.config import cfg

# __all__ = [
#     'core_path',
#     'external_path',
#     'raw_path',
#     'cache_path',
#     'heap_path',
#     'out_path',
#     'share_path',
#     'models_path',
#     'reports_path',
# ]

# Directories that use administrative referencing
ADMIN_REFERENCED_DIRS = {
    'core',
    'external',
    'cache',
    'heap',
    'out',
    'share',
    'models',
    'reports',
}

ENTITY_TYPES = [
    'admin',
    'building',
    'parcel',
    'person',
    'property',
    'transaction',
]

TOP_LEVEL_THEMATIC_DOMAINS = [
    'climate',  # temperature, precipitation, change
    'land',  # topography, geology, soils
    'landcover',  # land cover (usually remotely sensed)
    'water',  # rivers, lakes, coasts
    'bio',  # vegetation, species, land cover
    'built',  # buildings, infrastructure, roads & accessibility
    'people',  # demographics, ownership
    'risk',  # floods, storms, wildfires
    'rules',  # zoning, conservation
]

# Escape directory (to signal end of administrative folder depth)
ESCAPE_DIR = '_'

# String separator to use *between* the main parts of a OpenPlacesPath
STRING_SEPARATOR_BETWEEN = '_'

# String separator to use *within* the main parts of a OpenPlacesPath
STRING_SEPARATOR_WITHIN = '-'

# # Syntax of an administrative ID (e.g., 'USMAMI')
# AID_REGEX = '^[A-Z]{2}(?:[A-Z]{2}(?:[A-Z]{2}(?:[0-9]+)?)?)?$'

# # Minimum and maximum lengths of AID components
# AID_MIN, AID_MAX = [2, 2, 2], [2, 2, 3]

# # Syntax of a unit ID (e.g., 'g.core.2020'
# UNIT_REGEX = '^[a-z]{1}(?:\\.[a-z0-9._*?-]+)?$'

# # Syntax of a domain ID (e.g., '')
# DOMAIN_REGEX = '^[a-z]{2,}(?:\\.[A-Za-z0-9._*?-]+)?$'

# # Reserved extensions (forbidden to use as units or domains)
# EXT = {
#     'csv',
#     'dbf',
#     'gdb',
#     'gpkg',
#     'npy',
#     'pbf',
#     'ply',
#     'png',
#     'parquet',
#     'shp',
#     'tif',
#     'tsv',
#     'txt',
#     'xls',
#     'xlsx',
#     'zip',
# }

# # Extensions of companion files for shapefiles
# SHP_EXT = ['.cpg', '.dbf', '.prj', '.qpj', '.shp', '.shx', '.sbn', '.sbx']


@dataclass
class AdminId:
    """Administrative identifier with flexible depth."""

    levels: list[str] = field(default_factory=list)

    def __init__(self, *levels: str):
        """Initialize AdminId with administrative levels."""
        if len(levels) == 0 or levels[0] is None:
            self.levels = []
        elif len(levels) == 1 and STRING_SEPARATOR_WITHIN in levels[0]:
            self.levels = levels[0].split(STRING_SEPARATOR_WITHIN)
        elif isinstance(levels, (list, tuple, set)):
            self.levels = list(levels)
        else:
            raise ValueError(f'`levels` is {type(levels)}. Cannot interpret:\n{levels}')

    def __str__(self) -> str:
        """Return string representation

        (e.g., 'US-MA-MI-001' or '' for global)"""
        return STRING_SEPARATOR_WITHIN.join(self.levels)

    def to_prefix(self) -> str:
        """Return prefix string with separator

        (e.g., 'US-MA-MI-001_' or '' for global)"""
        admin_id_str = str(self)
        return admin_id_str + STRING_SEPARATOR_BETWEEN if admin_id_str else ''

    def to_path(self) -> Path:
        """Return path directory structure with escape directories."""
        if not self.levels:
            return Path(ESCAPE_DIR)

        parts = list(self.levels) + [ESCAPE_DIR]

        return Path(*parts)


@dataclass
class EntityType:
    """Entity type (e.g. parcel, property, transaction, building)"""

    entity_type: str

    def __init__(self, entity_type: str):
        """Initialize AdminId with administrative levels."""

        if entity_type not in ENTITY_TYPES:
            raise ValueError(
                f"Entity not recognized: '{entity_type}'. "
                'Rename or add to `openplaces.path.ENTITY_TYPES`.'
            )

        self.entity_type = entity_type

    def __str__(self) -> str:
        """Return string representation (e.g., 'property')"""
        return self.entity_type


@dataclass
class Source:
    """Data source with metadata."""

    source_id: str
    url: str | None = None
    doi: str | None = None

    def __init__(self, source_id: str = None, url: str = None, doi: str = None):
        """Initialize AdminId with administrative levels."""

        self.source_id = sanitize(source_id) if source_id is not None else None
        self.url = url
        self.doi = doi

    def __str__(self) -> str:
        return self.source_id if self.source_id is not None else ''


@dataclass
class Entity:
    """Entity, identified by entity type, source, and version/date"""

    entity_type: EntityType
    source: Source
    version: str

    def __init__(
        self,
        entity_type: [str, EntityType],
        source: [str, Source] = None,
        version: str = None,
    ):
        """Initialize AdminId with administrative levels."""

        # If the first passed string contains separators, assume it
        # contains the other parameters
        if isinstance(entity_type, str) and STRING_SEPARATOR_WITHIN in entity_type:
            parts = entity_type.split(STRING_SEPARATOR_WITHIN)

            if len(parts) == 3:
                entity_type, source, version = parts
            elif len(parts) == 2:
                entity_type, source = parts

        if isinstance(entity_type, EntityType):
            self.entity_type = entity_type
        else:
            self.entity_type = EntityType(entity_type)

        if isinstance(source, Source):
            self.source = source
        elif source is not None:
            self.source = Source(source)
        else:
            self.source = None

        if version:
            self.version = sanitize(str(version))
        else:
            self.version = None

    def __str__(self) -> str:
        parts = [self.entity_type, self.source, self.version]
        return STRING_SEPARATOR_WITHIN.join(
            [str(part) for part in parts if part is not None]
        )

    def to_path(self) -> Path:
        """Return path directory structure."""

        parts = [self.entity_type, self.source, self.version]

        return Path(*[str(part) for part in parts if part is not None])

    def to_prefix(self) -> Path:
        """Return path directory structure."""

        return str(self) + STRING_SEPARATOR_BETWEEN


@dataclass
class Theme:
    """Thematic identifier with flexible depth."""

    levels: list[str] = field(default_factory=list)

    def __init__(self, *levels: str):
        """Initialize Theme with levels."""
        if len(levels) == 0:
            raise ValueError('Empty themes are not allowed')

        # Accept passing of a sequence as first argument
        if isinstance(levels[0], (list, tuple, set)):
            levels = list(levels[0])

        # Split text
        if (
            len(levels) == 1
            and isinstance(levels[0], str)
            and STRING_SEPARATOR_WITHIN in levels[0]
        ):
            levels = levels[0].split(STRING_SEPARATOR_WITHIN)

        if levels[0] not in TOP_LEVEL_THEMATIC_DOMAINS:
            raise ValueError(
                f"'{levels[0]}' is not a registered top-level thematic domain.\n"
                'Pick from openplaces.path.THEMATIC_DOMAINS or add to it:\n- '
                + '\n- '.join(TOP_LEVEL_THEMATIC_DOMAINS)
            )

        if isinstance(levels, (list, tuple, set)):
            self.levels = list(levels)
        else:
            raise ValueError(f'`levels` is {type(levels)}. Cannot interpret:\n{levels}')

    def __str__(self) -> str:
        """Return string representation"""
        return STRING_SEPARATOR_WITHIN.join(self.levels)

    def to_path(self) -> Path:
        """Return path directory structure with escape directories."""
        if not self.levels:
            return Path(ESCAPE_DIR)

        parts = list(self.levels)

        return Path(*parts)

    def to_prefix(self) -> Path:
        """Return path directory structure."""

        return str(self) + STRING_SEPARATOR_BETWEEN


@dataclass
class DataSet:
    """Dataset for `openplaces`"""

    theme: Theme
    source: Source
    version: str

    def __init__(
        self,
        theme: [str, Theme],
        source: [str, Source] = None,
        version: str = None,
    ):
        """Initialize AdminId with administrative levels."""

        if (
            source is None
            and isinstance(theme, str)
            and STRING_SEPARATOR_WITHIN in theme
        ):
            parts = theme.split(STRING_SEPARATOR_WITHIN)
            for i, part in enumerate(parts[::-1]):
                if i == 0:
                    # Last
                    version = part
                elif i == 1:
                    # Second-to_last
                    source = part
            theme = parts[:-2]

        if isinstance(theme, Theme):
            self.theme = theme
        else:
            self.theme = Theme(theme)

        if isinstance(source, Source):
            self.source = source
        else:
            self.source = Source(source)

        if isinstance(version, str):
            self.version = version
        elif version:
            self.version = str(version)
        else:
            # If no version is provided, default to YYYYMMDD timestamp
            from datetime import datetime

            self.version = datetime.now().strftime("%Y%m%d")

    def __str__(self) -> str:
        return STRING_SEPARATOR_WITHIN.join(
            [str(self.theme), str(self.source), str(self.version)]
        )

    def to_path(self) -> Path:
        """Return path directory structure."""

        parts = [self.theme.to_path(), str(self.source), str(self.version)]

        return Path(*parts)

    def to_prefix(self) -> Path:
        """Return path directory structure."""

        return str(self) + STRING_SEPARATOR_BETWEEN


@dataclass
class OpenPlacesPath:
    """
    Fully resolved path for openplaces data files.

    Combines admin_id, entity, and theme into directory structure and filename.
    """

    admin_id: AdminId
    entity: Entity
    dataset: DataSet
    filename: str
    root: Path
    by_admin: bool
    use_prefix: bool

    def __init__(
        self,
        admin_id: str | AdminId = None,
        entity: str | Entity = None,
        dataset: str | DataSet = None,
        filename: str = None,
        root: str | Path = cfg.core_dir,
        by_admin: bool = True,
        use_prefix: bool = True,
    ):
        """
        Initialize OpenPlacesPath from flexible inputs.

        Examples:
            OpenPlacesPath('US-MA', 'property-massgis-2018', 'bio-species')
            OpenPlacesPath(AdminId('US', 'MA'), Entity(...), Theme(...))
        """

        # Handle admin_id
        if isinstance(admin_id, AdminId):
            self.admin_id = admin_id
        else:
            self.admin_id = AdminId(admin_id)

        # Handle entity
        if isinstance(entity, Entity) or entity is None:
            self.entity = entity
        else:
            self.entity = Entity(entity)

        # Handle dataset
        if isinstance(dataset, DataSet) or dataset is None:
            self.dataset = dataset
        else:
            # Parse string like 'bio-species'
            self.dataset = DataSet(dataset)

        self.filename = filename

        # Handle base directory
        if root is None:
            self.root = cfg.dir_core
        else:
            self.root = Path(root)

        self.by_admin = by_admin
        self.use_prefix = use_prefix

    def to_path(self, default_extension='parquet') -> Path:
        """
        Build complete path with directory structure and filename.

        Parameters
        ----------
        default_extension : str
            Is 'parquet' by default. Set to None to suppress extension.
        """

        # Is this an intersected dataset (`DataSet` by `Entity`)?
        is_intersected_dataset = self.entity is not None and self.dataset is not None

        # Is this a directory? Not if there's a filename, of if it's
        # an intersected dataset.
        is_directory = self.filename is None and not is_intersected_dataset

        # Start with base directory
        parts = [self.root]

        # If the directory structure is split by administrative level,
        # Add admin directory structure (might be a '_' for top level)
        if self.by_admin:
            parts.append(self.admin_id.to_path())

        # If entity exists, add entity directory structure
        if self.entity is not None:
            parts.append(self.entity.to_path())

        # Create dataset directories if there is no entity given
        if self.dataset is not None and self.entity is None:
            parts.append(self.dataset.to_path())

        # Build directory path
        dir_path = Path(*parts)

        if is_directory:
            return dir_path

        # Build filename with optional prefix
        if self.use_prefix:
            prefix_parts = [
                str(p)
                for p in [self.admin_id, self.entity, self.dataset]
                if p is not None and str(p) != ''
            ]
        else:
            prefix_parts = []

        # Get fileroot and extension
        if isinstance(self.filename, str) and '.' in self.filename:
            parts = self.filename.split('.')
            extension = parts[-1]
            fileroot = '.'.join(parts[:-1])
        elif self.filename is None and is_intersected_dataset:
            extension = default_extension
            fileroot = ''
        else:
            extension = ''
            fileroot = self.filename or ''

        if prefix_parts:
            if fileroot:
                filename_parts = prefix_parts + [
                    fileroot + (f'.{extension}' if extension else '')
                ]
            else:
                # Merge the last prefix with the extension
                filename_parts = prefix_parts[:-1] + [
                    prefix_parts[-1] + (f'.{extension}' if extension else '')
                ]
        else:
            if fileroot:
                filename_parts = [fileroot + (f'.{extension}' if extension else '')]
            else:
                raise ValueError(
                    'Must pass either `filename` or intersected dataset '
                    'with `use_prefix`.'
                )

        return dir_path / STRING_SEPARATOR_BETWEEN.join(filename_parts)

    def __str__(self) -> str:
        """Return the full path as string."""
        return str(self.to_path())

    def __fspath__(self):
        """Support Path-like interface."""
        return str(self.to_path())


def path(*args, **kwargs):
    """Shortcut for: OpenPlacesPath(*args, **kwargs).to_path()"""
    return OpenPlacesPath(*args, **kwargs).to_path()


def sanitize(s, max_length=255):
    """Ensure that string is safe for filenames: only [a-zA-Z0-9_-].

    Args:
        s: String to sanitize
        max_length: Maximum filename length (default 255)

    Returns:
        Sanitized filename string with only alphanumeric, underscore, and dash
    """
    # Replace any character that's not alphanumeric, underscore, or dash with tilde
    s = re.sub(r'[^a-zA-Z0-9_-]', '~', s)

    # Truncate to max length
    s = s[:max_length]

    # Ensure not empty
    return s if s else None


def external_path(*args, root=cfg.external_dir, use_prefix=False, **kwargs):
    return path(*args, root=root, use_prefix=use_prefix, **kwargs)


def raw_path(*args, root=cfg.raw_dir, use_prefix=False, **kwargs):
    return path(*args, root=root, use_prefix=use_prefix, **kwargs)


def cache_path(*args, root=cfg.cache_dir, **kwargs):
    return path(*args, root=root, **kwargs)


def heap_path(*args, root=cfg.heap_dir, use_prefix=False, **kwargs):
    return path(*args, root=root, use_prefix=use_prefix, **kwargs)


def out_path(*args, root=cfg.out_dir, **kwargs):
    return path(*args, root=root, **kwargs)


def share_path(*args, root=cfg.share_dir, **kwargs):
    return path(*args, root=root, **kwargs)


def models_path(*args, root=cfg.models_dir, by_admin=False, use_prefix=False, **kwargs):
    return path(*args, root=root, use_prefix=use_prefix, **kwargs)


def reports_path(
    *args, root=cfg.reports_dir, by_admin=False, use_prefix=False, **kwargs
):
    return path(*args, root=root, use_prefix=use_prefix, **kwargs)
