"""
src/openplaces/core/schema.py

Defines AdminId, Entity, EntityType, Dataset, Source, Theme etc.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from .constants import (
    ESCAPE_DIR,
    STRING_SEPARATOR_BETWEEN_IDS,
    STRING_SEPARATOR_WITHIN_IDS,
)

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
    'people',  # demographic data (aggregate)
    'persons',  # data of persons, e.g. property owners (sensitive)
    'risk',  # floods, storms, wildfires
    'rules',  # zoning, conservation
]


@dataclass(frozen=True)
class AdminId:
    """Administrative identifier with flexible depth."""

    levels: tuple[str] = field(default_factory=tuple)

    def __init__(self, *levels: str):
        """Initialize AdminId with administrative levels."""
        if len(levels) == 0 or levels[0] is None:
            tuple_of_levels = ()
        elif len(levels) == 1 and STRING_SEPARATOR_WITHIN_IDS in levels[0]:
            tuple_of_levels = tuple(levels[0].split(STRING_SEPARATOR_WITHIN_IDS))
        elif isinstance(levels, (list, tuple, set)):
            tuple_of_levels = tuple(levels)
        else:
            raise ValueError(f'`levels` is {type(levels)}. Cannot interpret:\n{levels}')

        # Verify that AdminId is correct
        for i, level in enumerate(tuple_of_levels):
            if not isinstance(level, str) or not re.match('[A-Z0-9]{2,3}', level):
                raise ValueError(
                    f"Admin ID {levels} is invalid at level {i}: '{level}'."
                )

        object.__setattr__(self, 'levels', tuple_of_levels)

    def __str__(self) -> str:
        """Return string representation

        (e.g., 'US-MA-MI-001' or '' for global)"""
        return STRING_SEPARATOR_WITHIN_IDS.join(self.levels)

    def to_prefix(self) -> str:
        """Return prefix string with separator

        (e.g., 'US-MA-MI-001_' or '' for global)"""
        admin_id_str = str(self)
        return admin_id_str + STRING_SEPARATOR_BETWEEN_IDS if admin_id_str else ''

    def to_path(self) -> Path:
        """Return path directory structure for AdminId."""
        if not self.levels:
            return Path(ESCAPE_DIR)

        parts = list(self.levels) + [ESCAPE_DIR]

        return Path(*parts)

    def is_parent_of(self, admin_id: "AdminId") -> bool:
        """Check if this AdminId is a parent of another AdminId.

        A parent AdminId has fewer levels and matches all its levels
        with the start of the child's levels.

        Args:
            admin_id: The potential child AdminId to check

        Returns:
            True if self is a parent of admin_id, False otherwise

        Examples:
            >>> AdminId('US', 'MA').is_parent_of(AdminId('US', 'MA', 'MI'))
            True
            >>> AdminId('US', 'MA').is_parent_of(AdminId('US', 'CA'))
            False
            >>> AdminId('US', 'MA').is_parent_of(AdminId('US', 'MA'))
            False
        """
        if len(self.levels) >= len(admin_id.levels):
            return False

        return admin_id.levels[: len(self.levels)] == self.levels

    def is_parent_or_equal_of(self, admin_id: "AdminId") -> bool:
        """Check if this AdminId is a parent or equal of another AdminId.

        A parent AdminId has fewer levels and matches all its levels
        with the start of the child's levels.

        Args:
            admin_id: The potential child AdminId to check

        Examples:
            >>> AdminId('US', 'MA').is_parent_or_equal_of(AdminId('US', 'MA', 'MI'))
            True
            >>> AdminId('US', 'MA').is_parent_or_equal_of(AdminId('US', 'CA'))
            False
            >>> AdminId('US', 'MA').is_parent_or_equal_of(AdminId('US', 'MA'))
            True
        """
        if len(self.levels) > len(admin_id.levels):
            return False

        return admin_id.levels[: len(self.levels)] == self.levels

    def get_level(self):
        return len(self.levels) - 1


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
    # URL for the website that permits data access (with registration)
    portal_url: str | None = None
    # Option 1: direct download URL (if set, allows automated download)
    download_url: str | None = None
    # Option 2: URL of website listing (changing) download URLs
    download_url_source: str | None = None
    # Option 2: Regular expression (string pattern) to extract URL
    download_url_source_regex: str | None = None
    doi: str | None = None

    def __init__(
        self,
        source_id: str = None,
        portal_url: str = None,
        download_url: str = None,
        download_url_source: str = None,
        download_url_source_regex: str = None,
        doi: str = None,
    ):
        """Initialize AdminId with administrative levels."""

        if download_url and download_url_source:
            raise ValueError(
                'An entity can have a `download_url` or a `download_url_source`, '
                'not both.'
            )

        if download_url_source and not download_url_source_regex:
            raise ValueError(
                'An entity with a `download_url_source` must also have a '
                '`download_url_source_regex` (string pattern) to extract the URLs.'
            )

        self.source_id = sanitize(source_id) if source_id is not None else None
        self.portal_url = portal_url
        self.download_url = download_url
        self.download_url_source = download_url_source
        self.download_url_source_regex = download_url_source_regex
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
        if isinstance(entity_type, str) and STRING_SEPARATOR_WITHIN_IDS in entity_type:
            parts = entity_type.split(STRING_SEPARATOR_WITHIN_IDS)

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
        return STRING_SEPARATOR_WITHIN_IDS.join(
            [str(part) for part in parts if part is not None]
        )

    def to_prefix(self) -> Path:
        """Return path directory structure."""

        return str(self) + STRING_SEPARATOR_BETWEEN_IDS

    def to_path(self) -> Path:
        """Return path directory structure for entity."""

        parts = [self.entity_type, self.source, self.version]

        return Path(*[str(part) for part in parts if part is not None])


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
            and STRING_SEPARATOR_WITHIN_IDS in levels[0]
        ):
            levels = levels[0].split(STRING_SEPARATOR_WITHIN_IDS)

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
        return STRING_SEPARATOR_WITHIN_IDS.join(self.levels)

    def to_prefix(self) -> Path:
        """Return path directory structure."""

        return str(self) + STRING_SEPARATOR_BETWEEN_IDS

    def to_path(self) -> Path:
        """Return path directory structure with escape directories."""
        if not self.levels:
            return Path(ESCAPE_DIR)

        parts = list(self.levels)

        return Path(*parts)


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
            and STRING_SEPARATOR_WITHIN_IDS in theme
        ):
            parts = theme.split(STRING_SEPARATOR_WITHIN_IDS)
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
        return STRING_SEPARATOR_WITHIN_IDS.join(
            [str(self.theme), str(self.source), str(self.version)]
        )

    def to_prefix(self) -> Path:
        """Return path directory structure."""

        return str(self) + STRING_SEPARATOR_BETWEEN_IDS

    def to_path(self) -> Path:
        """Return path directory structure."""

        parts = [self.theme.to_path(), str(self.source), str(self.version)]

        return Path(*parts)


def sanitize(s, max_length=255):
    """Ensure that string is safe for filenames: only [a-zA-Z0-9_-].

    Args:
        s: String to sanitize
        max_length: Maximum filename length (default 255)

    Returns:
        Sanitized filename string with only alphanumeric, underscore, and dash
    """
    # Replace any character that's not alphanumeric, underscore, dash
    # or a standard wildcard (*, ?) with tilde
    s = re.sub(r'[^a-zA-Z0-9*?_-]', '~', s)

    # Truncate to max length
    s = s[:max_length]

    # Ensure not empty
    return s if s else None
