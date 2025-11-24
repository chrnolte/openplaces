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
            processed = ()
        elif len(levels) == 1 and STRING_SEPARATOR_WITHIN_IDS in levels[0]:
            processed = tuple(levels[0].split(STRING_SEPARATOR_WITHIN_IDS))
        elif isinstance(levels, (list, tuple, set)):
            processed = tuple(levels)
        else:
            raise ValueError(f'`levels` is {type(levels)}. Cannot interpret:\n{levels}')

        object.__setattr__(self, 'levels', processed)

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
    # Replace any character that's not alphanumeric, underscore, or dash
    # with tilde (yes, also the dot '.', so we can identify directories)
    s = re.sub(r'[^a-zA-Z0-9_-]', '~', s)

    # Truncate to max length
    s = s[:max_length]

    # Ensure not empty
    return s if s else None
