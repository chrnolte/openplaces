"""
src/openplaces/path.py

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

import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PosixPath, WindowsPath
from warnings import warn

from openplaces.config import cfg
from openplaces.core.constants import STRING_SEPARATOR_BETWEEN_IDS
from openplaces.core.schema import AdminId, DataSet, Entity

__all__ = [
    'path',
    'core_path',
    'external_path',
    'external_dir',
    'raw_path',
    'cache_path',
    'heap_path',
    'heap_dir',
    'log_path',
    'out_path',
    'share_path',
    'models_path',
    'reports_path',
    'code_path',
    'recipe_path',
    'OpenPlacesReference',
]


def dataset_to_path(dataset) -> Path:
    """Return path directory structure."""

    parts = [theme_to_path(dataset.theme), str(dataset.source), str(dataset.version)]

    return Path(*parts)


class OpenPlacesPath(type(Path())):
    """Path subclass with custom repr for clean Jupyter display."""

    def __repr__(self) -> str:
        """Return clean string representation for Jupyter output."""
        return f"'{str(self)}'"


@dataclass
class OpenPlacesReference:
    """
    Reference for `openplaces` files and directories.

    Combines admin_id, entity, theme, filename, root directory.
    """

    admin_id: AdminId
    entity: Entity
    dataset: DataSet
    filename: str
    root: Path
    by_admin: bool
    use_prefix: bool
    as_dir: bool

    def __init__(
        self,
        admin_id: str | AdminId = None,
        entity: str | Entity = None,
        dataset: str | DataSet = None,
        filename: str = None,
        root: str | Path = cfg.core_dir,
        by_admin: bool = True,
        use_prefix: bool = True,
        as_dir: bool = None,
    ):
        """
        Initialize OpenPlacesReference from flexible inputs.

        Examples:
            OpenPlacesReference('US-MA', 'property-massgis-2018', 'bio-species')
            OpenPlacesReference(AdminId('US', 'MA'), Entity(...), Theme(...))
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
        self.as_dir = as_dir

    def to_path(self, default_extension='parquet', as_dir=None) -> Path:
        """
        Build complete path with directory structure and filename.

        Parameters
        ----------
        default_extension : str
            Is 'parquet' by default. Set to None to suppress extension.
        as_dir : bool
            If True, return directory path.
        """

        # Is this an intersected dataset (`DataSet` by `Entity`)?
        # is_intersected_dataset = self.entity is not None and self.dataset is not None

        # Is this a directory?
        if as_dir in [True, False]:
            is_directory = as_dir
        elif self.as_dir in [True, False]:
            is_directory = self.as_dir
        else:
            # `as_dir` not set. Check if there's a filename or prefix
            prefix_elements = [
                p
                for p in [self.admin_id, self.entity, self.dataset]
                if p is not None and str(p) != ''
            ]
            is_directory = not self.filename and not (
                prefix_elements and self.use_prefix
            )

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
            return OpenPlacesPath(dir_path)

        # Get fileroot and extension
        if isinstance(self.filename, str) and '.' in self.filename:
            parts = self.filename.split('.')
            extension = parts[-1]
            fileroot = '.'.join(parts[:-1])
        else:
            extension = default_extension
            fileroot = self.filename or ''

        prefix_parts = [str(p) for p in prefix_elements] if self.use_prefix else []
        extension_suffix = f'.{extension}' if extension else ''
        if prefix_parts:
            if fileroot:
                filename_parts = prefix_parts + [fileroot + extension_suffix]
            else:
                # Merge the last prefix with the extension
                filename_parts = prefix_parts[:-1] + [
                    prefix_parts[-1] + extension_suffix
                ]
        else:
            if not fileroot:
                fileroot = '__null__'
            filename_parts = [fileroot + extension_suffix]

        final_path = dir_path / STRING_SEPARATOR_BETWEEN_IDS.join(filename_parts)

        return OpenPlacesPath(final_path)

    def __str__(self) -> str:
        """Return the full path as string."""
        return str(self.to_path())

    def __fspath__(self):
        """Support Path-like interface."""
        return str(self.to_path())


def path(*args, **kwargs):
    """Shortcut for: OpenPlacesReference(*args, **kwargs).to_path()"""
    return OpenPlacesReference(*args, **kwargs).to_path()


def external_path(*args, root=cfg.external_dir, use_prefix=False, **kwargs):
    return path(*args, root=root, use_prefix=use_prefix, **kwargs)


def external_dir(*args, **kwargs):
    return external_path(*args, as_dir=True, **kwargs)


def raw_path(*args, root=cfg.raw_dir, use_prefix=False, **kwargs):
    return path(*args, root=root, use_prefix=use_prefix, **kwargs)


def cache_path(*args, root=cfg.cache_dir, **kwargs):
    return path(*args, root=root, **kwargs)


def heap_path(*args, root=cfg.heap_dir, use_prefix=False, **kwargs):
    return path(*args, root=root, use_prefix=use_prefix, **kwargs)


def heap_dir(*args, **kwargs):
    return heap_path(*args, as_dir=True, **kwargs)


def logs_path(*args, root=cfg.logs_dir, **kwargs):
    return path(*args, root=root, **kwargs)


def out_path(*args, root=cfg.out_dir, **kwargs):
    return path(*args, root=root, **kwargs)


def share_path(*args, root=cfg.share_dir, **kwargs):
    return path(*args, root=root, **kwargs)


def models_path(*args, root=cfg.models_dir, use_prefix=False, **kwargs):
    return path(*args, root=root, use_prefix=use_prefix, **kwargs)


def reports_path(*args, root=cfg.reports_dir, use_prefix=False, **kwargs):
    return path(*args, root=root, use_prefix=use_prefix, **kwargs)


def code_path(*args):
    return cfg.code_root.joinpath(*args)


def recipe_path(
    *args,
    root=cfg.code_root.joinpath('src', 'openplaces', 'recipes'),
    use_prefix=False,
    source=None,
    **kwargs,
):
    if source and 'filename' in kwargs:
        raise ValueError(
            "`source` implies `filename='{source}.yaml'`. "
            'Use `filename` without source.'
        )

    if source:
        kwargs['filename'] = f'{source}.yaml'

    return path(*args, root=root, use_prefix=use_prefix, **kwargs)
