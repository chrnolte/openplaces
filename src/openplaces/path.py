"""
Standardized path generation for `openplaces` data files.
"""

import fnmatch
import importlib.metadata
import inspect
import json
import warnings
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from openplaces.config import cfg
from openplaces.core.constants import STRING_SEPARATOR_BETWEEN_IDS
from openplaces.core.schema import AdminId, DataSet, Entity, cast_dataset_or_entity

__all__ = [
    'path',
    'external_path',
    'external_dir',
    'raw_path',
    'cache_path',
    'heap_path',
    'heap_dir',
    'logs_path',
    'out_path',
    'share_path',
    'models_path',
    'reports_path',
    'code_path',
    'recipe_path',
    'recipe_roots',
    'recipe_root_records',
    'recipe_roots_footer',
    'spine_path',
    'OpenPlacesReference',
]

# The committed admin spine, resolved against the installed
# package rather than `cfg.code_root`: it ships with `openplaces`
# and is read by code well below the recipe layer.
# `io.admin_codes.registry` imports `spine_path` from here, so its
# established import path keeps working.
SPINE_DIR = Path(__file__).parent / 'recipes' / '_all' / 'admin' / 'spine' / '2026'

# The recipes that ship with `openplaces`. Other packages contribute
# their own directory through the `openplaces.recipes` entry-point
# group (see `recipe_roots`).
BUNDLED_RECIPES_DIR = Path(__file__).parent / 'recipes'
RECIPES_ENTRY_POINT_GROUP = 'openplaces.recipes'


RECIPE_ROOTS_METADATA_KEY = 'openplaces:recipe_roots'


def _external_recipe_root_records() -> list[dict]:
    """Recipe directories contributed by installed packages, described.

    Each `openplaces.recipes` entry point names a path, or a callable
    returning one. A broken entry point is reported and skipped rather
    than raised, so one misinstalled package cannot take the bundled
    recipes down with it. Each record carries the entry point's name and
    the distribution and version it came from, which is what a build
    records so that "the best source available at build time" can be
    named later.
    """
    records = []
    for entry_point in importlib.metadata.entry_points(group=RECIPES_ENTRY_POINT_GROUP):
        try:
            target = entry_point.load()
            root = Path(target() if callable(target) else target)
        except Exception as exc:  # noqa: BLE001 - any failure is a skip
            warnings.warn(
                f'Recipe entry point {entry_point.name!r} ({entry_point.value}) '
                f'could not be loaded and is ignored: {exc}',
                stacklevel=2,
            )
            continue
        if not root.is_dir():
            warnings.warn(
                f'Recipe entry point {entry_point.name!r} names {root}, which '
                'is not a directory, and is ignored.',
                stacklevel=2,
            )
            continue
        dist = getattr(entry_point, 'dist', None)
        records.append(
            {
                'root': str(root),
                'provider': entry_point.name,
                'distribution': getattr(dist, 'name', None),
                'version': getattr(dist, 'version', None),
            }
        )
    return records


def _external_recipe_roots() -> list[Path]:
    return [Path(record['root']) for record in _external_recipe_root_records()]


def _bundled_recipe_root_record() -> dict:
    """The bundled recipe directory, with the version it shipped in.

    From an installed package the version is the distribution's; from a
    checkout it is the commit the working tree is on, which is the
    honest identity of a recipe tree that changes between releases.
    """
    try:
        version = importlib.metadata.version('openplaces')
    except importlib.metadata.PackageNotFoundError:
        version = None
    commit = None
    head = cfg.code_root / '.git' / 'HEAD'
    try:
        ref = head.read_text(encoding='utf-8').strip()
        if ref.startswith('ref: '):
            commit = (cfg.code_root / '.git' / ref[5:]).read_text(encoding='utf-8')
        else:
            commit = ref
        commit = commit.strip()[:12] or None
    except OSError:
        commit = None
    return {
        'root': str(BUNDLED_RECIPES_DIR),
        'provider': 'bundled',
        'distribution': 'openplaces',
        'version': version,
        'commit': commit,
    }


@cache
def recipe_root_records() -> tuple[dict, ...]:
    """Every recipe root a build reads from, described, bundled first.

    A build's outputs record this in their parquet footer
    (`RECIPE_ROOTS_METADATA_KEY`) and a delivery lists it in its terms
    notice, because auto-discovery picks the most specific recipe that
    covers a unit from whatever roots are installed: without the list,
    "the best source available at build time" is not reproducible once
    installing a recipe package can change that set.
    """
    records = [_bundled_recipe_root_record()]
    seen = {Path(records[0]['root']).resolve()}
    for record in _external_recipe_root_records():
        resolved = Path(record['root']).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        records.append(record)
    return tuple(records)


@cache
def recipe_roots() -> tuple[Path, ...]:
    """Every directory recipes are read from, bundled first.

    The bundled directory wins on a name collision, so an external
    package cannot shadow a recipe that ships with `openplaces`; it can
    only add. External roots come from the `openplaces.recipes`
    entry-point group, in installation order. Cached for the process,
    like the recipe index itself.
    """
    return tuple(Path(record['root']) for record in recipe_root_records())


def recipe_roots_footer() -> str:
    """The recipe-root record as the JSON string a parquet footer carries."""
    return json.dumps(list(recipe_root_records()))


RECIPE_EXTENSIONS = ('.yaml', '.csv', '.xlsx', '.xls')


class OpenPlacesPath(type(Path())):
    """Path subclass with custom repr for clean Jupyter display."""

    def __repr__(self) -> str:
        """Return clean string representation for Jupyter output."""
        return f"'{str(self)}'"


@dataclass
class OpenPlacesReference:
    """
    Reference for `openplaces` files and directories.

    Combines admin_id, entity, dataset, filename, and root directory.
    """

    admin_id: AdminId
    entity: Entity
    dataset: DataSet | Entity
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

        # Handle dataset (may also be another Entity being attached/crosswalked)
        if isinstance(dataset, DataSet | Entity) or dataset is None:
            self.dataset = dataset
        else:
            # Parse string like 'bio-species'
            self.dataset = cast_dataset_or_entity(dataset)

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


def path(*args, default_extension='parquet', **kwargs):
    """Shortcut for ``OpenPlacesReference(*args, **kwargs).to_path()``."""
    return OpenPlacesReference(*args, **kwargs).to_path(
        default_extension=default_extension
    )


def external_path(*args, root=cfg.external_dir, use_prefix=False, **kwargs):
    return path(*args, root=root, use_prefix=use_prefix, **kwargs)


def external_dir(*args, **kwargs):
    return external_path(*args, as_dir=True, **kwargs)


def resolve_raster_path(raster_path) -> Path:
    """Resolve a recipe's raster reference against the configured root.

    Recipes name rasters by a path relative to the ``rasters`` directory, so
    the same recipe runs on any machine: only the root differs, and it is set
    once per user or project in the configuration. An absolute path is
    returned unchanged, which keeps ad-hoc callers and one-off notebooks
    working.

    Parameters
    ----------
    raster_path : str or pathlib.Path
        Relative path under the configured raster root, or an absolute path.

    Returns
    -------
    pathlib.Path
        The resolved location. It is not checked for existence; the caller
        opens it and reports a missing file in its own terms.
    """
    candidate = Path(raster_path)
    if candidate.is_absolute():
        return candidate
    return cfg.rasters_dir / candidate


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


def spine_path(level: int) -> Path:
    """Return the committed spine CSV for one admin level."""
    return SPINE_DIR / f'admin-spine-2026_admin{level}.csv'


def recipe_path(*args, root=None, **kwargs):
    """Path of a recipe file, searched across every recipe root.

    With no *root*, the first root (bundled first, then each external
    package's directory) holding the file wins, and a file no root holds
    resolves under the bundled directory so the error names the expected
    place. Pass *root* to address one directory explicitly.
    """
    # Get integer position of `filename` argument in OpenPlacesReference
    pos_filename = list(inspect.signature(OpenPlacesReference).parameters.keys()).index(
        'filename'
    )

    # Infer filename
    if 'filename' in kwargs:
        filename = kwargs.pop('filename')
    elif len(args) == pos_filename + 1:
        filename = args[pos_filename]
        args = args[:pos_filename]
    elif len(args) < pos_filename + 1:
        filename = None
    else:
        raise NotImplementedError(
            f'Not implemented: `recipe_path` with more than {pos_filename + 1} unnamed '
            'arguments.'
        )

    # If extension is absent, add default extension for recipes (.yaml)
    if isinstance(filename, str) and '.' not in filename:
        filename += '.yaml'

    if root is not None:
        return path(
            *args, filename=filename, root=root, default_extension='yaml', **kwargs
        )
    candidates = [
        path(*args, filename=filename, root=r, default_extension='yaml', **kwargs)
        for r in recipe_roots()
    ]
    for candidate in candidates:
        if candidate.exists() or any(
            candidate.with_suffix(ext).exists() for ext in RECIPE_EXTENSIONS
        ):
            return candidate
    return candidates[0]


def path_matches_pattern(path: str, pattern: str) -> bool:
    """Check if path is a resolved instance of pattern with wildcards.

    Parameters
    ----------
    path : str
        Concrete path to check.
    pattern : str
        Path pattern with wildcards (*, ?, [seq], etc.).

    Returns
    -------
    bool
        True if path matches the pattern.
    """
    # Convert to Path objects for normalization
    concrete_path = Path(path)
    pattern_path = Path(pattern)

    # Must have same number of parts
    if len(pattern_path.parts) != len(concrete_path.parts):
        return False

    # Check each part against pattern
    return all(
        fnmatch.fnmatch(concrete_path_part, pattern_path_part)
        for pattern_path_part, concrete_path_part in zip(
            pattern_path.parts, concrete_path.parts
        )
    )
