"""Recipe ids, versions, sources and provenance suffixes."""

import glob
import re
from functools import cache
from pathlib import Path

# Looked up through the package at call time, not bound here: the tests
# patch these three names on `openplaces.recipe`, and a function that
# had imported them would keep the originals.
from openplaces import recipe as _recipe
from openplaces.core.schema import (
    AdminId,
    DataSet,
    Entity,
    cast_dataset_or_entity,
)
from openplaces.path import recipe_path

_VERSION_CHUNK_REGEX = re.compile(r'(\d+)')


def version_sort_key(version) -> tuple:
    """Return a sortable key for a recipe version string.

    Versions are compared as raw strings almost everywhere, and the tree
    mixes formats within one scope and stage: US ingest footprints carry
    both 'v2' (microsoft) and '2026' (microsoftglobal, osm). As strings
    'v2' > '2026', so an un-sourced lookup selected the legacy layer, and
    a future 'v10' would lose to 'v2'.

    Digit runs compare numerically and rank above letter runs, so '2026'
    beats 'v2', 'v10' beats 'v2', and '2026pc' beats '2026'.

    Parameters
    ----------
    version : str or None
        Version string as declared on the recipe's entity or dataset.

    Returns
    -------
    tuple
        Key for use with sorted()/max(); comparable with any other key
        this function returns.
    """
    chunks = [c for c in _VERSION_CHUNK_REGEX.split(str(version or '')) if c]
    return tuple((1, int(c), '') if c.isdigit() else (0, 0, c) for c in chunks)


def find_recipe_id(admin_id, entity_or_dataset, filename=None, silent=False):
    """Find a recipe ID by admin_id and entity/dataset identifier.

    Parameters
    ----------
    admin_id : str
        Administrative unit identifier.
    entity_or_dataset : str
        Entity or dataset identifier string, may contain glob wildcards
        (e.g. 'parcel-*-*', 'admin-census-2021').
    filename : str, optional
        Filename stem to match within the recipe directory. When None
        (default), matches any .yaml file in the entity directory. A .yaml
        extension is appended automatically if absent.
    silent : bool
        If True, suppress the message printed when multiple recipes are found.
    """
    glob_recipe_path = recipe_path(admin_id, entity_or_dataset, filename=filename)
    recipe_paths_found = glob.glob(str(glob_recipe_path))
    if len(recipe_paths_found) == 0:
        return None
    elif len(recipe_paths_found) == 1:
        return Path(recipe_paths_found[0]).name
    recipe_paths_found = sorted(
        recipe_paths_found, key=lambda p: version_sort_key(Path(p).parent.name)
    )
    if not silent:
        print(
            f'Multiple recipes found for {admin_id} ({entity_or_dataset}):\n'
            + '\n'.join([Path(fp).name for fp in recipe_paths_found])
        )
    recipe_id = Path(recipe_paths_found[-1]).name
    if not silent:
        print(f'Picked last, sorted by version: {recipe_id}')
    return recipe_id


@cache
def iter_entity_sources() -> frozenset:
    """Return the ``(entity_type, source_id)`` pairs across all entity recipes.

    Scans the bundled recipes directory once (cached) and parses each recipe
    filename for its entity token. Files whose entity token does not parse are
    skipped. A bare entity token (e.g. ``footprint``) followed by a
    ``{theme}-{source}-{version}`` remainder — an entity+dataset enrich recipe
    such as ``US_footprint_built-n-stories-brails-2026`` — falls back to the
    dataset's own source id. Used to auto-generate the provenance suffix
    vocabulary so adding a new source needs no hardcoded list edits.
    """
    return frozenset(
        (entity, source) for entity, source, _v in iter_entity_source_versions()
    )


@cache
def iter_entity_source_versions() -> frozenset:
    """Return ``(entity_type, source_id, version)`` triples across all entity recipes.

    The scan behind :func:`iter_entity_sources`, kept separately because
    the provenance suffix vocabulary needs the version as well: the
    crosswalk enrichment names its evidence columns by the reference
    recipe's *version* (``wetland_share_fmv2026``), since a version names
    the data product where a source id names who produced it.
    """
    pairs: set[tuple[str, str | None, str | None]] = set()
    for filepath in (
        f for root in _recipe.recipe_roots() for f in root.rglob('*.yaml')
    ):
        parts = filepath.stem.split('_')
        try:
            AdminId(parts[0])
            parts = parts[1:]
        except ValueError:
            pass
        if not parts:
            continue
        try:
            entity = Entity(parts[0])
        except (ValueError, IndexError):
            continue
        source_id = getattr(entity.source, 'source_id', None) if entity.source else None
        if source_id is None and len(parts) > 1:
            try:
                source_id = DataSet(parts[1]).source.source_id
            except (ValueError, IndexError):
                pass
        version = None if entity.version is None else str(entity.version)
        pairs.add((str(entity.entity_type), source_id, version))
    return frozenset(pairs)


@cache
def provenance_suffixes() -> tuple[tuple[str, str], ...]:
    """Provenance suffix -> source key, auto-generated from existing recipes.

    For every ``(entity_type, source)`` pair known to the recipes, generate the
    column suffixes the harmonizer can produce: ``_{entity}_{source}`` and the
    bare ``_{source}`` fallback (e.g. ``_building_nsi`` and ``_nsi``;
    ``_footprint_fema`` and ``_fema``). Parcels are interchangeable, so they map
    by the entity-only ``_parcel``. Returned longest-first so a specific suffix
    wins over its bare fallback. No hardcoded list — adding a source recipe
    extends this automatically.
    """
    suffixes: dict[str, str] = {}
    for entity, source, version in iter_entity_source_versions():
        # A version that is not a bare year is a data-product name
        # (`fmv2026`), which the crosswalk enrichment uses as its suffix.
        # A bare year is left out: `_2026` would strip from any
        # unregistered column that happens to end in a year.
        if version and not version.isdigit():
            suffixes.setdefault(f'_{version}', version)
        if source is None:
            continue
        if entity == 'parcel':
            suffixes.setdefault('_parcel', 'parcel')
        else:
            suffixes.setdefault(f'_{entity}_{source}', source)
            suffixes.setdefault(f'_{source}', source)
    return tuple(sorted(suffixes.items(), key=lambda kv: len(kv[0]), reverse=True))


def split_provenance_suffix(name: str) -> tuple[str, str | None]:
    """Split a trailing provenance suffix off *name*; return (base, source)."""
    for suffix, source in provenance_suffixes():
        if name.endswith(suffix):
            return name[: -len(suffix)], source
    return name, None


def resolve_attribute_name(column: str) -> str:
    """Resolve a possibly provenance-suffixed column to its registry attribute.

    An exact registry entry always wins, so genuinely distinct attributes whose
    names merely end in a source-like token (``n_footprints_per_parcel``,
    ``priority_on_parcel``, ``parcel_id_local``) resolve to themselves. Only
    unregistered names fall back to stripping a provenance suffix
    (``improvement_value_parcel`` -> ``improvement_value``); a name that is
    neither registered nor suffixed is returned unchanged.
    """
    from openplaces.core.attribute_registry import load_registry

    if column in load_registry().index:
        return column
    return split_provenance_suffix(column)[0]


def source_id_from_recipe_id(recipe_id: str) -> str:
    """Extract the source id from a recipe id.

    A recipe id is `{admin_id}_{entity_or_theme}-{source}-{version}`,
    optionally followed by a filename suffix. The entity or dataset token
    is parsed with the same caster :func:`get_recipe_by_id` uses, so the
    source is read off the parsed object rather than by counting
    `-`-delimited fields (`'US_building-nsi-2022'` -> `'nsi'`;
    `'US_footprint_built-n-stories-brails-2026'` -> `'brails'`).

    Reading the *last* token used to be enough only because no committed
    recipe id carried a suffix: `'CO_parcel-igac-2026_rural'` yielded
    `'rural'` and `'US_admin-census-2025_admin3'` yielded `'admin3'`,
    which would have reached provenance tokens, column suffixes and
    licence keys as though the suffix were the source.

    Falls back to the last token when nothing in the id parses (an
    un-versioned or otherwise irregular recipe id).
    """
    parts = recipe_id.removesuffix('.yaml').split('_')
    try:
        AdminId(parts[0])
        parts = parts[1:]
    except ValueError:
        pass
    for token in parts:
        try:
            parsed = cast_dataset_or_entity(token)
        except (ValueError, IndexError):
            continue
        source_id = getattr(getattr(parsed, 'source', None), 'source_id', None)
        if source_id:
            return str(source_id)
    return parts[-1] if parts else recipe_id


def find_admin_recipe_id(admin_id, admin_level, silent=False):
    """Find the ID of an administrative data ingestion recipe

    Parameters
    ----------
    admin_id : str
        Administrative unit identifier
    admin_level : int
        Administrative level for which a recipe is sought.
    silent : bool
        If True, suppress the message printed when multiple recipes are found.
    """
    return find_recipe_id(
        admin_id, 'admin-*-*', filename=f'admin{admin_level}', silent=silent
    )
