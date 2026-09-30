"""Read and validate a recipe: get_recipe, get_recipe_dict,
get_recipe_by_id, get_recipe_id, coverage, the implicit
stacked-units layer.
"""

import inspect
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

# Looked up through the package at call time, not bound here: the tests
# patch these three names on `openplaces.recipe`, and a function that
# had imported them would keep the originals.
from openplaces import recipe as _recipe
from openplaces.core.constants import (
    RECIPE_STAGES,
    RETENTION_CLASSES,
    STANDARD_DIRS,
    STRING_SEPARATOR_BETWEEN_IDS,
)
from openplaces.core.schema import (
    AdminId,
    Entity,
    Source,
    cast_dataset_or_entity,
)
from openplaces.path import OpenPlacesReference, recipe_path


def get_recipe(*args, **kwargs):
    """Load recipe (.yaml, .csv or .xlsx)

    Parameters
    ----------
    args : tuple
        Arguments for `openplaces.path.recipe_path`
    kwargs : dict
        Keywords arguments. Those in `openplaces.path.OpenPlacesReference`
        and `openplaces.path.recipe_path` will be used to find the path,
        the remainder is passed to the reading functions:
        - yaml.safe_load()
        - pd.read_csv()
        - pd.read_excel()
    """

    # Separate keywords: those that don't go to the path go to reading
    recipe_path_kwargs = set(inspect.signature(OpenPlacesReference).parameters) | set(
        inspect.signature(recipe_path).parameters
    )
    path_kwargs = {k: v for k, v in kwargs.items() if k in recipe_path_kwargs}
    read_kwargs = {k: v for k, v in kwargs.items() if k not in recipe_path_kwargs}

    filepath = recipe_path(*args, **path_kwargs)

    if filepath.suffix in ['.csv', '.xlsx', '.xlsx']:
        # To avoid ambiguity between versions, only one tabular format
        # should exist for a given filename. Removing extensions in the
        # arguments for the recipe filepath is one way to enforce that.
        raise Exception(
            f'Remove extensions in filepath when using `get_recipe()`: {filepath.name}'
        )

    if filepath.with_suffix('.yaml').exists():
        return get_recipe_dict(filepath.with_suffix('.yaml'), *args, **kwargs)
    elif filepath.with_suffix('.csv').exists():
        recipe_table = pd.read_csv(filepath.with_suffix('.csv'), **read_kwargs)
    elif filepath.with_suffix('.xlsx').exists():
        recipe_table = pd.read_excel(filepath.with_suffix('.xlsx'), **read_kwargs)
    elif filepath.with_suffix('.xls').exists():
        recipe_table = pd.read_excel(filepath.with_suffix('.xls'), **read_kwargs)
    else:
        raise OSError('Not found: ' + str(filepath.with_suffix('.(yaml|csv|xlsx|xls)')))

    return recipe_table


def _cast_entity(entity):
    """Cast a raw dict (or already-cast Entity) to an Entity object."""
    if isinstance(entity, Entity):
        return entity
    if isinstance(entity.get('source'), dict):
        entity['source'] = Source(**entity['source'])
    return Entity(**entity)


def _cast_dataset(dataset):
    """Cast a raw dict/string (or already-cast DataSet/Entity) to whichever fits."""
    return cast_dataset_or_entity(dataset)


def get_recipe_dict(filepath, *args, **kwargs):
    """Read a recipe `.yaml` file as a dictionary, cast it to schema

    Parameters
    ----------
    filepath : pathlib.Path
        Filepath to .yaml file
    args : list
        Passed on from `get_recipe`
    kwargs : dict
        Passed on from `get_recipe`
    """
    with open(filepath, encoding='utf-8') as f:
        recipe_dict = yaml.safe_load(f)

    # Record the canonical recipe ID (the file stem), so a loaded recipe can
    # be traced back to its ID even when it carries a filename suffix
    recipe_dict['recipe_id'] = Path(filepath).stem

    # Get `admin_id` from arguments
    if len(args) > 0:
        admin_id_arg = args[0]
    elif 'admin_id' in kwargs:
        admin_id_arg = kwargs['admin_id']
    else:
        admin_id_arg = None

    # Cast AdminId from arguments
    if not isinstance(admin_id_arg, AdminId):
        admin_id_arg = AdminId(admin_id_arg)

    # Default stage to 'ingest' for recipes that pre-date the stage field
    if 'stage' not in recipe_dict:
        recipe_dict['stage'] = 'ingest'

    # Validated here rather than at the point of use: a typo
    # ('harmonise') loaded silently, ranked below every ingest recipe
    # in `find_entity_recipe_id`, and surfaced only when the
    # orchestrated job ran and argparse rejected the stage name.
    if recipe_dict['stage'] not in RECIPE_STAGES:
        stage = recipe_dict['stage']
        raise ValueError(
            f"Recipe 'stage' is '{stage}', which is not a known openplaces "
            'pipeline stage. Valid options:' + '\n- ' + '\n- '.join(RECIPE_STAGES)
        )

    # A patch recipe (`patches: <recipe id>`) amends that recipe's
    # pipeline for the units in its own scope and has no pipeline of its
    # own (see apply_recipe_patches). Checked at load so a county file
    # that forgot the one or wrote the other fails when read, not when
    # its county is curated.
    if recipe_dict.get('patches'):
        if not isinstance(recipe_dict.get('pipeline_patch'), list):
            raise ValueError(
                f'{recipe_dict["recipe_id"]} declares patches: but no '
                'pipeline_patch list.'
            )
        if recipe_dict.get('pipeline'):
            raise ValueError(
                f'{recipe_dict["recipe_id"]} declares both patches: and a '
                'pipeline; a patch recipe has no pipeline of its own.'
            )

    # Ensure that 'admin_id' exists in recipe
    if 'admin_id' not in recipe_dict:
        recipe_dict['admin_id'] = admin_id_arg

    # Cast AdminId from .yaml file
    if not isinstance(recipe_dict['admin_id'], AdminId):
        recipe_dict['admin_id'] = AdminId(recipe_dict['admin_id'])

    # Sanity check: are there any conflicting values?
    if admin_id_arg and (str(recipe_dict['admin_id']) != str(admin_id_arg)):
        raise ValueError(
            'Inconsistent `admin_id` in get_recipe(admin_id, ...) and recipe `.yaml`:\n'
            f'get_recipe: {admin_id_arg} {type(admin_id_arg)}\n'
            f'.yaml file: {recipe_dict["admin_id"]} {type(recipe_dict["admin_id"])}'
        )

    # Cast Entity (if there is one)
    if 'entity' in recipe_dict:
        recipe_dict['entity'] = _cast_entity(recipe_dict['entity'])

    # Cast DataSet (if there is one)
    if 'dataset' in recipe_dict:
        recipe_dict['dataset'] = _cast_dataset(recipe_dict['dataset'])

    # Cast additional_layers entities (if any)
    for layer_spec in recipe_dict.get('additional_layers', []):
        if 'entity' in layer_spec:
            layer_spec['entity'] = _cast_entity(layer_spec['entity'])

    _add_stacked_units_layer(recipe_dict)

    # Validate save_to (if present)
    if 'save_to' in recipe_dict and isinstance(recipe_dict['save_to'], dict):
        data_dir = recipe_dict['save_to'].get('data_dir')
        if data_dir is not None:
            if data_dir not in STANDARD_DIRS:
                raise ValueError(
                    f"Recipe 'save_to.data_dir' is '{data_dir}', which is not a "
                    'known openplaces directory. Valid options:\n- '
                    + '\n- '.join(sorted(STANDARD_DIRS))
                )
        retention = recipe_dict['save_to'].get('retention')
        if retention is not None and retention not in RETENTION_CLASSES:
            raise ValueError(
                f"Recipe 'save_to.retention' is '{retention}', which is not a "
                'known retention class. Valid options:\n- '
                + '\n- '.join(RETENTION_CLASSES)
            )

    return recipe_dict


#: Marker on the implicit property layer a parcel ingest recipe carries.
STACKED_UNITS_LAYER_KEY = 'stacked_units_layer'


def _add_stacked_units_layer(recipe_dict: dict) -> None:
    """Give a parcel ingest recipe the property layer its stacked units land in.

    Every parcel ingest table is split at ingest into one row per lot
    and a property row per stacked ownership record
    (`io.stacked_units`). The property rows are written as an
    `additional_layers` entry of the same recipe, so discovery, the
    property spine, readers and cleanup see them exactly like MassGIS's
    bundled property table. The entry is added here, at load time,
    rather than declared in every YAML: it is the default, and a recipe
    that declares its own property layer, opts out with
    `stacked_units: false`, or is not an ingest recipe gets none. The
    ingester never reads this layer from the source file; the marker
    tells its layer loop to skip it, and the parcel table's own
    processing writes it.
    """
    entity = recipe_dict.get('entity')
    if (
        entity is None
        or str(entity.entity_type) != 'parcel'
        or recipe_dict.get('stage') != 'ingest'
        or recipe_dict.get('stacked_units') is False
    ):
        return
    layers = recipe_dict.setdefault('additional_layers', [])
    for spec in layers:
        spec_entity = spec.get('entity')
        if spec_entity is not None and str(spec_entity.entity_type) == 'property':
            return
    spec = {
        'entity': Entity('property', entity.source, entity.version),
        # The column in which a unit names its lot. The unit's own
        # `parcel_id_local` stays its own (io.stacked_units), so a join
        # of this layer onto parcels pairs `lot_id_local` with the
        # parcel's `parcel_id_local`.
        'layer_key': 'lot_id_local',
        STACKED_UNITS_LAYER_KEY: True,
    }
    if isinstance(recipe_dict.get('save_to'), dict):
        # A per-table key: the layer saves where its parcel table saves.
        spec['save_to'] = dict(recipe_dict['save_to'])
    layers.append(spec)


def coverage_is_complete(recipe_id) -> bool:
    """Whether *recipe_id* declares complete coverage of its admin scope.

    A recipe may state `coverage: complete`, meaning every admin unit in
    its scope is expected to have data (NSI covers every U.S. county).
    A harmonize step that finds such a reference missing then raises
    instead of soft-skipping: the absence is an unfinished or broken
    ingest, not a data gap. A recipe without the key keeps the tolerant
    default - sources like Overture genuinely cover some states and not
    others, and their absence stays an expected, silent skip.
    """
    try:
        return _recipe.get_recipe_by_id(recipe_id).get('coverage') == 'complete'
    except OSError:
        # Only a missing recipe file reads as "no declaration, stay
        # tolerant". A recipe that exists but fails to load (bad YAML, a
        # value the schema rejects) must not be swallowed: it would be
        # reported as tolerant and soft-skipped, which is the exact
        # outcome this guard exists to prevent, and the caller has
        # already swallowed the load failure once.
        return False


def raise_if_coverage_complete(recipe_id, admin_id, cause=None):
    """Escalate a missing reference that declares complete coverage.

    Absence of a complete-coverage reference (NSI anywhere in the U.S.)
    is an unfinished or broken ingest, and skipping it silently degrades
    every vote downstream; callers invoke this before their tolerant
    soft-skip path.
    """
    if coverage_is_complete(recipe_id):
        raise RuntimeError(
            f'{recipe_id} declares complete coverage but has no data for '
            f'{admin_id}. Ingest it for this admin unit (or fix the '
            'failed load) rather than letting the pipeline run without '
            'it.'
        ) from cause


def get_recipe_by_id(recipe_id, **kwargs):
    """Shortcut to get recipe_id by its parts

    Assumes syntax: {admin_id}_{entity}_{filename}.{extension}

    admin_id or filename can be missing

    (Datasets for non-entities aren't yet supported)

    Parameters
    ----------
    recipe_id : str
        Identifier or a recipe
    kwargs : dict
        Keyword arguments will be passed on to get_recipe()
    """

    n_dots = Counter(recipe_id)['.']
    if n_dots == 0:
        filename_stem = recipe_id
        extension = None
    elif n_dots == 1:
        filename_stem, extension = recipe_id.split('.')
    else:
        raise ValueError('Recipe name cannot contain more than one dot.')

    remaining_parts = filename_stem.split('_')

    # Split off admin_id, if valid
    try:
        admin_id = AdminId(remaining_parts[0])
        remaining_parts = remaining_parts[1:]
    except ValueError:
        admin_id = None

    # Split off entity, if valid
    try:
        entity = Entity(remaining_parts[0])
        remaining_parts = remaining_parts[1:]
    except ValueError:
        entity = None

    # Split off dataset (or a linked entity), if valid
    try:
        dataset = cast_dataset_or_entity(remaining_parts[0])
        remaining_parts = remaining_parts[1:]
    except (ValueError, IndexError):
        dataset = None

    filename = remaining_parts.pop(0) if remaining_parts else None

    if remaining_parts:
        raise ValueError(f'Cannot interpret recipe_id {recipe_id}')

    if isinstance(filename, str) and isinstance(extension, str):
        filename += '.' + extension

    return get_recipe(
        admin_id,
        entity,
        dataset,
        filename=filename,
        **kwargs,
    )


def get_recipe_id(recipe: str | dict) -> str:
    """Return the canonical recipe ID of a loaded recipe.

    The ID is the recipe file's stem, recorded by get_recipe_dict at load
    time. For recipe dicts constructed without a file (e.g. in tests), the
    ID is rebuilt from admin_id and entity/dataset; filename suffixes cannot
    be recovered in that case.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID string (returned unchanged, minus a .yaml extension) or a
        loaded recipe dictionary.
    """
    if isinstance(recipe, str):
        return recipe.removesuffix('.yaml')
    if 'recipe_id' in recipe:
        return str(recipe['recipe_id']).removesuffix('.yaml')
    parts = []
    admin_id = recipe.get('admin_id')
    if admin_id is not None and len(str(admin_id)) > 0:
        parts.append(str(admin_id))
    entity_or_dataset = recipe.get('entity') or recipe.get('dataset')
    if entity_or_dataset is None:
        raise ValueError('Recipe has neither an entity nor a dataset.')
    parts.append(str(entity_or_dataset))
    return STRING_SEPARATOR_BETWEEN_IDS.join(parts)
