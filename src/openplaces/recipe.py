# recipe.py

# Functions to read data ingestion recipes

import inspect

import pandas as pd
import yaml

from openplaces.core.schema import AdminId, DataSet, Entity, Source
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
            f"Remove extensions in filepath when using `get_recipe()`: {filepath.name}"
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
        raise Exception('Not found: ' + str(filepath) + '.(yaml|csv|xlsx|xls)')

    return recipe_table


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
            f"get_recipe: {admin_id_arg} {type(admin_id_arg)}\n"
            f".yaml file: {recipe_dict['admin_id']} {type(recipe_dict['admin_id'])}"
        )

    # Cast Entity (if there is one)
    if 'entity' in recipe_dict:
        if 'source' in recipe_dict['entity'] and isinstance(
            recipe_dict['entity']['source'], dict
        ):
            recipe_dict['entity']['source'] = Source(**recipe_dict['entity']['source'])
        if isinstance(recipe_dict['entity'], dict):
            recipe_dict['entity'] = Entity(**recipe_dict['entity'])

    # Cast DataSet (if there is one)
    if 'dataset' in recipe_dict:
        if 'source' in recipe_dict['dataset'] and isinstance(
            recipe_dict['dataset']['source'], dict
        ):
            recipe_dict['dataset']['source'] = Source(
                **recipe_dict['dataset']['source']
            )
        if isinstance(recipe_dict['dataset'], dict):
            recipe_dict['dataset'] = DataSet(**recipe_dict['dataset'])

    return recipe_dict


def get_recipe_by_id(recipe_id, **kwargs):
    """Shortcut to get recipe_id by its parts

    Assumes syntax: {admin_id}_{entity}_{filename}.{extension}

    (Datasets for non-entities aren't yet supported)

    Parameters
    ----------
    recipe_id : str
        Identifier or a recipe
    kwargs : dict
        Keyword arguments will be passed on to get_recipe()
    """
    if '.' in recipe_id:
        filename_stem, extension = recipe_id.split('.')
    else:
        filename_stem = recipe_id
        extension = None
    filename_stem_parts = filename_stem.split('_')
    if len(filename_stem_parts) == 2:
        admin_id, entity = filename_stem_parts
        filename = None
    elif len(filename_stem_parts) == 3:
        admin_id, entity, filename = filename_stem_parts
    else:
        raise ValueError(f'Could not split recipe_id into its parts: {recipe_id}')

    if isinstance(filename, str) and isinstance(extension, str):
        filename += '.' + extension

    return get_recipe(
        admin_id,
        entity,
        filename=filename,
        **kwargs,
    )
