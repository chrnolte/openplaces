"""
Transformation engine for applying variable transformations to DataFrames.

This module provides a flexible system for transforming variables based on
YAML recipe specifications. It supports:
- Unary operations (log, arcsinh, power, etc.)
- Binary operations (arithmetic on two columns)
- Aggregate operations (sum, min, max across multiple columns)
- String remapping and reclassification
- Conditional transformations
- Date/time extractions
- Complex expressions
- Pattern-based transformations
"""

import os
import warnings
from collections.abc import Callable
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from openplaces.io.readers import get_admin
from openplaces.recipe import get_recipe_by_id

# Operations


def _parse_currency(x: pd.Series) -> pd.Series:
    """Parse currency-formatted strings while preserving invalid values as NA."""
    values = x.astype('string').str.strip()
    values = values.str.replace(r'[$,]', '', regex=True)
    values = values.str.replace(r'^\((.*)\)$', r'-\1', regex=True)
    return pd.to_numeric(values, errors='coerce')


def _resolve_century(x: pd.Series, pivot: int = 68) -> pd.Series:
    """Expand a 2-digit year to 4 digits using the POSIX ``%y`` convention.

    Values 0-``pivot`` map to 20xx; (``pivot``, 99] map to 19xx. Values
    already >= 100 pass through unchanged. Matches the ``strptime('%y')``
    rule (default pivot 68: 00-68 -> 2000-2068, 69-99 -> 1969-1999).
    """
    values = pd.to_numeric(x, errors='coerce')
    century = pd.Series(np.where(values <= pivot, 2000, 1900), index=values.index)
    return values.where(values >= 100, values + century)


UNARY_OPS: dict[str, Callable] = {
    'log': np.log,
    'arcsinh': np.arcsinh,
    'arcsinh_median_centered': lambda x: np.arcsinh(x / x.median()),
    'sqrt': np.sqrt,
    'exp': np.exp,
    'abs': np.abs,
    'power': lambda x, exponent: x**exponent,
    'parse_currency': _parse_currency,
    'resolve_century': _resolve_century,
    'to_numeric': lambda x: pd.to_numeric(x, errors='coerce'),
    'to_datetime': lambda x: pd.to_datetime(x, errors='coerce'),
}

BINARY_OPS: dict[str, Callable] = {
    'add': lambda x, y: x + y,
    'subtract': lambda x, y: x - y,
    'multiply': lambda x, y: x * y,
    'divide': lambda x, y: x / y,
    'power': lambda x, y: x**y,
    'mod': lambda x, y: x % y,
}

AGGREGATE_OPS: dict[str, Callable] = {
    'sum': lambda cols, fill_na=None: _aggregate_cols(cols, 'sum', fill_na),
    'min': lambda cols, fill_na=None: _aggregate_cols(cols, 'min', fill_na),
    'max': lambda cols, fill_na=None: _aggregate_cols(cols, 'max', fill_na),
    'mean': lambda cols, fill_na=None: _aggregate_cols(cols, 'mean', fill_na),
    'any_gt': lambda cols, threshold=0, fill_na=0: _any_threshold(
        cols, threshold, fill_na, '>'
    ),
    'any_lt': lambda cols, threshold=0, fill_na=0: _any_threshold(
        cols, threshold, fill_na, '<'
    ),
    'all_gt': lambda cols, threshold=0, fill_na=0: _all_threshold(
        cols, threshold, fill_na, '>'
    ),
}

CONDITIONAL_OPS: dict[str, Callable] = {
    'less_than': lambda x, threshold: (x < threshold).astype(int),
    'less_equal': lambda x, threshold: (x <= threshold).astype(int),
    'greater_than': lambda x, threshold: (x > threshold).astype(int),
    'greater_equal': lambda x, threshold: (x >= threshold).astype(int),
    'equal': lambda x, threshold: (x == threshold).astype(int),
    'not_equal': lambda x, threshold: (x != threshold).astype(int),
}

DATETIME_OPS: dict[str, Callable] = {
    'year': lambda x: x.dt.year,
    'month': lambda x: x.dt.month,
    'year_month': lambda x: x.dt.month + '-' + x.dt.month,
    'day': lambda x: x.dt.day,
    'dayofyear': lambda x: x.dt.dayofyear,
    'quarter': lambda x: x.dt.quarter,
    'year_quarter': lambda x: (
        x.dt.year.astype(str) + '-' + x.dt.month.sub(1).floordiv(3).add(1).astype(str)
    ),
    'year_continuous': lambda x: x.dt.year + x.dt.dayofyear.div(365),
}
TITLE_LOWERCASE = {
    'de',
    'del',
    'la',
    'el',
    'los',
    'las',
    'y',
    'e',
    'o',
    'al',
    'en',
    'a',
}
STRING_OPS: dict[str, Callable] = {
    'substring': lambda x, start, end=None: x.str[start:end],
    'upper': lambda x: x.str.upper(),
    'lower': lambda x: x.str.lower(),
    'title_smart': lambda x: x.apply(
        lambda s: (
            ' '.join(
                w.capitalize()
                if i == 0 or w.lower() not in TITLE_LOWERCASE
                else w.lower()
                for i, w in enumerate(str(s).split())
            )
            if pd.notna(s)
            else s
        )
    ),
    'strip': lambda x: x.str.strip(),
    'lstrip': lambda x, chars=None: x.str.lstrip(chars),
    'replace': lambda x, old, new: x.str.replace(old, new, regex=False),
    'concat': lambda cols, sep='': pd.concat(
        [c.fillna('').astype(str) for c in cols], axis=1
    ).agg(sep.join, axis=1),
    'add_prefix': lambda x, prefix: prefix + x.astype(str),
    'add_suffix': lambda x, suffix: x.astype(str) + suffix,
    'split_take': lambda x, sep, index=0: x.str.split(sep).str[index],
    'extract_named': lambda x, pattern: x.str.extract(pattern),
    'zfill': lambda x, width: x.str.zfill(width),
}

# Helper functions for aggregate operations


def _aggregate_cols(cols: list[pd.Series], operation: str, fill_na) -> pd.Series:
    """Aggregate multiple columns with an operation."""
    df_temp = pd.concat(cols, axis=1)
    if fill_na is not None:
        df_temp = df_temp.fillna(fill_na)
    return getattr(df_temp, operation)(axis=1)


def _any_threshold(
    cols: list[pd.Series], threshold: float, fill_na, comparison: str
) -> pd.Series:
    """Check if any column meets threshold condition."""
    df_temp = pd.concat(cols, axis=1)
    if fill_na is not None:
        df_temp = df_temp.fillna(fill_na)
    if comparison == '>':
        return (df_temp > threshold).any(axis=1).astype(int)
    elif comparison == '<':
        return (df_temp < threshold).any(axis=1).astype(int)


def _all_threshold(
    cols: list[pd.Series], threshold: float, fill_na, comparison: str
) -> pd.Series:
    """Check if all columns meet threshold condition."""
    df_temp = pd.concat(cols, axis=1)
    if fill_na is not None:
        df_temp = df_temp.fillna(fill_na)
    if comparison == '>':
        return (df_temp > threshold).all(axis=1).astype(int)
    elif comparison == '<':
        return (df_temp < threshold).all(axis=1).astype(int)


def _get_input_value(df, input_ref):
    """Get value from dataframe or return scalar."""
    if isinstance(input_ref, str) and input_ref in df.columns:
        return df[input_ref]
    else:
        # It's a scalar (int, float, str, etc.)
        return input_ref


# Main transformation engine


def apply_transformations(
    df: pd.DataFrame | gpd.GeoDataFrame,
    recipe: dict[str, Any],
    silent: bool = False,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Apply transformations from recipe to dataframe.

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        Input data to transform
    recipe : dict
        Recipe dictionary containing 'transformations' and optionally
        'transformation_patterns' keys
    silent : bool, default False
        If True, suppress warnings

    Returns
    -------
    DataFrame or GeoDataFrame
        Transformed dataframe with new columns added
    """
    df = df.copy()

    # Check for duplicate columns
    if df.columns.duplicated().any():
        if not silent:
            warnings.warn(
                'Duplicate column names in dataframe will be deleted: '
                + ', '.join(df.columns[df.columns.duplicated()].unique())
            )
        df = df.loc[:, ~df.columns.duplicated()]

    # Apply individual transformations
    if 'transformations' in recipe:
        for transform_config in recipe['transformations']:
            df = apply_transformation(df, transform_config, silent)

    # Apply pattern-based transformations
    if 'transformation_patterns' in recipe:
        for pattern_config in recipe['transformation_patterns']:
            df = apply_transformation_pattern(df, pattern_config, silent)

    return df


def apply_legacy_columns(
    df: pd.DataFrame | gpd.GeoDataFrame,
    recipe: dict[str, Any],
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Rename legacy columns declared by a recipe, merging when both exist."""
    renames = recipe.get('legacy_columns') or {}
    if not renames:
        return df

    df = df.copy()
    for old, new in renames.items():
        if old not in df.columns:
            continue
        if new in df.columns:
            df[new] = df[new].where(df[new].notna(), df[old])
            df = df.drop(columns=old)
        else:
            df = df.rename(columns={old: new})
    return df


def apply_transformation(
    df: pd.DataFrame | gpd.GeoDataFrame,
    config: dict[str, Any],
    silent: bool = False,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Apply a single transformation based on configuration."""
    transform_type = config['type']
    output_col = config['output']

    # Check if output column already exists
    if output_col in df.columns and not silent:
        warnings.warn(f"Column '{output_col}' already exists and will be overwritten")

    # A recipe is written for the full source schema; a particular file (or a
    # focused test frame) may legitimately lack some of those columns. Skip a
    # transformation whose declared input column(s) are entirely absent rather
    # than failing the whole run — mirroring apply_legacy_columns and the
    # aggregate handler, which already tolerate missing inputs.
    input_cols = config.get('inputs')
    if input_cols is None and 'input' in config:
        input_cols = [config['input']]
    if input_cols and all(col not in df.columns for col in input_cols):
        if not silent:
            warnings.warn(
                f"Skipping transformation to '{output_col}': none of its input "
                f'columns {list(input_cols)} are present.'
            )
        return df

    try:
        if transform_type == 'unary':
            df[output_col] = _apply_unary(df, config)

        elif transform_type == 'binary':
            df[output_col] = _apply_binary(df, config)

        elif transform_type == 'aggregate':
            df[output_col] = _apply_aggregate(df, config)

        elif transform_type == 'conditional':
            df[output_col] = _apply_conditional(df, config)

        elif transform_type == 'datetime':
            df[output_col] = _apply_datetime(df, config)

        elif transform_type == 'string':
            df[output_col] = _apply_string(df, config)

        elif transform_type == 'expression':
            df[output_col] = _apply_expression(df, config)

        elif transform_type == 'remap':
            df[output_col] = _apply_remap(
                df[config['input']], config['mapping'], config.get('default')
            )

        elif transform_type == 'remap_pattern':
            df[output_col] = _apply_remap_pattern(
                df[config['input']], config['patterns'], config.get('default')
            )

        elif transform_type == 'remap_file':
            if 'crosswalk_id' in config:
                # A recipe-relative crosswalk asset (e.g. a '*-remap.csv'
                # beside the recipe), resolved by recipe id like the
                # harmonizer's remap_id.
                df[output_col] = df[config['input']].map(
                    get_crosswalk({'recipe_id': config['crosswalk_id']})
                )
            else:
                df[output_col] = _apply_remap_file(
                    df[config['input']],
                    config['crosswalk_file'],
                    config.get('key_col', 0),
                    config.get('value_col', 1),
                )

        elif transform_type == 'remap_conditional':
            df[output_col] = _apply_remap_conditional(
                df[config['input']], config['conditions'], config.get('default')
            )

        else:
            raise ValueError(f'Unknown transformation type: {transform_type}')

    except Exception as e:
        raise RuntimeError(f"Error applying transformation '{config}': {e}") from e

    return df


def _apply_unary(
    df: pd.DataFrame | gpd.GeoDataFrame, config: dict[str, Any]
) -> pd.Series:
    """Apply unary operation to single column."""
    operation = config['operation']
    input_col = config['input']

    if operation not in UNARY_OPS:
        raise ValueError(f'Unknown unary operation: {operation}')

    input_series = df[input_col]

    # Handle special cases with arguments
    if 'args' in config:
        return UNARY_OPS[operation](input_series, **config['args'])
    else:
        return UNARY_OPS[operation](input_series)


def _apply_binary(
    df: pd.DataFrame | gpd.GeoDataFrame, config: dict[str, Any]
) -> pd.Series:
    """Apply binary operation to two columns."""
    operation = config['operation']
    inputs = config['inputs']

    if len(inputs) != 2:
        raise ValueError(
            f'Binary operation requires exactly 2 inputs, got {len(inputs)}'
        )

    if operation not in BINARY_OPS:
        raise ValueError(f'Unknown binary operation: {operation}')

    return BINARY_OPS[operation](df[inputs[0]], df[inputs[1]])


def _apply_aggregate(
    df: pd.DataFrame | gpd.GeoDataFrame, config: dict[str, Any]
) -> pd.Series:
    """Apply aggregate operation across multiple columns."""
    operation = config['operation']
    inputs = config['inputs']

    if operation not in AGGREGATE_OPS:
        raise ValueError(f'Unknown aggregate operation: {operation}')

    # Filter to only existing columns
    existing_inputs = [col for col in inputs if col in df.columns]
    if not existing_inputs:
        raise ValueError(f'None of the input columns exist: {inputs}')

    cols = [df[col] for col in existing_inputs]

    # Extract additional arguments
    args = config.get('args', {})
    fill_na = config.get('fill_na', args.get('fill_na'))

    if fill_na is not None:
        if operation in ['any_gt', 'any_lt', 'all_gt']:
            return AGGREGATE_OPS[operation](
                cols, threshold=args.get('threshold', 0), fill_na=fill_na
            )
        else:
            return AGGREGATE_OPS[operation](cols, fill_na=fill_na)
    else:
        if operation in ['any_gt', 'any_lt', 'all_gt']:
            return AGGREGATE_OPS[operation](
                cols, threshold=args.get('threshold', 0), fill_na=args.get('fill_na', 0)
            )
        else:
            return AGGREGATE_OPS[operation](cols)


def _apply_conditional(
    df: pd.DataFrame | gpd.GeoDataFrame, config: dict[str, Any]
) -> pd.Series:
    """Apply conditional operation to create binary indicator."""
    operation = config['operation']
    input_col = config['input']
    threshold = config['args']['threshold']

    if operation not in CONDITIONAL_OPS:
        raise ValueError(f'Unknown conditional operation: {operation}')

    return CONDITIONAL_OPS[operation](df[input_col], threshold)


def _apply_datetime(
    df: pd.DataFrame | gpd.GeoDataFrame, config: dict[str, Any]
) -> pd.Series:
    """Apply datetime extraction operation."""
    operation = config['operation']
    input_col = config['input']

    if operation not in DATETIME_OPS:
        raise ValueError(f'Unknown datetime operation: {operation}')

    # Ensure column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[input_col]):
        warnings.warn(
            f"Column '{input_col}' is not datetime type, attempting conversion"
        )
        input_series = pd.to_datetime(df[input_col])
    else:
        input_series = df[input_col]

    return DATETIME_OPS[operation](input_series)


def _apply_string(
    df: pd.DataFrame | gpd.GeoDataFrame, config: dict[str, Any]
) -> pd.Series:
    """Apply string operation to column(s)."""
    operation = config['operation']

    if operation not in STRING_OPS:
        raise ValueError(f'Unknown string operation: {operation}')

    # Handle multi-column operations like concat
    if operation == 'concat':
        inputs = config['inputs']
        # Check if all input columns exist
        missing_cols = [col for col in inputs if col not in df.columns]
        if missing_cols:
            raise ValueError(f'Missing columns for concat: {missing_cols}')

        cols = [df[col] for col in inputs]
        sep = config.get('args', {}).get('sep', '')
        return STRING_OPS[operation](cols, sep=sep)

    # Single column operations
    input_col = config['input']

    if input_col not in df.columns:
        raise ValueError(f"Input column '{input_col}' not found in dataframe")

    input_series = df[input_col]

    # Ensure column is string type
    if not pd.api.types.is_string_dtype(input_series):
        warnings.warn(f"Column '{input_col}' is not string type, attempting conversion")
        input_series = input_series.astype(str)

    # Get additional arguments if any
    args = config.get('args', {})

    # Apply operation based on required arguments
    if operation == 'substring':
        start = args.get('start', 0)
        end = args.get('end', None)
        return STRING_OPS[operation](input_series, start, end)
    elif operation == 'replace':
        old = args['old']
        new = args['new']
        return STRING_OPS[operation](input_series, old, new)
    elif operation == 'add_prefix':
        prefix = args['prefix']
        return STRING_OPS[operation](input_series, prefix)
    elif operation == 'add_suffix':
        suffix = args['suffix']
        return STRING_OPS[operation](input_series, suffix)
    elif operation == 'split_take':
        sep = args['sep']
        index = args.get('index', 0)
        return STRING_OPS[operation](input_series, sep, index)
    elif operation == 'zfill':
        width = args['width']
        return STRING_OPS[operation](input_series, width)
    elif operation == 'lstrip':
        return STRING_OPS[operation](input_series, args.get('chars'))
    elif operation == 'extract_named':
        pattern = args['pattern']
        result = STRING_OPS[operation](input_series, pattern)
        # Named groups become new columns; merge them into df is the caller's
        # responsibility — but single-group patterns still return a Series.
        if result.shape[1] == 1:
            return result.iloc[:, 0]
        raise NotImplementedError(
            '`openplaces.io.transform.str.extract_named` returned multiple groups:\n\n'
            + str(result)
        )
    else:
        # Operations with no arguments (upper, lower, strip)
        return STRING_OPS[operation](input_series)


def _apply_expression(
    df: pd.DataFrame | gpd.GeoDataFrame, config: dict[str, Any]
) -> pd.Series:
    """Apply complex expression using eval or formatting."""
    expression = config['expression']
    inputs = config.get('inputs', [])

    # Check if all input columns exist
    missing_cols = [col for col in inputs if col not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing columns for expression: {missing_cols}')

    # Try to use eval if expression doesn't contain formatting placeholders
    if '{' not in expression:
        try:
            return df.eval(expression)
        except Exception as e:
            raise RuntimeError(f"Error evaluating expression '{expression}': {e}")

    # Otherwise, format the expression with column names and eval
    try:
        # Replace {column} placeholders with actual column references
        formatted_expr = expression
        for col in inputs:
            formatted_expr = formatted_expr.replace(f'{{{col}}}', f'`{col}`')
        return df.eval(formatted_expr)
    except Exception as e:
        raise RuntimeError(f"Error evaluating formatted expression '{expression}': {e}")


def apply_transformation_pattern(
    df: pd.DataFrame | gpd.GeoDataFrame,
    config: dict[str, Any],
    silent: bool = False,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Apply pattern-based transformation to multiple columns."""
    pattern = config['pattern']
    transform_type = config['type']
    apply_to_columns = config.get('apply_to_columns', [])

    # Filter to only existing columns
    existing_cols = [col for col in apply_to_columns if col in df.columns]

    if not existing_cols and not silent:
        warnings.warn(f"None of the columns for pattern '{pattern}' exist in dataframe")

    # Apply transformation to each column
    for col in existing_cols:
        # Generate output column name from pattern
        output_col = pattern.replace('{column}', col).replace('{input}', col)

        # Create individual transformation config
        individual_config = {
            'output': output_col,
            'type': transform_type,
        }

        # Copy relevant fields from pattern config
        if transform_type == 'unary':
            individual_config['input'] = col
            individual_config['operation'] = config['operation']
            if 'args' in config:
                individual_config['args'] = config['args']

        else:
            raise ValueError(
                "Pattern-based transformation only supports 'unary' type, "
                f"got '{transform_type}'"
            )

        df = apply_transformation(df, individual_config, silent)

    return df


def _apply_remap(
    series: pd.Series, mapping: dict[str, Any], default: Any = None
) -> pd.Series:
    """Apply simple dictionary mapping to series."""
    result = series.map(mapping)
    if default is not None:
        result = result.fillna(default)
    return result


def _apply_remap_pattern(
    series: pd.Series, patterns: list[dict], default: Any = None
) -> pd.Series:
    """Apply regex pattern-based mapping."""
    result = pd.Series(default, index=series.index)
    for pattern_dict in patterns:
        pattern = pattern_dict['pattern']
        value = pattern_dict['value']
        mask = series.str.match(pattern, na=False)
        result.loc[mask] = value
    return result


def _apply_remap_file(
    series: pd.Series, crosswalk_file: str, key_col: int = 0, value_col: int = 1
) -> pd.Series:
    """Load and apply external crosswalk file."""
    if not os.path.exists(crosswalk_file):
        raise FileNotFoundError(f'Crosswalk file not found: {crosswalk_file}')

    crosswalk = pd.read_csv(crosswalk_file)
    mapping = dict(zip(crosswalk.iloc[:, key_col], crosswalk.iloc[:, value_col]))
    return series.map(mapping)


def _apply_remap_conditional(
    series: pd.Series, conditions: list[dict], default: Any = None
) -> pd.Series:
    """Apply conditional logic for remapping."""
    result = pd.Series(default, index=series.index)

    for cond in conditions:
        condition_type = cond['condition']
        output_value = cond['output']

        if condition_type == 'in':
            mask = series.isin(cond['values'])
        elif condition_type == 'startswith':
            mask = series.str.startswith(cond['value'], na=False)
        elif condition_type == 'endswith':
            mask = series.str.endswith(cond['value'], na=False)
        elif condition_type == 'contains':
            mask = series.str.contains(cond['value'], na=False, regex=False)
        elif condition_type == 'regex':
            mask = series.str.contains(cond['pattern'], na=False, regex=True)
        else:
            raise ValueError(f'Unknown condition type: {condition_type}')

        result.loc[mask] = output_value

    return result


def get_crosswalk(crosswalk_dict, flip=False):
    """Get a crosswalk (Series of default keys -> source keys)

    Parameters
    ----------
    crosswalk_dict : dict
        Dictionary with crosswalk arguments
    flip : bool
        Flips keys (index) and value column (usually for joining)
    """
    if not isinstance(crosswalk_dict, dict):
        raise ValueError('crosswalk_dict must be dict.')

    if 'recipe_id' in crosswalk_dict:
        crosswalk_table = get_recipe_by_id(
            crosswalk_dict['recipe_id'],
            dtype=crosswalk_dict['dtype'] if 'dtype' in crosswalk_dict else None,
        )
        # Create a pd.Series from the first two columns:
        crosswalk_series = crosswalk_table.set_index(crosswalk_table.columns[0])[
            crosswalk_table.columns[1]
        ]
    elif 'admin_level' in crosswalk_dict:
        admin_id_crosswalk = get_admin(
            crosswalk_dict['admin_id'],
            crosswalk_dict['admin_level'],
            columns=[crosswalk_dict['admin_id_column']],
            recipe=crosswalk_dict.get('admin_recipe_id'),
        )

        crosswalk_series = admin_id_crosswalk[crosswalk_dict['admin_id_column']]
    else:
        raise ValueError(f'Crosswalk dictionary not interpretable:\n\n{crosswalk_dict}')

    if flip:
        crosswalk_series = (
            crosswalk_series[crosswalk_series.notnull()]
            .reset_index()
            .set_index(crosswalk_series.name)
        )

    mask_index_duplicates = crosswalk_series.index.duplicated(keep=False)
    if mask_index_duplicates.any():
        raise ValueError(
            'Crosswalk returned duplicated indices:\n\n'
            + str(crosswalk_series[mask_index_duplicates])
        )

    return crosswalk_series


def remap(df, recipe_id):
    """Remap values in dataframe column using recipe table

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        Data
    recipe_id : str
        ID of recipe table that contains the remapping
    """
    crosswalk = get_crosswalk({'recipe_id': recipe_id})
    shared_columns = set([crosswalk.name]) & set(df)
    if shared_columns:
        print(
            ('Column ' if len(shared_columns) == 1 else 'Columns ')
            + ', '.join([f'`{c}`' for c in shared_columns])
            + f' overwritten by `{recipe_id}`.',
        )
    return df.drop(columns=shared_columns).join(crosswalk, on=crosswalk.index.names)


def add_unique_suffix(s):
    """Make string Series unique by appending unique integer suffices.

    All duplicate occurrences are suffixed (``-1``, ``-2``, …), including the
    first one.  Use `make_index_unique` when operating on a DataFrame index and
    the first (or largest) occurrence should keep the unsuffixed value.

    Parameters
    ----------
    s : pd.Series
        String Series containing duplicate entries
    """
    # Avoid warnings about setting slices
    s = s.copy()
    duplicates = s.duplicated(keep=False)
    # Handle collisions with suffix
    counts = s[duplicates].groupby(s[duplicates], sort=False).cumcount() + 1
    s.loc[duplicates] = s.loc[duplicates].astype(str) + '-' + counts.astype(str)
    return s


def rename_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Rename `df`'s index to `name`, leaving values untouched.

    Usable as a recipe's `create_index.function`
    (`openplaces.io.transform.rename_index`) to opt a parcel entity out of
    `TableIngester`'s automatic geometry-hash `geo_id` indexing while
    keeping whatever index the source data already carries -- e.g. to
    preserve a shared join key across several tables meant to be merged
    later (see `io.aggregate.join_partitions_by_index`).

    Parameters
    ----------
    df : pd.DataFrame
    name : str
        New name for `df.index`.
    """
    return df.rename_axis(name)


def index_by(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Index `df` by `name`, whether it's currently the index or a plain column.

    Usable as a recipe's `create_index.function`
    (`openplaces.io.transform.index_by`) for a shared join key that a
    multi-table source doesn't expose consistently -- e.g. one table
    already indexed by it (nothing to do) and another carrying it as an
    ordinary column (promoted via `set_index`) -- so a single
    `create_index` config works across all of a recipe's
    `download_by: {partition: table}` tables regardless of which shape a
    given one arrives in.

    Parameters
    ----------
    df : pd.DataFrame
    name : str
        Column or existing index name to make `df`'s index.

    Raises
    ------
    ValueError
        If `name` is neither `df`'s current index name nor one of its
        columns.
    """
    if df.index.name == name:
        return df
    if name in df.columns:
        return df.set_index(name)
    raise ValueError(f'{name!r} is neither the current index nor a column of df.')


def make_index_unique(
    df: pd.DataFrame,
    sort_by: str | None = None,
    ascending: bool = False,
    separator: str = '-',
    *,
    sort_duplicates_by_area: bool = False,
    area_crs: str = 'EPSG:6933',
) -> pd.DataFrame:
    """Return a copy of a DataFrame / GeoDataFrame with a unique string index.

    Duplicate index values are resolved so that the first occurrence keeps the
    original index value and later duplicates receive suffixes ``-1``, ``-2``,
    …  Sorting controls which occurrence counts as "first".

    Unlike `add_unique_suffix`, which operates on a Series and suffixes every
    duplicate (including the first), this function preserves the unsuffixed
    value for the winning row.

    Parameters
    ----------
    df : pd.DataFrame or gpd.GeoDataFrame
        Input frame whose index will be made unique.
    sort_by : str, optional
        Column to sort the entire frame by before resolving duplicates.
    ascending : bool
        Sort direction. Default ``False`` so larger values sort first.
    separator : str
        String inserted between the original index value and the counter.
    sort_duplicates_by_area : bool
        If True, and ``df`` is a GeoDataFrame, compute equal-area geometry
        area for rows with duplicated index values and sort within each group
        so the largest polygon keeps the unsuffixed index.
    area_crs : str
        Equal-area CRS used for area calculation. Default: ``EPSG:6933``.
    """
    out = df.copy()

    if not all(isinstance(x, str) for x in out.index):
        raise TypeError('All index values must be strings.')

    if sort_by is not None:
        if sort_by not in out.columns:
            raise KeyError(f'Column not found: {sort_by!r}')
        out = out.sort_values(sort_by, ascending=ascending, kind='stable')

    elif sort_duplicates_by_area:
        if not hasattr(out, 'geometry'):
            raise TypeError(
                'sort_duplicates_by_area=True requires a GeoDataFrame '
                'with a geometry column.'
            )
        if getattr(out, 'crs', None) is None:
            raise ValueError(
                'GeoDataFrame must have a CRS to compute area '
                'in an equal-area projection.'
            )

        tmp = out.reset_index(names='__orig_index__').copy()
        tmp['__orig_order__'] = range(len(tmp))

        dup_mask = tmp['__orig_index__'].duplicated(keep=False)
        if dup_mask.any():
            dup = tmp.loc[dup_mask].copy()
            dup_gdf = dup.set_geometry(out.geometry.name, crs=out.crs)
            dup['__sort_area__'] = dup_gdf.to_crs(area_crs).geometry.area.values
            tmp['__sort_area__'] = pd.NA
            tmp.loc[dup.index, '__sort_area__'] = dup['__sort_area__'].values
            tmp = tmp.sort_values(
                by=['__orig_index__', '__sort_area__', '__orig_order__'],
                ascending=[True, ascending, True],
                kind='stable',
            )

        tmp = tmp.set_index('__orig_index__')
        tmp.index.name = out.index.name
        out = tmp.drop(columns=['__orig_order__', '__sort_area__'], errors='ignore')

    counts: dict[str, int] = {}
    new_idx: list[str] = []
    for s in out.index:
        n = counts.get(s, 0)
        new_idx.append(s if n == 0 else f'{s}{separator}{n}')
        counts[s] = n + 1

    out.index = pd.Index(new_idx, name=df.index.name)
    return out


def convert_area_unit(value, from_unit: str, to_unit: str):
    """Convert a per-unit-area value (e.g. a $/m2 rate) between area units.

    Parameters
    ----------
    value : float, numpy.ndarray, or pandas.Series
        Value(s) already expressed as an amount per `from_unit` of area.
    from_unit, to_unit : str
        Area units, keys of `core.constants.M2_PER_AREA_UNIT` (`'m2'`, `'ha'`,
        `'km2'`, `'ac'`, `'sqft'`, `'ft2'`).

    Returns
    -------
    Same type as `value`.
    """
    from openplaces.core.constants import M2_PER_AREA_UNIT

    for unit in (from_unit, to_unit):
        if unit not in M2_PER_AREA_UNIT:
            raise ValueError(
                f'Unsupported area unit {unit!r}; must be one of '
                f'{sorted(M2_PER_AREA_UNIT)}.'
            )
    if from_unit == to_unit:
        return value
    return value * M2_PER_AREA_UNIT[to_unit] / M2_PER_AREA_UNIT[from_unit]
