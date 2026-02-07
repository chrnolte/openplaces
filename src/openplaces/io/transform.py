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

import warnings
from collections.abc import Callable
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from openplaces.api import get_admin_by_level
from openplaces.recipe import get_recipe_by_id

# Operations

UNARY_OPS: dict[str, Callable] = {
    'log': np.log,
    'arcsinh': np.arcsinh,
    'arcsinh_median_centered': lambda x: np.arcsinh(x / x.median()),
    'sqrt': np.sqrt,
    'exp': np.exp,
    'abs': np.abs,
    'power': lambda x, exponent: x**exponent,
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

STRING_OPS: dict[str, Callable] = {
    'substring': lambda x, start, end=None: x.str[start:end],
    'upper': lambda x: x.str.upper(),
    'lower': lambda x: x.str.lower(),
    'strip': lambda x: x.str.strip(),
    'replace': lambda x, old, new: x.str.replace(old, new, regex=False),
    'concat': lambda cols, sep='': pd.Series(sep.join(c.astype(str) for c in cols)),
    'add_prefix': lambda x, prefix: prefix + x.astype(str),
    'add_suffix': lambda x, suffix: x.astype(str) + suffix,
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


# ============================================================================
# Main transformation engine
# ============================================================================


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
            raise ValueError(f"Unknown transformation type: {transform_type}")

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
        raise ValueError(f"Unknown unary operation: {operation}")

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
            f"Binary operation requires exactly 2 inputs, got {len(inputs)}"
        )

    if operation not in BINARY_OPS:
        raise ValueError(f"Unknown binary operation: {operation}")

    return BINARY_OPS[operation](df[inputs[0]], df[inputs[1]])


def _apply_aggregate(
    df: pd.DataFrame | gpd.GeoDataFrame, config: dict[str, Any]
) -> pd.Series:
    """Apply aggregate operation across multiple columns."""
    operation = config['operation']
    inputs = config['inputs']

    if operation not in AGGREGATE_OPS:
        raise ValueError(f"Unknown aggregate operation: {operation}")

    # Filter to only existing columns
    existing_inputs = [col for col in inputs if col in df.columns]
    if not existing_inputs:
        raise ValueError(f"None of the input columns exist: {inputs}")

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
        raise ValueError(f"Unknown conditional operation: {operation}")

    return CONDITIONAL_OPS[operation](df[input_col], threshold)


def _apply_datetime(
    df: pd.DataFrame | gpd.GeoDataFrame, config: dict[str, Any]
) -> pd.Series:
    """Apply datetime extraction operation."""
    operation = config['operation']
    input_col = config['input']

    if operation not in DATETIME_OPS:
        raise ValueError(f"Unknown datetime operation: {operation}")

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
        raise ValueError(f"Unknown string operation: {operation}")

    # Handle multi-column operations like concat
    if operation == 'concat':
        inputs = config['inputs']
        # Check if all input columns exist
        missing_cols = [col for col in inputs if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for concat: {missing_cols}")

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
        raise ValueError(f"Missing columns for expression: {missing_cols}")

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
    import os

    if not os.path.exists(crosswalk_file):
        raise FileNotFoundError(f"Crosswalk file not found: {crosswalk_file}")

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
            raise ValueError(f"Unknown condition type: {condition_type}")

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
        admin_id_crosswalk = get_admin_by_level(
            crosswalk_dict['admin_level'],
            crosswalk_dict['admin_id'],
            columns=[crosswalk_dict['admin_id_column']],
            recipe=get_recipe_by_id(crosswalk_dict['admin_recipe_id']),
        )

        crosswalk_series = admin_id_crosswalk[crosswalk_dict['admin_id_column']]
    else:
        raise ValueError(f'Crosswalk dictionary not interpretable:\n\n{crosswalk_dict}')

    if flip:
        crosswalk_series = crosswalk_series.reset_index().set_index(
            crosswalk_series.name
        )

    return crosswalk_series


def add_unique_suffix(s):
    """Make string Series unique by appending unique integer suffices

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
