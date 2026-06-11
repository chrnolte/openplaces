"""
openplaces utilities module

General-purpose utility functions for formatting, display, and debugging.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def pretty_print(
    obj: Any,
    indent: int = 2,
    max_depth: int = 10,
    return_string: bool = False,
    _current_depth: int = 0,
) -> str | None:
    """
    Pretty print nested dictionaries, lists, and Python class instances.

    Converts dataclasses and objects with __dict__ to dictionaries
    recursively, then formats them in YAML-style (no quotes around keys,
    indentation-based).

    By default, prints the output directly. Set return_string=True to
    get the string instead.

    Parameters
    ----------
    obj : Any
        Object to print (dict, list, dataclass, or any class instance)
    indent : int, default=2
        Number of spaces per indentation level
    max_depth : int, default=10
        Maximum nesting depth to prevent infinite recursion
    return_string : bool, default=False
        If True, return the formatted string instead of printing it
    _current_depth : int
        Internal parameter for tracking recursion depth

    Returns
    -------
    str | None
        If return_string=True, returns formatted string. Otherwise
        prints and returns None.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Person:
    ...     name: str
    ...     age: int
    >>>
    >>> data = {'people': [Person('Alice', 30), Person('Bob', 25)]}
    >>> pretty_print(data)
    >>>
    >>> # Get string without printing
    >>> s = pretty_print(data, return_string=True)
    """
    if _current_depth >= max_depth:
        result = repr(obj)
        if return_string:
            return result
        print(result)
        return None

    # Convert object to serializable form
    serialized = _to_serializable(
        obj, max_depth=max_depth, current_depth=_current_depth
    )

    # Format as YAML-style
    output = _format_yaml_style(serialized, indent=indent)

    if return_string:
        return output

    print(output)
    return None


def format_list(
    items: list,
    nmax: int = 7,
    sep: str = ', ',
):
    """Format list to short string, making sure start and end are shown

    [1, 2, 3, 4, 5, 6, 7] > '1, 2, 3, 4, 5, 6, 7'

    [1, 2, 3, 4, 5, 6, 7, 8] > '1, 2, 3, ..., 6, 7, 8'

    """
    n = len(items)
    if n <= nmax:
        parts = [str(x) for x in items]
    else:
        npos = math.floor(nmax / 2)
        parts = (
            [str(x) for x in items[:npos]] + ['...'] + [str(x) for x in items[-npos:]]
        )
    return sep.join(parts)


def inspect_table(
    data: pd.DataFrame,
    n: int = 5,
    columns: list[str] | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Print an output summary and return a transposed row sample."""
    if columns is None:
        columns = [column for column in data.columns if column != 'geometry']
    else:
        columns = [column for column in columns if column in data.columns]

    details = []
    output_paths = data.attrs.get('openplaces_output_paths', [])
    partition_ids = data.attrs.get('openplaces_partition_ids', [])
    if output_paths:
        details.append(f'{len(output_paths)} file(s)')
    if partition_ids:
        details.append(f'{len(partition_ids)} partition(s)')

    summary = f'{len(data):,} rows x {len(data.columns):,} columns'
    if details:
        summary += f' from {", ".join(details)}'
    print(summary)

    if data.empty:
        return data.loc[:, columns].T
    return (
        data.loc[:, columns]
        .sample(min(n, len(data)), random_state=random_state)
        .sort_index()
        .T
    )


def _format_yaml_style(obj: Any, indent: int = 2, _level: int = 0) -> str:
    """
    Format object as YAML-style string (no quotes on keys, indentation-based).
    """
    spaces = ' ' * (indent * _level)

    if obj is None:
        return 'null'
    elif isinstance(obj, bool):
        return 'true' if obj else 'false'
    elif isinstance(obj, int | float):
        return str(obj)
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        if not obj:
            return '{}'
        lines = []
        for key, value in obj.items():
            if isinstance(value, dict):
                if not value:
                    lines.append(f'{spaces}{key}: {{}}')
                else:
                    lines.append(f'{spaces}{key}:')
                    lines.append(_format_yaml_style(value, indent, _level + 1))
            elif isinstance(value, list):
                if not value:
                    lines.append(f'{spaces}{key}: []')
                else:
                    lines.append(f'{spaces}{key}:')
                    lines.append(_format_yaml_style(value, indent, _level + 1))
            else:
                formatted_value = _format_yaml_style(value, indent, _level + 1)
                lines.append(f'{spaces}{key}: {formatted_value}')
        return '\n'.join(lines)
    elif isinstance(obj, list):
        if not obj:
            return '[]'
        lines = []
        min_indent = spaces + ' ' * indent
        for item in obj:
            if isinstance(item, dict | list):
                item_str = _format_yaml_style(item, indent, _level + 1)
                item_lines = item_str.split('\n')
                lines.append(f'{spaces}- {item_lines[0][len(min_indent) :]}')
                for line in item_lines[1:]:
                    lines.append(f'{spaces}  {line[len(min_indent) :]}')
            else:
                formatted_item = _format_yaml_style(item, indent, _level + 1)
                lines.append(f'{spaces}- {formatted_item}')
        return '\n'.join(lines)
    else:
        return str(obj)


def _to_serializable(obj: Any, max_depth: int = 10, current_depth: int = 0) -> Any:
    """
    Recursively convert objects to JSON-serializable form.

    Handles:
    - Dataclasses (via asdict)
    - Class instances with __dict__
    - Paths (to strings)
    - Lists and tuples
    - Dictionaries
    - Sets (to lists)
    - Primitives (int, float, str, bool, None)
    """
    if current_depth >= max_depth:
        return repr(obj)

    # Handle None and primitives
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj

    # Handle Path objects
    if isinstance(obj, Path):
        return str(obj)

    # Handle dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        obj_dict = asdict(obj)
        return {
            k: _to_serializable(v, max_depth, current_depth + 1)
            for k, v in obj_dict.items()
        }

    # Handle dictionaries
    if isinstance(obj, dict):
        return {
            k: _to_serializable(v, max_depth, current_depth + 1) for k, v in obj.items()
        }

    # Handle lists and tuples
    if isinstance(obj, list | tuple):
        return [_to_serializable(item, max_depth, current_depth + 1) for item in obj]

    # Handle sets
    if isinstance(obj, set):
        return [
            _to_serializable(item, max_depth, current_depth + 1) for item in sorted(obj)
        ]

    # Handle objects with __dict__
    if hasattr(obj, '__dict__'):
        return {
            k: _to_serializable(v, max_depth, current_depth + 1)
            for k, v in obj.__dict__.items()
            if not k.startswith('_')  # Skip private attributes
        }

    # Fallback to string representation
    return str(obj)


__all__ = [
    'format_list',
    'inspect_table',
    'pretty_print',
    'short_number',
]


def short_number(x, round_to=0, suffix='', sep=' ', base=1000):
    """Convert a large number to a short string for display.

    Examples: 14124521 -> '14 M', or '14.1 M' with round_to=1.

    Parameters
    ----------
    x : numeric
        Number to format.
    round_to : int
        Number of decimal digits to display.
    suffix : str
        Suffix appended after the unit (e.g. a base unit such as 'B'
        for bytes).
    sep : str
        Separator between number and unit.
    base : int
        Scaling factor between units; use 1024 for computer storage.
    """
    units = ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    for unit in units:
        if abs(x) < base or unit == units[-1]:
            break
        x /= base
    x = round(x, round_to)
    if x == int(x):
        x = int(x)
    if unit or suffix:
        return f'{x}{sep}{unit}{suffix}'
    return str(x)


def remove_accents(x):
    """Turns latin-derived special characters into ascii alphabet."""
    if not isinstance(x, str):
        return x
    x = unicodedata.normalize('NFKD', x)
    for from_char, to_char in {'ə': 'e', 'ı': 'i', 'ħ': 'h'}.items():
        x = x.replace(from_char, to_char)
    return x.encode('ASCII', 'ignore').decode('ascii')


def standardize_names(x):
    """Standardize country name characters"""
    if not isinstance(x, str):
        return x
    x = remove_accents(x)
    check = re.compile('[A-Za-z- ]')
    x = ''.join([a for a in x if check.match(a)]).title()
    x = x.replace(' Apskritis', '').strip()  # Latvia
    return x


# Some standard conversions, currently specific to linking up US counties
CONVERSIONS = {
    'sainte': 'ste',
    'saint': 'st',
}


def create_comparable_name_link(x, replace_dict=CONVERSIONS):
    if not isinstance(x, str):
        return x
    x = remove_accents(x)
    check = re.compile('[A-Za-z- ]')
    x = ''.join([a for a in x if check.match(a)]).lower()

    if isinstance(replace_dict, dict):
        for k, v in replace_dict.items():
            x = x.replace(k, v)

    return x
