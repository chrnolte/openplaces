"""The spine as its own identifier registry.

An admin id is a name that outlives the run that minted it: it is written
into file paths, crosswalk CSVs and delivered datasets. So the generator
must be able to reproduce an existing id exactly, and must never hand an
id that once meant one unit to a different one.

Neither property comes free from the assignment in
:mod:`~openplaces.io.admin_codes.assign`. That solves each sibling group as
one global optimum, which is what makes it order-independent -- and also
what makes it unstable under growth: adding one municipality to a group can
move a code from one existing unit to another. Measured on Texas, inserting
a single name moved two of 254 counties, and moved 'NA' from Nacogdoches to
Navarro. A lookup keyed on 'NA' would then silently return the wrong county.

The fix is to treat the committed spine as the registry of record. Units it
already names keep their codes; only genuinely new units are assigned, from
what is left. Codes the spine issued to units that have since disappeared
stay reserved rather than returning to the pool, so a retired id fails to
resolve instead of resolving to the wrong place.

Identity is matched on (parent, name). A unit whose name changes is treated
as new and its former code is retired -- conservative on purpose, since the
alternative is guessing that two differently-named rows are the same place.
"""

from functools import cache

import pandas as pd

# `spine_path` lives in `path` (Layer 1) so that `geo` can read
# the spine without importing from `io`. Imported here because
# this is where callers have always got it from.
from openplaces.path import spine_path


def level_of(admin_id_column: str) -> int | None:
    """Return the admin level a column name refers to, e.g. 'admin4_id' -> 4.

    Returns None when the column does not name a level, which is how a
    caller using some other identifier column opts out of pinning.
    """
    digits = ''.join(c for c in admin_id_column if c.isdigit())
    return int(digits) if digits else None


@cache
def load_registry(level: int, id_separator: str = '-'):
    """Return the codes the spine has already issued at one level.

    Parameters
    ----------
    level : int
        Admin level, 2 to 4.
    id_separator : str, optional
        Separator between the parent id and the code.

    Returns
    -------
    tuple of (dict, dict, dict)
        ``pins`` maps (parent id, name) to the code that unit already
        holds. ``by_external`` maps (parent id, source code) the same way,
        for the units a name cannot identify: those with no name at all,
        and those sharing a name with a sibling. ``issued`` maps a parent
        id to every code the spine records under it, including those
        whose unit may since have gone.

    Notes
    -----
    Cached: the spine is committed reference data and does not change
    within a run. Returns empty mappings when no spine exists for the
    level, so a first-time bootstrap generates freely.
    """
    path = spine_path(level)
    if not path.exists():
        return {}, {}, {}
    column = f'admin{level}_id'
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8')
    parents = frame[column].str.rsplit(id_separator, n=1).str[0]
    codes = frame[column].str.rsplit(id_separator, n=1).str[1]
    externals = external_codes(frame, level)

    aliases = (
        frame['name_alternatives']
        if 'name_alternatives' in frame
        else pd.Series([''] * len(frame), index=frame.index)
    )

    pins: dict[tuple[str, str], str | None] = {}
    by_external: dict[tuple[str, str], str | None] = {}
    issued: dict[str, set[str]] = {}
    rows = list(zip(parents, codes, frame['name'], externals, aliases))

    for parent, code, name, external, _ in rows:
        issued.setdefault(parent, set()).add(code)
        name = str(name).strip()
        if name:
            key = (parent, name)
            # A name duplicated under one parent identifies nothing, so
            # neither row is pinned by it; the source code below is what
            # tells them apart.
            pins[key] = None if key in pins else code
        if external:
            key = (parent, external)
            by_external[key] = None if key in by_external else code

    # Second pass, so a real unit's name always outranks somebody else's
    # alternative spelling of it. Recorded spellings pin too, which is
    # what keeps a source that switches romanisation -- Esfahan to
    # Isfahan, Kordestan to Kurdistan -- from minting a new unit for one
    # that already exists.
    for parent, code, name, _, alias in rows:
        for spelling in str(alias).split('|'):
            spelling = spelling.strip()
            if spelling and spelling != str(name).strip():
                pins.setdefault((parent, spelling), code)

    return (
        {k: v for k, v in pins.items() if v is not None},
        {k: v for k, v in by_external.items() if v is not None},
        issued,
    )


def external_codes(frame, level: int) -> pd.Series:
    """Return each row's source-assigned code, or '' where it has none.

    The source's own code -- a national statistical code where the frame
    carries one, otherwise the GADM id -- identifies a unit that its name
    cannot: one with no name, or one sharing a name with a sibling. It is
    a fallback for pinning only, never a component of the id itself.
    """
    blank = pd.Series([''] * len(frame), index=frame.index, dtype=object)
    result = blank
    for column in (f'admin{level}_id_admin1', f'admin{level}_id_gadm'):
        if column in frame:
            candidate = frame[column].fillna('').astype(str).str.strip()
            result = result.where(result != '', candidate)
    return result
