"""
The admin layers needed to draw one unit in its context: the unit
among its neighbors, inside its region, inside its country, each
layer scoped to the ancestor it sits in.
"""

from openplaces.core.constants import (
    STRING_SEPARATOR_WITHIN_IDS,
)


def context_layers(admin_id):
    """Return the admin layers needed to draw one unit in its context.

    A reader looking at a single town wants that town among its
    neighbours, that neighbourhood inside its region, and the region
    inside its country -- not a global layer per level. Each layer is
    therefore scoped to the ancestor it sits inside, so drawing one town
    reads a handful of small files rather than several worldwide ones.

    For ``US-MA-SOM`` the layers are the outline of ``US``, the level-2
    units of ``US``, and the level-3 units of ``US-MA`` -- the last of
    which contains the requested town. A level-4 identifier adds the
    level-4 units of its level-3 parent.

    Parameters
    ----------
    admin_id : AdminId or str
        The unit to be drawn.

    Returns
    -------
    list of dict
        One entry per layer, outermost first, each with:

        ``admin_level``
            Level of the units in the layer.
        ``scope``
            Identifier of the ancestor the layer is scoped to. The layer
            holds exactly that unit's children, except the outline,
            which holds the country itself.
        ``role``
            ``'outline'`` for the country boundary, ``'focus'`` for the
            layer containing the requested unit, ``'context'`` otherwise.

    Examples
    --------
    >>> [(d['admin_level'], d['scope']) for d in context_layers('US-MA-SOM')]
    [(1, 'US'), (2, 'US'), (3, 'US-MA')]
    """
    parts = [
        part.strip()
        for part in str(admin_id).strip().split(STRING_SEPARATOR_WITHIN_IDS)
    ]
    if not all(parts):
        raise ValueError(f'Not an admin identifier: {admin_id!r}')
    depth = len(parts)
    country = parts[0]

    layers = [{'admin_level': 1, 'scope': country, 'role': 'outline'}]
    for level in range(2, depth + 1):
        # The layer at `level` holds the children of the ancestor one
        # level up, which is the identifier truncated to `level - 1`.
        scope = STRING_SEPARATOR_WITHIN_IDS.join(parts[: level - 1])
        layers.append(
            {
                'admin_level': level,
                'scope': scope,
                'role': 'focus' if level == depth else 'context',
            }
        )
    return layers
