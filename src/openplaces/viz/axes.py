"""Axis helpers for plots of transformed (log- or arcsinh-scaled) values."""

import numpy as np

from openplaces.utils import short_number

# Smallest decade a tick can be placed at (10**-9).
_MIN_DECADE = -9


def add_log_ticks(
    ax, transform=np.arcsinh, axis='x', prefix='', sep='', subs=(1,), **kwargs
):
    """Place ticks at powers of ten on an axis plotted in transformed units.

    Use after plotting data passed through a log-like transform (e.g.
    np.arcsinh or np.log10): ticks are positioned at transform(m * 10**n) for
    every power of ten (and, if `subs` names more than one multiplier, each
    sub-decade multiple) within the current axis limits, labeled with the
    original value via :func:`openplaces.utils.short_number` (e.g. '$10K',
    '$1M').

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose x or y values are transformed data.
    transform : callable
        The transform that was applied to the plotted data.
    axis : str
        'x' or 'y'.
    prefix : str
        Prepended to each label (e.g. '$').
    sep : str
        Separator between number and unit, passed to short_number.
    subs : tuple of int
        Within-decade multipliers to also place ticks at, e.g. `(1, 3)` for
        ticks at 1, 3, 10, 30, 100, ... Defaults to `(1,)` (decades only).
        Multipliers apply to sub-1 decades too, so `(1, 3)` also gives
        0.3, 0.1, 0.03 and so on.
    **kwargs
        Further keyword arguments passed to short_number.

    Returns
    -------
    matplotlib.axes.Axes
        The same axes. Its ticks are left untouched when no power of ten
        falls inside the current limits.
    """
    lo, hi = ax.get_xlim() if axis == 'x' else ax.get_ylim()
    ticks = []
    labels = []
    # Starting at 10**0 left an axis of shares, rates or ratios in
    # the unit interval with no tick in range at all, which then wiped
    # matplotlib's own ticks (below). Decades below 1 are labeled in
    # plain decimal form, since short_number rounds 0.01 to '0'.
    for n in range(_MIN_DECADE, 19):
        for m in subs:
            value = m * 10**n
            position = transform(value)
            if lo <= position <= hi:
                ticks.append(position)
                if value < 1:
                    labels.append(prefix + np.format_float_positional(value, trim='-'))
                else:
                    labels.append(prefix + short_number(value, sep=sep, **kwargs))
    # Nothing in range: leave the axis as matplotlib formatted it rather
    # than replacing its ticks with none.
    if not ticks:
        return ax
    if axis == 'x':
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    return ax
