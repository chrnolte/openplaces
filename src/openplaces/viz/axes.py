"""Axis helpers for plots of transformed (log- or arcsinh-scaled) values."""

import numpy as np

from openplaces.utils import short_number


def add_log_ticks(ax, transform=np.arcsinh, axis='x', prefix='', sep='', **kwargs):
    """Place ticks at powers of ten on an axis plotted in transformed units.

    Use after plotting data passed through a log-like transform (e.g.
    np.arcsinh or np.log10): ticks are positioned at transform(10**n) for
    every power of ten within the current axis limits and labeled with the
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
    **kwargs
        Further keyword arguments passed to short_number.

    Returns
    -------
    matplotlib.axes.Axes
    """
    lo, hi = ax.get_xlim() if axis == 'x' else ax.get_ylim()
    ticks = []
    labels = []
    for n in range(19):
        position = transform(10**n)
        if lo <= position <= hi:
            ticks.append(position)
            labels.append(prefix + short_number(10**n, sep=sep, **kwargs))
    if axis == 'x':
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    return ax
