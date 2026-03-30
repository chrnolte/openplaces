"""Stacked horizontal bar charts for cross-tabulated data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openplaces.viz.colors import match_palette

_NASTR = '(N/A)'


def tabulate(
    df,
    y_cat,
    x_cat,
    v='n',
    y_max_n=None,
    x_max_n=None,
    y_cat_order=None,
    x_cat_order=None,
    show_empty_category=True,
):
    """Cross-tabulate a numeric variable by two categorical columns.

    Parameters
    ----------
    df : pd.DataFrame or gpd.GeoDataFrame
    y_cat : str
        Column for y-axis categories.
    x_cat : str
        Column for x-axis categories (the stacked dimension).
    v : str
        Numeric column to aggregate; ``'n'`` counts rows.
    y_max_n : int, optional
        Keep only the top-N y categories by total weight; remainder
        is collapsed into ``'(all others)'``.
    x_max_n : int, optional
        Same for x categories.
    y_cat_order : list, optional
        Explicit ordering for y-axis values.
    x_cat_order : list, optional
        Explicit ordering for x-axis values (stack order).
    show_empty_category : bool
        If True, fill NaN labels with ``'(N/A)'`` instead of dropping.

    Returns
    -------
    pd.DataFrame
        Normalized crosstab (values sum to 1), shape (y_cats, x_cats).
    """
    cols = [x_cat, y_cat] if v == 'n' else [v, x_cat, y_cat]
    dc = df[cols].copy()

    if show_empty_category:
        for col in (x_cat, y_cat):
            if dc[col].dtype.name != 'category':
                dc[col] = dc[col].fillna(_NASTR)

    if v == 'n':
        da = dc.groupby([y_cat, x_cat])[x_cat].count().unstack().fillna(0)
    else:
        da = dc.groupby([y_cat, x_cat])[v].sum().unstack().fillna(0)

    if y_cat_order:
        da = da.loc[y_cat_order]
    if x_cat_order:
        da = da[x_cat_order]

    if y_max_n and y_max_n < len(da.index):
        top = set(da.sum(axis=1).nlargest(y_max_n).index)
        top_ordered = [i for i in da.index if i in top]
        remainder = da[~da.index.isin(top)].sum().rename('(all others)')
        da = pd.concat([da.loc[top_ordered], remainder.to_frame().T])

    if x_max_n and x_max_n < len(da.columns):
        top = set(da.sum(axis=0).nlargest(x_max_n).index)
        top_ordered = [c for c in da.columns if c in top]
        remainder = da.loc[:, ~da.columns.isin(top)].sum(axis=1).rename('(all others)')
        da = pd.concat([da[top_ordered], remainder], axis=1)

    return (da / da.sum().sum()).round(4)


def plot_tabulation(
    df,
    y_cat,
    x_cat,
    v='n',
    title=None,
    y_max_n=None,
    x_max_n=None,
    y_cat_order=None,
    x_cat_order=None,
    show_empty_category=True,
    y_lab_maxlength=30,
    x_lab_maxlength=30,
    gap_perc=0.01,
    cmap='tab20b',
    alpha=0.8,
    savefig=None,
    figsize=(7, 5),
    legend_kwds=None,
    colors=None,
    fontsize=9,
    titlesize=12,
):
    """Stacked horizontal bar chart of v ~ f(y_cat, x_cat).

    Bar heights are proportional to group totals; bar widths show the
    x_cat breakdown within each y_cat group.

    Parameters
    ----------
    df : pd.DataFrame or gpd.GeoDataFrame
    y_cat : str
        Column for y-axis groups (one bar per value).
    x_cat : str
        Column for the stacked dimension (legend entries).
    v : str
        Numeric column to aggregate; ``'n'`` counts rows.
    title : str, optional
        Plot title. Defaults to ``'% of <v> by <y_cat>'``.
    y_max_n, x_max_n : int, optional
        Cap the number of categories shown per axis.
    y_cat_order, x_cat_order : list, optional
        Explicit category orderings.
    show_empty_category : bool
        Show NaN values as ``'(N/A)'``.
    y_lab_maxlength, x_lab_maxlength : int
        Truncate labels longer than this.
    gap_perc : float
        Gap between bars as a fraction of total data weight.
    cmap : str
        Matplotlib colormap name, used when ``colors`` is None.
    alpha : float
        Bar opacity.
    savefig : str or Path, optional
        Save to this path if provided.
    figsize : tuple, optional
        Figure size ``(width, height)``.
    legend_kwds : dict, optional
        Keyword arguments passed to ``ax.legend()``. Defaults to
        ``{'loc': 'upper right', 'bbox_to_anchor': (0.985, 0.985)}``.
        Any key overrides the default; omitted keys keep their default value.
    colors : list, optional
        Explicit list of colors, one per x_cat value.
    fontsize, titlesize : int
        Font sizes for labels and title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    da = tabulate(
        df,
        y_cat,
        x_cat,
        v,
        y_max_n,
        x_max_n,
        y_cat_order,
        x_cat_order,
        show_empty_category,
    )

    dp = da.loc[da.index[::-1]]
    dp = dp[dp.sum(axis=1) > 0]
    dp.index = dp.index.astype(str)
    heights = dp.sum(axis=1)
    widths = dp.div(heights, axis=0)

    gap = dp.sum().sum() * gap_perc
    ypos = heights.add(gap).cumsum().shift(1).fillna(0) + heights / 2

    if colors is None:
        if set(widths.columns) == {True, False}:
            colors = ['#ff8a8a', '#8fd184']
        else:
            palette = match_palette(widths.columns, col_name=x_cat, weights=heights)
            if palette is not None:
                colors = [palette.get(str(c), '#cccccc') for c in widths.columns]
            else:
                colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(widths.columns)))

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        ypos,
        widths.iloc[:, 0],
        heights,
        color=colors[0],
        edgecolor='k',
        linewidth=0.5,
        alpha=alpha,
    )
    for i in range(1, len(widths.columns)):
        ax.barh(
            ypos,
            widths.iloc[:, i],
            heights,
            left=widths.cumsum(axis=1).iloc[:, i - 1],
            color=colors[i],
            edgecolor='k',
            linewidth=0.5,
            alpha=alpha,
        )

    total = heights.sum() + gap * (len(heights) - 1)
    ax.set_ylim(-total * 0.005, total * 1.01)
    ax.set_xlim(-0.005, 1.01)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

    def _crop(s, maxlen):
        s = str(s)
        return s[:maxlen] + '...' if len(s) > maxlen else s

    ax.set_yticks(ypos)
    ax.set_yticklabels([_crop(y, y_lab_maxlength) for y in dp.index], fontsize=fontsize)
    ax.set_title(
        title or f"% of {'observations' if v == 'n' else v} by '{y_cat}'",
        fontsize=titlesize,
    )

    _legend_kwds = {
        'loc': 'upper left',
        'bbox_to_anchor': (1, 1),
        'title': x_cat,
        'fontsize': fontsize,
        'title_fontsize': fontsize,
        'framealpha': 0.7,
    }
    if legend_kwds:
        _legend_kwds.update(legend_kwds)
    leg = ax.legend(
        [_crop(c, x_lab_maxlength) for c in dp.columns],
        **_legend_kwds,
    )
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor('k')

    plt.tight_layout()

    if savefig is not None:
        fig.savefig(savefig, bbox_inches='tight')

    return fig, ax
