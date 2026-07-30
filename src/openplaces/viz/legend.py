"""Shared legend rendering for the large-scale entity map tracks.

:mod:`openplaces.viz.interactive` (Lonboard/deck.gl) and
:mod:`openplaces.viz.raster` (Datashader) both color entities by the same
``color_by`` convention -- a categorical palette match via
:func:`openplaces.viz.colors.match_palette`, or a numeric min-max ramp --
so the legend-content logic lives here once. Each track composites the
result into its own native output: an HTML widget for Lonboard's
ipywidgets-based map, a corner box drawn directly onto the Datashader
PIL image.
"""

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from openplaces.viz.colors import MISSING_LABEL, RESERVED_NEUTRAL_COLOR

MAX_CATEGORICAL_ENTRIES = 8


def categorical_legend_entries(
    values: pd.Series,
    palette: dict,
    max_entries: int = MAX_CATEGORICAL_ENTRIES,
) -> list[tuple[str, str]]:
    """Build ordered ``(label, hex_color)`` legend entries for a categorical column.

    Parameters
    ----------
    values : pandas.Series
        Category values actually present in the plotted data (post-filter).
    palette : dict
        ``{label: hex_color}`` palette covering every distinct label in
        `values`, e.g. from `openplaces.viz.colors.resolve_category_colors`
        (which guarantees full coverage -- no per-call fallback needed here).
    max_entries : int
        Cap on the number of distinct categories shown; the least frequent
        categories beyond the cap are folded into a single ``'Other'`` entry
        so the legend doesn't grow unbounded for high-cardinality columns.

    Returns
    -------
    list of (str, str)
        ``(label, hex_color)`` pairs, most frequent category first.
    """
    counts = values.astype('object').fillna(MISSING_LABEL).value_counts()
    entries = [
        (str(label), palette.get(str(label), RESERVED_NEUTRAL_COLOR))
        for label in counts.index
    ]
    if len(entries) <= max_entries:
        return entries
    return [*entries[: max_entries - 1], ('Other', RESERVED_NEUTRAL_COLOR)]


def legend_html(
    title: str,
    entries: list[tuple[str, str]] | None = None,
    numeric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    low_color: str | None = None,
    high_color: str | None = None,
) -> str:
    """Render a compact, self-contained HTML legend.

    Meant for `IPython.display.display` alongside a Lonboard map (a
    deck.gl/WebGL canvas can't host DOM overlays the way a raster image
    can be drawn on directly, so the legend is a separate displayed widget
    rather than composited into the map itself).

    Parameters
    ----------
    title : str
        Legend heading -- the `color_by` column name.
    entries : list of (str, str), optional
        ``(label, hex_color)`` pairs for a categorical legend, e.g. from
        `categorical_legend_entries`. Ignored if `numeric` is True.
    numeric : bool
        If True, render a horizontal gradient bar labeled `vmin`/`vmax`
        instead of discrete swatches.
    vmin, vmax : float, optional
        Data range labels for a numeric legend.
    low_color, high_color : str, optional
        Hex endpoints of the numeric color ramp.

    Returns
    -------
    str
        HTML markup, ready to wrap in `IPython.display.HTML`.
    """
    if numeric:
        body = (
            f'<div style="width:130px;height:10px;border-radius:2px;'
            f'background:linear-gradient(to right, {low_color}, {high_color});'
            f'margin-bottom:3px;"></div>'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:10px;color:#333;width:130px;">'
            f'<span>{vmin:,.0f}</span><span>{vmax:,.0f}</span></div>'
        )
    else:
        rows = []
        for label, color in entries:
            rows.append(
                '<div style="display:flex;align-items:center;margin:1px 0;">'
                f'<span style="display:inline-block;width:9px;height:9px;'
                f'border-radius:50%;margin-right:6px;background:{color};'
                f'flex-shrink:0;"></span>'
                f'<span style="font-size:11px;color:#222;">{label}</span></div>'
            )
        body = ''.join(rows)

    return (
        '<div style="font-family:sans-serif;padding:6px 10px;'
        'border:1px solid #ccc;border-radius:4px;display:inline-block;'
        'background:white;">'
        f'<div style="font-size:11px;font-weight:600;margin-bottom:4px;'
        f'color:#111;">{title}</div>'
        f'{body}</div>'
    )


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return (r, g, b, alpha)


def _gradient_strip(
    width: int, height: int, low_color: str, high_color: str
) -> Image.Image:
    low = _hex_to_rgba(low_color)
    high = _hex_to_rgba(high_color)
    strip = Image.new('RGBA', (width, height))
    pixels = strip.load()
    for x in range(width):
        t = x / max(width - 1, 1)
        color = tuple(round(low[i] + t * (high[i] - low[i])) for i in range(4))
        for y in range(height):
            pixels[x, y] = color
    return strip


def _anchor(
    image_size: tuple[int, int], box_size: tuple[int, int], position: str, margin: int
) -> tuple[int, int]:
    img_w, img_h = image_size
    box_w, box_h = box_size
    x = margin if 'left' in position else img_w - box_w - margin
    y = margin if 'upper' in position else img_h - box_h - margin
    return x, y


def draw_legend(
    img: Image.Image,
    title: str,
    entries: list[tuple[str, str]] | None = None,
    numeric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    low_color: str | None = None,
    high_color: str | None = None,
    position: str = 'lower right',
    margin: int = 12,
) -> Image.Image:
    """Composite a legend box directly onto a Datashader raster image.

    Datashader output is a flat raster with no DOM/widget to overlay
    (unlike Lonboard's `legend_html`, displayed as a separate widget), so
    the legend is drawn as pixels into a copy of the image itself.

    Parameters
    ----------
    img : PIL.Image.Image
        Rendered raster to draw onto.
    title : str
        Legend heading -- the `color_by` column name.
    entries : list of (str, str), optional
        ``(label, hex_color)`` pairs for a categorical legend, e.g. from
        `categorical_legend_entries`. Ignored if `numeric` is True.
    numeric : bool
        If True, draw a horizontal gradient bar labeled `vmin`/`vmax`
        instead of discrete swatches.
    vmin, vmax : float, optional
        Data range labels for a numeric legend.
    low_color, high_color : str, optional
        Hex endpoints of the numeric color ramp.
    position : {'lower right', 'lower left', 'upper right', 'upper left'}
        Corner of the image to anchor the legend box.
    margin : int
        Pixel margin between the legend box and the image edge.

    Returns
    -------
    PIL.Image.Image
        A new RGBA image with the legend composited on top.
    """
    if not numeric and not entries:
        return img

    font_title = ImageFont.load_default(size=13)
    font_label = ImageFont.load_default(size=11)

    pad = 10
    row_h = 16

    if numeric:
        bar_w, bar_h = 130, 10
        content_w = bar_w
        content_h = bar_h + 14
    else:
        content_h = row_h * len(entries)
        content_w = max(font_label.getlength(label) for label, _ in entries) + 20

    title_w = font_title.getlength(title)
    box_w = int(max(content_w, title_w) + pad * 2)
    box_h = int(row_h + content_h + pad * 2)

    legend = Image.new('RGBA', (box_w, box_h), (255, 255, 255, 235))
    draw = ImageDraw.Draw(legend)
    draw.rectangle([0, 0, box_w - 1, box_h - 1], outline=(120, 120, 120, 255))
    draw.text((pad, pad - 2), title, font=font_title, fill=(20, 20, 20, 255))

    y = pad + row_h
    if numeric:
        strip = _gradient_strip(bar_w, bar_h, low_color, high_color)
        legend.paste(strip, (pad, y), strip)
        label_min, label_max = f'{vmin:,.0f}', f'{vmax:,.0f}'
        draw.text(
            (pad, y + bar_h + 2), label_min, font=font_label, fill=(60, 60, 60, 255)
        )
        max_w = font_label.getlength(label_max)
        draw.text(
            (pad + bar_w - max_w, y + bar_h + 2),
            label_max,
            font=font_label,
            fill=(60, 60, 60, 255),
        )
    else:
        swatch_r = 4
        for label, hex_color in entries:
            cy = y + row_h // 2
            draw.ellipse(
                [pad, cy - swatch_r, pad + 2 * swatch_r, cy + swatch_r],
                fill=_hex_to_rgba(hex_color),
            )
            draw.text(
                (pad + 2 * swatch_r + 6, y + 2),
                label,
                font=font_label,
                fill=(20, 20, 20, 255),
            )
            y += row_h

    base = img.convert('RGBA')
    x0, y0 = _anchor(base.size, legend.size, position, margin)
    composited = base.copy()
    composited.alpha_composite(legend, (x0, y0))
    return composited
