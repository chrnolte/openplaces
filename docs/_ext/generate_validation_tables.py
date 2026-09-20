"""Regenerate the validation tables and heatmaps shown in the docs annexes.

Standalone script, not a live Sphinx extension, for the same reason as
``generate_coverage_maps``: the docs build has neither openplaces nor the
data root, so the fragments are rendered locally from each delivery
region's ``accuracies/`` folder and committed. The orchestrated
``validate`` job runs it after its notebooks; by hand::

    conda activate openplaces
    python docs/_ext/generate_validation_tables.py

For every ``{name}_confusion.csv`` (with its ``_accuracy.csv`` and
``_confusion.json`` sidecar, see
``openplaces.io.curator.validation.write_confusion_report``) it writes:

- a CSV snapshot of the three inputs under ``_generated/data/``, so a
  fragment can be regenerated, and checked, without the data root;
- a reST fragment under ``docs/3_examples/curate/annexes/_generated/``,
  opening with a do-not-edit comment, holding the pooled matrix of the
  primary source as a ``list-table`` (producer's accuracy column,
  consumer's accuracy row, overall accuracy and kappa in the corner),
  a per-class table for every source, and a per-stratum table;
- a PNG heatmap of that matrix under ``docs/_static/images/validation/``.

Rendering reads only the snapshot, and dates the fragment by the
validation run, not by the render, so regenerating from an unchanged
snapshot reproduces the committed fragment byte for byte.

Only aggregates are written, as in the ``accuracies/`` folder itself:
the writer has already refused every stratum below its minimum size.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNEX_DIR = REPO_ROOT / 'docs' / '3_examples' / 'curate' / 'annexes'
GENERATED_DIR = ANNEX_DIR / '_generated'
SNAPSHOT_DIR = GENERATED_DIR / 'data'
IMAGE_DIR = REPO_ROOT / 'docs' / '_static' / 'images' / 'validation'

#: Curate recipes whose delivery regions are published with validation
#: tables. Each region's accuracies/ folder is read if it exists.
RECIPES = ('US_footprint-openplaces-2026',)

#: Source whose matrix a fragment leads with, first match wins.
PRIMARY_SOURCES = ('final_vote', 'vote', 'inventory')

#: Single hue, light to dark: the heatmap encodes magnitude only.
HEATMAP_CMAP = 'Purples'
INK = '#2A2A32'

GENERATOR = 'docs/_ext/generate_validation_tables.py'


def _fmt_count(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '-'
    return f'{int(value):,}'


def _fmt_share(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '-'
    return f'{float(value):.3f}'


def _fmt_years(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '-'
    return f'{float(value):.1f}'


def list_table(header: list[str], rows: list[list[str]], title: str = '') -> list[str]:
    """Render one reST list-table.

    Parameters
    ----------
    header : list of str
        Header cells.
    rows : list of list of str
        Body cells, already formatted.
    title : str, optional
        Table caption.

    Returns
    -------
    list of str
        Lines, without a trailing blank line.
    """
    lines = [f'.. list-table:: {title}'.rstrip(), '   :header-rows: 1', '']
    for row in [header, *rows]:
        cells = [str(cell) if str(cell) != '' else ' ' for cell in row]
        lines.append(f'   * - {cells[0]}')
        lines += [f'     - {cell}' for cell in cells[1:]]
    return lines


def _primary_source(sources: list[str]) -> str:
    for source in PRIMARY_SOURCES:
        if source in sources:
            return source
    return sources[0]


def _pooled(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    return frame[
        frame['source'].eq(source)
        & frame['stratum_by'].eq('all')
        & frame['stratum'].eq('all')
    ]


def _wide(confusion: pd.DataFrame, source: str) -> pd.DataFrame:
    block = _pooled(confusion, source)
    matrix = block.pivot(index='reference', columns='predicted', values='n')
    return matrix.reindex(
        index=list(dict.fromkeys(block['reference'])),
        columns=list(dict.fromkeys(block['predicted'])),
    ).astype(int)


def matrix_table(
    confusion: pd.DataFrame, accuracy: pd.DataFrame, source: str
) -> list[str]:
    """The pooled matrix of one source, with both accuracies on its margins."""
    matrix = _wide(confusion, source)
    scores = _pooled(accuracy, source).set_index('class')
    overall = scores.loc['ALL']
    header = [
        'reference (rows) / predicted (columns)',
        *matrix.columns,
        "producer's accuracy (recall)",
    ]
    rows = []
    for reference, counts in matrix.iterrows():
        cells = [reference]
        for predicted, n in counts.items():
            text = _fmt_count(n)
            cells.append(f'**{text}**' if predicted == reference else text)
        in_classes = reference in scores.index and reference != 'ALL'
        cells.append(
            _fmt_share(scores.loc[reference, 'producers_accuracy_recall'])
            if in_classes
            else ''
        )
        rows.append(cells)
    footer = ["consumer's accuracy (precision)"]
    for predicted in matrix.columns:
        in_classes = predicted in scores.index and predicted != 'ALL'
        footer.append(
            _fmt_share(scores.loc[predicted, 'consumers_accuracy_precision'])
            if in_classes
            else ''
        )
    footer.append(
        f'overall {_fmt_share(overall["overall_accuracy_answered"])}, '
        f'kappa {_fmt_share(overall["kappa"])}'
    )
    rows.append(footer)
    return list_table(header, rows, f'Confusion matrix, {source}')


def class_table(accuracy: pd.DataFrame, sources: list[str]) -> list[str]:
    """Per-class producer's and consumer's accuracy for every source."""
    rows = []
    for source in sources:
        for _, row in _pooled(accuracy, source).iterrows():
            if row['class'] == 'ALL':
                continue
            rows.append(
                [
                    source,
                    row['class'],
                    _fmt_count(row['n_reference']),
                    _fmt_share(row['producers_accuracy_recall']),
                    _fmt_share(row['consumers_accuracy_precision']),
                    _fmt_share(row['f1']),
                ]
            )
    header = [
        'source',
        'class',
        'support',
        "producer's accuracy (recall)",
        "consumer's accuracy (precision)",
        'F1',
    ]
    return list_table(header, rows, 'Per class, pooled')


def source_table(accuracy: pd.DataFrame, sources: list[str]) -> list[str]:
    """Overall figures for every source."""
    rows = []
    for source in sources:
        overall = _pooled(accuracy, source).set_index('class').loc['ALL']
        rows.append(
            [
                source,
                _fmt_count(overall['n_reference']),
                _fmt_share(overall['overall_accuracy_answered']),
                _fmt_share(overall['overall_accuracy_all_rows']),
                _fmt_share(overall['macro_f1']),
                _fmt_share(overall['kappa']),
                _fmt_share(overall['abstention_rate']),
            ]
        )
    header = [
        'source',
        'reference rows',
        'overall accuracy (answered)',
        'overall accuracy (all rows)',
        'macro-F1',
        "Cohen's kappa",
        'abstention rate',
    ]
    return list_table(header, rows, 'Overall, pooled')


def stratum_tables(accuracy: pd.DataFrame, source: str) -> list[str]:
    """Overall figures of one source for every written stratum."""
    lines: list[str] = []
    block = accuracy[
        accuracy['source'].eq(source)
        & accuracy['stratum_by'].ne('all')
        & accuracy['class'].eq('ALL')
    ]
    for stratum_by, rows in block.groupby('stratum_by', sort=True):
        body = [
            [
                row['stratum'],
                _fmt_count(row['n_reference']),
                _fmt_share(row['overall_accuracy_answered']),
                _fmt_share(row['kappa']),
                _fmt_share(row['abstention_rate']),
            ]
            for _, row in rows.sort_values('stratum').iterrows()
        ]
        header = [
            stratum_by,
            'reference rows',
            'overall accuracy (answered)',
            "Cohen's kappa",
            'abstention rate',
        ]
        if lines:
            lines.append('')
        lines += list_table(header, body, f'By {stratum_by}, {source}')
    return lines


def year_tables(accuracy: pd.DataFrame, sources: list[str]) -> list[str]:
    """Agreement-bin shares and error statistics for year reports."""
    header = [
        'source',
        'stratum',
        'reference rows',
        'answered',
        'exact',
        'within 1 year',
        'within 5 years',
        'mean absolute error (years)',
        'median absolute error (years)',
        'bias (years)',
    ]
    rows = []
    for source in sources:
        block = accuracy[accuracy['source'].eq(source)]
        for _, row in block.iterrows():
            label = (
                'all'
                if row['stratum_by'] == 'all'
                else f'{row["stratum_by"]} {row["stratum"]}'
            )
            rows.append(
                [
                    source,
                    label,
                    _fmt_count(row['n_reference']),
                    _fmt_count(row['n_answered']),
                    _fmt_share(row['share_exact']),
                    _fmt_share(row['share_within_1_year']),
                    _fmt_share(row['share_within_5_years']),
                    _fmt_years(row['mean_absolute_error']),
                    _fmt_years(row['median_absolute_error']),
                    _fmt_years(row['bias']),
                ]
            )
    return list_table(header, rows, 'Year built agreement')


def render_fragment(
    region: str,
    name: str,
    confusion: pd.DataFrame,
    accuracy: pd.DataFrame,
    metadata: dict,
    image: str | None = None,
) -> str:
    """The reST fragment for one report.

    Parameters
    ----------
    region : str
        Delivery region the report scores.
    name : str
        Report file stem.
    confusion, accuracy : pandas.DataFrame
        The report's long-form matrix and accuracy tables.
    metadata : dict
        Its JSON sidecar.
    image : str, optional
        Docs-absolute path of the heatmap, included when given.

    Returns
    -------
    str
    """
    sources = list(dict.fromkeys(confusion['source']))
    primary = _primary_source(sources)
    validated = str(metadata.get('generated_at') or 'unknown date')
    lines = [
        f'.. Generated by {GENERATOR} from {name}_confusion.csv of region',
        f'   {region}, validation run of {validated}; do not edit.',
        '',
    ]
    details = [
        f'Reference: {metadata.get("reference") or "not recorded"}',
    ]
    if metadata.get('tier'):
        details.append(f'match tier {metadata["tier"]}')
    details.append(f'{_fmt_count(metadata.get("n_rows"))} reference rows')
    lines += ['; '.join(details) + '.', '']

    if metadata.get('kind') == 'year_agreement':
        lines += year_tables(accuracy, sources)
    else:
        overall = _pooled(accuracy, primary).set_index('class').loc['ALL']
        lines += [
            f'Overall accuracy of {primary} '
            f'{_fmt_share(overall["overall_accuracy_answered"])} on answered '
            f'rows and {_fmt_share(overall["overall_accuracy_all_rows"])} on '
            f"all rows; Cohen's kappa {_fmt_share(overall['kappa'])}, "
            f'macro-F1 {_fmt_share(overall["macro_f1"])}.',
            '',
        ]
        lines += matrix_table(confusion, accuracy, primary)
        lines.append('')
        lines += source_table(accuracy, sources)
        lines.append('')
        lines += class_table(accuracy, sources)
        strata = stratum_tables(accuracy, primary)
        if strata:
            lines.append('')
            lines += strata
    if image:
        lines += ['', f'.. image:: {image}', f'   :alt: Heatmap of {name}, {region}']
    refused = metadata.get('refused_strata') or []
    if refused:
        lines += [
            '',
            f'{len(refused)} strata held fewer than {metadata.get("min_rows")} '
            'reference rows and are not shown.',
        ]
    return '\n'.join(lines) + '\n'


def draw_heatmap(confusion: pd.DataFrame, source: str, title: str):
    """Heatmap of one pooled matrix, shaded by row share, counts in ink."""
    import matplotlib.pyplot as plt

    matrix = _wide(confusion, source)
    totals = matrix.sum(axis=1).replace(0, 1)
    shares = matrix.div(totals, axis=0)
    fig, ax = plt.subplots(
        figsize=(1.6 + 0.9 * matrix.shape[1], 1.2 + 0.5 * matrix.shape[0])
    )
    image = ax.imshow(
        shares.to_numpy(), cmap=HEATMAP_CMAP, vmin=0, vmax=1, aspect='auto'
    )
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=35, ha='right')
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            n = int(matrix.iat[r, c])
            # Ink flips on the dark end so the count stays legible.
            ax.text(
                c,
                r,
                f'{n:,}',
                ha='center',
                va='center',
                fontsize=8,
                color='white' if shares.iat[r, c] > 0.55 else INK,
            )
    fig.colorbar(image, ax=ax, shrink=0.8, label='share of the reference row')
    ax.set_xlabel('predicted', color=INK)
    ax.set_ylabel('reference', color=INK)
    ax.set_title(title, fontsize=10, color=INK, pad=8)
    return fig


def discover_reports(recipes=RECIPES) -> list[tuple[str, Path]]:
    """(region, confusion CSV) for every report in every region's folder."""
    from openplaces.io.delivery import delivery_accuracy_dir, delivery_regions

    found = []
    for recipe in recipes:
        for spec in delivery_regions(recipe):
            if spec.get('audience') == 'team':
                continue
            region = spec['region_id']
            folder = delivery_accuracy_dir(recipe, region=region)
            for path in sorted(folder.glob('*_confusion.csv')):
                found.append((region, path))
    return found


def snapshot(region: str, confusion_path: Path, snapshot_dir=SNAPSHOT_DIR) -> Path:
    """Copy one report's three files into the docs snapshot folder."""
    name = confusion_path.name.removesuffix('_confusion.csv')
    target = Path(snapshot_dir) / region
    target.mkdir(parents=True, exist_ok=True)
    for suffix in ('_confusion.csv', '_accuracy.csv', '_confusion.json'):
        shutil.copyfile(
            confusion_path.with_name(name + suffix), target / (name + suffix)
        )
    return target / f'{name}_confusion.csv'


def render_snapshot(
    confusion_path: Path,
    generated_dir=GENERATED_DIR,
    image_dir=IMAGE_DIR,
    images: bool = True,
) -> Path:
    """Write the fragment (and heatmap) for one snapshot report.

    Parameters
    ----------
    confusion_path : pathlib.Path
        `{snapshot}/{region}/{name}_confusion.csv`.
    generated_dir, image_dir : pathlib.Path, optional
        Output folders.
    images : bool, optional
        Render the PNG heatmap (default True).

    Returns
    -------
    pathlib.Path
        The fragment written.
    """
    region = confusion_path.parent.name
    name = confusion_path.name.removesuffix('_confusion.csv')
    confusion = pd.read_csv(confusion_path, keep_default_na=False)
    confusion['n'] = confusion['n'].astype(int)
    accuracy = pd.read_csv(confusion_path.with_name(f'{name}_accuracy.csv'))
    for column in ('source', 'stratum_by', 'stratum', 'class'):
        if column in accuracy:
            accuracy[column] = accuracy[column].astype(str)
    metadata = json.loads(
        confusion_path.with_name(f'{name}_confusion.json').read_text(encoding='utf-8')
    )
    stem = f'{region}_{name}'
    image = None
    if images:
        import matplotlib.pyplot as plt

        sources = list(dict.fromkeys(confusion['source']))
        primary = _primary_source(sources)
        Path(image_dir).mkdir(parents=True, exist_ok=True)
        fig = draw_heatmap(confusion, primary, f'{name}: {primary}, {region}')
        fig.savefig(Path(image_dir) / f'{stem}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        image = f'/_static/images/validation/{stem}.png'
    fragment = Path(generated_dir) / f'{stem}.rst'
    fragment.parent.mkdir(parents=True, exist_ok=True)
    text = render_fragment(region, name, confusion, accuracy, metadata, image)
    fragment.write_text(text, encoding='utf-8', newline='\n')
    return fragment


def main(argv=None) -> None:
    """Snapshot every region's reports, then render every snapshot."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--from-snapshot',
        action='store_true',
        help='Skip the data root; re-render the committed snapshots only',
    )
    args = parser.parse_args(argv)

    if not args.from_snapshot:
        for region, path in discover_reports():
            snapshot(region, path)
    for path in sorted(SNAPSHOT_DIR.glob('*/*_confusion.csv')):
        fragment = render_snapshot(path)
        print(f'wrote {fragment.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()
