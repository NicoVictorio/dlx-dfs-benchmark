"""Draw the three paper figures from the saved summary CSV files.

Example:
    python plot_results.py --results results --out figures
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

SOLVERS = ['dlx', 'dfs']
LABELS = {'dlx': 'DLX (Algorithm X + Dancing Links)', 'dfs': 'Row-wise DFS'}
COLORS = {'dlx': '#2a78d6', 'dfs': '#eb6834'}
MARKERS = {'dlx': 'o', 'dfs': 's'}
OFFSET = {'dlx': -0.08, 'dfs': 0.08}  # small horizontal shift so error bars do not overlap

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'axes.grid': True,
    'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def size_label(n):
    return f'{n}×{n}'


def finish_axes(ax, sizes, ylabel, note=None):
    """Shared styling: board-size ticks, plain-number log axis, legend above the plot."""
    ax.set_xticks(sizes, [size_label(n) for n in sizes])
    ax.set_xlim(sizes[0] - 0.5, sizes[-1] + 0.5)
    ax.set_xlabel('Board size')
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    legend = ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0),
                       title=note, title_fontsize=7, alignment='left')
    legend.get_title().set_color('#52514e')


def summary_plot(size_rows, column, ylabel, path, note):
    """Median across boards with Q1-Q3 error bars, one line per solver."""
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for solver in SOLVERS:
        rows = [r for r in size_rows if r['solver'] == solver]
        x = [int(r['size']) + OFFSET[solver] for r in rows]
        med = [float(r[f'median_{column}']) for r in rows]
        lo = [m - float(r[f'q1_{column}']) for m, r in zip(med, rows)]
        hi = [float(r[f'q3_{column}']) - m for m, r in zip(med, rows)]
        ax.errorbar(x, med, yerr=[lo, hi], color=COLORS[solver], marker=MARKERS[solver],
                    markersize=5, linewidth=1.5, capsize=3, elinewidth=1, label=LABELS[solver])
    finish_axes(ax, sorted({int(r['size']) for r in size_rows}), ylabel, note)
    fig.savefig(path + '.png')
    fig.savefig(path + '.pdf')
    plt.close(fig)


def boxplot(board_rows, path):
    """Distribution across boards of each board's median time (10 repetitions)."""
    sizes = sorted({int(r['size']) for r in board_rows})
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for solver in SOLVERS:
        data = [[float(r['median_ms']) for r in board_rows
                 if r['solver'] == solver and int(r['size']) == n] for n in sizes]
        positions = [n + 2.2 * OFFSET[solver] for n in sizes]
        ax.boxplot(data, positions=positions, widths=0.3, patch_artist=True,
                   boxprops={'facecolor': COLORS[solver], 'alpha': 0.35, 'edgecolor': COLORS[solver]},
                   medianprops={'color': COLORS[solver], 'linewidth': 1.5},
                   whiskerprops={'color': COLORS[solver]}, capprops={'color': COLORS[solver]},
                   flierprops={'marker': MARKERS[solver], 'markersize': 3,
                               'markeredgecolor': COLORS[solver], 'markerfacecolor': 'none'})
        ax.plot([], [], color=COLORS[solver], marker=MARKERS[solver], linewidth=6, alpha=0.5,
                label=LABELS[solver])
    finish_axes(ax, sizes, 'Per-board median time (ms, log scale)',
                'Each box: 100 boards; whiskers 1.5\u00d7IQR')
    fig.savefig(path + '.png')
    fig.savefig(path + '.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results', default='results')
    parser.add_argument('--out', default='figures')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    size_rows = read_csv(os.path.join(args.results, 'summary', 'size_summary.csv'))
    board_rows = read_csv(os.path.join(args.results, 'summary', 'board_summary.csv'))
    note = 'Points: median over boards; bars: Q1\u2013Q3 across boards'
    summary_plot(size_rows, 'ms', 'Elapsed time (ms, log scale)',
                 os.path.join(args.out, 'fig1_time_vs_size'), note)
    boxplot(board_rows, os.path.join(args.out, 'fig2_time_boxplots'))
    summary_plot(size_rows, 'states', 'Search states per solve (log scale)',
                 os.path.join(args.out, 'fig3_states_vs_size'), note)
    print(f'Wrote figures to {args.out}/')


if __name__ == '__main__':
    main()
