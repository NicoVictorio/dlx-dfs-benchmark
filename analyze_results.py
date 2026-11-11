"""Extended analysis of the saved benchmark results, without re-running the solvers.

Reads results/summary/board_summary.csv and results/raw/dlx_construction.csv,
writes results/summary/extended_analysis.json and the per-board scatterplot
figures/fig4_time_vs_states.(png|pdf).

Example:
    python analyze_results.py --results results --figures figures
"""
import argparse
import csv
import json
import math
import os
import statistics

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from plot_results import COLORS, LABELS, MARKERS, SOLVERS, read_csv, size_label


def quartiles(values):
    q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
    return q1, q3


def log_log_slope(x, y):
    """Least-squares slope of log10(y) on log10(x), or None when x is constant."""
    lx = [math.log10(v) for v in x]
    ly = [math.log10(v) for v in y]
    mx, my = statistics.fmean(lx), statistics.fmean(ly)
    sxx = sum((a - mx) ** 2 for a in lx)
    if sxx == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sxx


def best_state_threshold(pairs):
    """Threshold t on DFS states for the rule 'DLX is faster iff DFS states > t'.

    pairs is a list of (dfs_states, dlx_faster). Returns the t with the most
    correctly classified boards (smallest t on ties) and that number.
    """
    best = None
    for t in sorted({s for s, _ in pairs} | {0}):
        correct = sum((s > t) == faster for s, faster in pairs)
        if best is None or correct > best[1]:
            best = (t, correct)
    return best


def board_table(results):
    rows = read_csv(os.path.join(results, 'summary', 'board_summary.csv'))
    boards = {}
    for r in rows:
        entry = boards.setdefault(r['board_id'], {'size': int(r['size'])})
        entry[r['solver']] = {'ms': float(r['median_ms']), 'states': int(r['search_states']),
                              'cv': float(r['cv_percent'])}
    with open(os.path.join(results, 'raw', 'dlx_construction.csv')) as f:
        raw = [r for r in csv.DictReader(f) if r['repetition'] != '0']
    per_board = {}
    for r in raw:
        per_board.setdefault(r['board_id'], []).append(float(r['elapsed_ms']))
    for board_id, times in per_board.items():
        boards[board_id]['construction_ms'] = statistics.median(times)
    return boards


def describe_board(board_id, b):
    return {'board_id': board_id,
            'ratio_dfs_over_dlx': b['dfs']['ms'] / b['dlx']['ms'],
            'dlx_ms': b['dlx']['ms'], 'dfs_ms': b['dfs']['ms'],
            'dlx_states': b['dlx']['states'], 'dfs_states': b['dfs']['states']}


def analyse_size(n, boards):
    ids = sorted(i for i, b in boards.items() if b['size'] == n)
    out = {'size': n, 'boards': len(ids), 'solvers': {}}
    for solver in SOLVERS:
        ms = [boards[i][solver]['ms'] for i in ids]
        states = [boards[i][solver]['states'] for i in ids]
        q1, q3 = quartiles(ms)
        out['solvers'][solver] = {
            'median_ms': statistics.median(ms), 'min_ms': min(ms), 'max_ms': max(ms),
            'max_over_min_time': max(ms) / min(ms),
            'relative_iqr_time': (q3 - q1) / statistics.median(ms),
            'median_states': statistics.median(states), 'min_states': min(states),
            'max_states': max(states), 'max_over_min_states': max(states) / min(states),
            'log_log_slope_time_vs_states': log_log_slope(states, ms),
            'median_us_per_state': statistics.median(1000 * boards[i][solver]['ms'] /
                                                     boards[i][solver]['states'] for i in ids),
        }
    ratios = [boards[i]['dfs']['ms'] / boards[i]['dlx']['ms'] for i in ids]
    out['paired'] = {
        'median_of_ratios': statistics.median(ratios),
        'ratio_of_medians': out['solvers']['dfs']['median_ms'] / out['solvers']['dlx']['median_ms'],
        'boards_dlx_faster': sum(x > 1 for x in ratios),
    }

    # Which boards does DLX win? Compare DFS tree sizes of the two groups.
    pairs = [(boards[i]['dfs']['states'], boards[i]['dfs']['ms'] > boards[i]['dlx']['ms'])
             for i in ids]
    won = [s for s, faster in pairs if faster]
    lost = [s for s, faster in pairs if not faster]
    threshold, correct = best_state_threshold(pairs)
    out['crossover'] = {
        'dfs_states_median_where_dlx_faster': statistics.median(won) if won else None,
        'dfs_states_median_where_dfs_faster': statistics.median(lost) if lost else None,
        'dfs_states_range_where_dlx_faster': [min(won), max(won)] if won else None,
        'dfs_states_range_where_dfs_faster': [min(lost), max(lost)] if lost else None,
        'best_dfs_state_threshold': threshold,
        'boards_classified_correctly': correct,
    }

    # DLX construction (timed in separate runs, so every difference is an estimate).
    construction = [boards[i]['construction_ms'] for i in ids]
    remainder = [boards[i]['dlx']['ms'] - boards[i]['construction_ms'] for i in ids]
    r1, r3 = quartiles(remainder)
    out['construction'] = {
        'median_construction_ms': statistics.median(construction),
        'boards_dfs_faster_than_dlx_construction_alone':
            sum(boards[i]['dfs']['ms'] < boards[i]['construction_ms'] for i in ids),
        'median_estimated_dlx_search_ms': statistics.median(remainder),
        'iqr_estimated_dlx_search_ms': [r1, r3],
        'boards_estimated_dlx_search_below_dfs_time':
            sum(rem < boards[i]['dfs']['ms'] for rem, i in zip(remainder, ids)),
        'median_estimated_dlx_search_us_per_state':
            statistics.median(1000 * rem / boards[i]['dlx']['states'] for rem, i in zip(remainder, ids)),
    }
    out['dlx_boards_without_branching'] = sum(boards[i]['dlx']['states'] == n + 1 for i in ids)

    by_ratio = sorted(ids, key=lambda i: boards[i]['dfs']['ms'] / boards[i]['dlx']['ms'])
    out['extremes'] = {'smallest_ratio': describe_board(by_ratio[0], boards[by_ratio[0]]),
                       'largest_ratio': describe_board(by_ratio[-1], boards[by_ratio[-1]])}
    return out


def scatter(boards, path):
    """Per-board median time against search states, one panel per board size."""
    sizes = sorted({b['size'] for b in boards.values()})
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.4), sharex=True, sharey=True)
    for ax, n in zip(axes.flat, sizes):
        for solver in SOLVERS:
            pts = [(b[solver]['states'], b[solver]['ms']) for b in boards.values() if b['size'] == n]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=10, alpha=0.6,
                       marker=MARKERS[solver], facecolors='none', edgecolors=COLORS[solver],
                       linewidths=0.8, label=LABELS[solver])
        wins = sum(b['dlx']['ms'] < b['dfs']['ms'] for b in boards.values() if b['size'] == n)
        ax.set_title(size_label(n), pad=3)
        ax.text(0.97, 0.05, f'DLX faster: {wins}/100', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=7, color='#52514e')
        ax.set_xscale('log')
        ax.set_yscale('log')
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            axis.set_minor_formatter(NullFormatter())
    for ax in axes[1]:
        ax.set_xlabel('Search states (log scale)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Median time (ms, log scale)')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02), markerscale=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path + '.png')
    fig.savefig(path + '.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results', default='results')
    parser.add_argument('--figures', default='figures')
    args = parser.parse_args()
    boards = board_table(args.results)
    sizes = sorted({b['size'] for b in boards.values()})
    analysis = [analyse_size(n, boards) for n in sizes]

    # Pooled log-log slope over all sizes, per solver.
    pooled = {s: log_log_slope([b[s]['states'] for b in boards.values()],
                               [b[s]['ms'] for b in boards.values()]) for s in SOLVERS}
    result = {'per_size': analysis, 'pooled_log_log_slope_time_vs_states': pooled}
    path = os.path.join(args.results, 'summary', 'extended_analysis.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    os.makedirs(args.figures, exist_ok=True)
    scatter(boards, os.path.join(args.figures, 'fig4_time_vs_states'))
    print(f'Wrote {path} and {args.figures}/fig4_time_vs_states.(png|pdf)')


if __name__ == '__main__':
    main()
