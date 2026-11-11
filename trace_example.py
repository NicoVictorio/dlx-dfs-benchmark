"""Trace the real DFS and DLX searches; no timing is done here.

Two outputs, both from solve_dfs/solve_dlx with limit=2 and the trace callback
(the callback does not change candidate order, stopping rule, returned
solution or search_states; collect_trace checks this against a silent run):

* results/summary/trace_states.csv: for every board and solver, the search
  states visited up to and including the first solution, and the states
  visited afterwards until uniqueness was established.
* the worked example: full event traces of one board in results/trace/ and
  figures/fig5_worked_example.(png|pdf) (board, DFS tree, DLX tree).

The example board is, unless --id is given, the first board of the smallest
size (in dataset order) whose DLX and DFS search-state counts both equal the
median counts for that size.

Example:
    python trace_example.py --dataset dataset --results results --figures figures
"""
import argparse
import csv
import json
import os
import statistics

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import plot_results  # noqa: F401  (shared figure style)
from solver import validate_grid
from visualize_solver import SOLVERS, collect_trace, region_colors, write_jsonl


def load_boards(dataset):
    with open(os.path.join(dataset, 'boards.jsonl'), encoding='utf-8') as f:
        boards = [json.loads(line) for line in f if line.strip()]
    for board in boards:
        validate_grid(board['regions'])
    return boards


def state_nodes(events):
    """Search-tree nodes, one per event that counts as a search state, in visit order."""
    nodes = []
    for event in events:
        if event['counts_as_state']:
            nodes.append({'index': event['search_states'], 'kind': event['kind'],
                          'path': tuple(tuple(cell) for cell in event['placements']),
                          'column': event.get('column'),
                          'candidates': len(event.get('candidates') or [])})
    return nodes


def split_at_first_solution(nodes):
    """(states up to and including the first solution, states after it)."""
    first = next((node['index'] for node in nodes if node['kind'] == 'solution'), None)
    if first is None:
        return len(nodes), 0
    return first, len(nodes) - first


def trace_all(boards, out_csv):
    rows = []
    for board in boards:
        for key, _, solver in SOLVERS:
            result, events = collect_trace(solver, board['regions'], 2, None)
            before, after = split_at_first_solution(state_nodes(events))
            rows.append({'board_id': board['id'], 'size': board['size'], 'solver': key,
                         'solutions': result[0], 'search_states': result[1],
                         'states_to_first_solution': before,
                         'states_after_first_solution': after,
                         'rejected_candidates': sum(e['kind'] == 'reject' for e in events)})
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarise(rows):
    summary = []
    for n in sorted({r['size'] for r in rows}):
        for key, _, _ in SOLVERS:
            sel = [r for r in rows if r['size'] == n and r['solver'] == key]
            share = [100 * r['states_after_first_solution'] / r['search_states'] for r in sel]
            summary.append({
                'size': n, 'solver': key,
                'median_states_to_first_solution': statistics.median(
                    r['states_to_first_solution'] for r in sel),
                'median_states_after_first_solution': statistics.median(
                    r['states_after_first_solution'] for r in sel),
                'median_share_after_first_solution_percent': statistics.median(share),
                'min_share_after_first_solution_percent': min(share),
                'max_share_after_first_solution_percent': max(share),
            })
    return summary


def pick_example(boards, rows):
    """First board of the smallest size whose two state counts equal the size medians."""
    n = min(b['size'] for b in boards)
    states = {(r['board_id'], r['solver']): r['search_states'] for r in rows}
    med = {key: statistics.median(r['search_states'] for r in rows
                                  if r['size'] == n and r['solver'] == key) for key, _, _ in SOLVERS}
    for board in boards:
        if board['size'] == n and all(states[(board['id'], key)] == med[key] for key in med):
            return board, med
    raise SystemExit('No board matches both median state counts.')


def tree_layout(nodes):
    """x by leaf order (depth-first), y by depth; parents centred over children."""
    children = {}
    for node in nodes:
        if node['path']:
            children.setdefault(node['path'][:-1], []).append(node['path'])
    pos, leaf = {}, [0]

    def place(path, depth):
        kids = children.get(path, [])
        for kid in kids:
            place(kid, depth + 1)
        if kids:
            x = (pos[kids[0]][0] + pos[kids[-1]][0]) / 2
        else:
            x = leaf[0]
            leaf[0] += 1
        pos[path] = (x, -depth)

    place((), 0)
    return pos, children


AFTER_FIRST = '#f0c8ae'  # fill for states entered after the first solution


def constraint_text(column):
    kind, index = column[0], column[1]
    return {'row': 'row', 'col': 'column', 'region': 'region'}[kind] + f' {index}'


def draw_tree(ax, nodes, title, branch_notes=False):
    """Search tree: one marker per search state, numbered in the order entered."""
    pos, children = tree_layout(nodes)
    first = next(node['index'] for node in nodes if node['kind'] == 'solution')
    for parent, kids in children.items():
        for kid in kids:
            ax.plot([pos[parent][0], pos[kid][0]], [pos[parent][1], pos[kid][1]],
                    color='#9a9893', linewidth=0.8, zorder=1)
    for node in nodes:
        x, y = pos[node['path']]
        is_leaf = node['path'] not in children
        after = node['index'] > first
        if node['kind'] == 'solution':
            face, edge, marker, size = '#1f1f1f', '#1f1f1f', '*', 120
        elif node['kind'] == 'dead_end' or (is_leaf and node['path']):
            face, edge, marker, size = (AFTER_FIRST if after else 'white'), '#c03a2b', 'X', 55
        else:
            face, edge, marker, size = (AFTER_FIRST if after else 'white'), '#52514e', 'o', 40
        ax.scatter([x], [y], s=size, marker=marker, facecolors=face, edgecolors=edge,
                   linewidths=1.0, zorder=3)
        cell = 'root' if not node['path'] else f'({node["path"][-1][0]},{node["path"][-1][1]})'
        text = f'{node["index"]}: {cell}'
        if branch_notes and node['column'] is not None and node['kind'] == 'visit':
            plural = '' if node['candidates'] == 1 else 's'
            text += f'\nbranch on {constraint_text(node["column"])}, {node["candidates"]} option{plural}'
        elif branch_notes and node['kind'] == 'dead_end':
            text += f'\n{constraint_text(node["column"])} has no option'
        if branch_notes:
            ax.annotate(text, (x, y), xytext=(6, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=6.5, color='#33322f', linespacing=1.1)
        else:  # narrow tree: number and cell stacked under the marker
            ax.annotate(f'{node["index"]}\n{cell}', (x, y), xytext=(0, -5),
                        textcoords='offset points', ha='center', va='top', fontsize=5.8,
                        color='#33322f', linespacing=1.0, zorder=2,
                        bbox={'boxstyle': 'square,pad=0.05', 'facecolor': 'white',
                              'edgecolor': 'none'})
    ax.set_title(title, fontsize=9, loc='left')
    ax.axis('off')
    ax.margins(x=0.06, y=0.08)


def draw_example(board, events, path):
    regions = board['regions']
    n = len(regions)
    colors = region_colors(n)
    solution = {tuple(cell) for cell in board['solution']}
    fig = plt.figure(figsize=(6.8, 5.6))
    ax_board = fig.add_axes([0.05, 0.56, 0.25, 0.38])
    ax_dfs = fig.add_axes([0.35, 0.40, 0.64, 0.56])
    ax_dlx = fig.add_axes([0.03, 0.02, 0.36, 0.48])
    ax_key = fig.add_axes([0.45, 0.02, 0.54, 0.30])

    ax_board.imshow(regions, cmap=ListedColormap(colors), vmin=-0.5, vmax=n - 0.5,
                    interpolation='nearest')
    for r in range(n):
        for c in range(n):
            ax_board.text(c, r + 0.33, str(regions[r][c]), ha='center', va='center',
                          fontsize=6, color='#33322f')
            if (r, c) in solution:
                ax_board.scatter([c], [r - 0.05], s=110, marker='*', color='#1f1f1f', zorder=3)
    ax_board.set_xticks(range(n))
    ax_board.set_yticks(range(n))
    ax_board.set_xticks([i - 0.5 for i in range(n + 1)], minor=True)
    ax_board.set_yticks([i - 0.5 for i in range(n + 1)], minor=True)
    ax_board.grid(which='minor', color='white', linewidth=2)
    ax_board.grid(which='major', visible=False)
    ax_board.tick_params(which='both', length=0, labelsize=7)
    ax_board.set_xlabel('column', fontsize=7)
    ax_board.set_ylabel('row', fontsize=7)
    ax_board.set_title(f'(a) Board {board["id"]} and its solution', fontsize=9, loc='left')

    dfs_nodes = state_nodes(events['dfs'])
    dlx_nodes = state_nodes(events['dlx'])
    draw_tree(ax_dfs, dfs_nodes, f'(b) Row-wise DFS: all {len(dfs_nodes)} search states')
    draw_tree(ax_dlx, dlx_nodes, f'(c) DLX: all {len(dlx_nodes)} search states', branch_notes=True)

    ax_key.axis('off')
    keys = [('o', 'white', '#52514e', 40, 'search state with at least one child'),
            ('X', 'white', '#c03a2b', 55, 'dead end: no legal continuation'),
            ('*', '#1f1f1f', '#1f1f1f', 120, 'solution'),
            ('o', AFTER_FIRST, '#52514e', 40, 'state entered after the first solution')]
    for k, (marker, face, edge, size, text) in enumerate(keys):
        y = 0.9 - 0.2 * k
        ax_key.scatter([0.03], [y], s=size, marker=marker, facecolors=face, edgecolors=edge,
                       linewidths=1.0)
        ax_key.text(0.07, y, text, va='center', fontsize=7.5)
    ax_key.text(0.0, 0.0, 'k and (r,c): the k-th search state entered; the token at row r,\n'
                'column c was placed on the edge into it. Rejected candidates are not states.',
                fontsize=7, color='#52514e', va='bottom')
    ax_key.set_xlim(0, 1)
    ax_key.set_ylim(0, 1)
    fig.savefig(path + '.png')
    fig.savefig(path + '.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='dataset')
    parser.add_argument('--results', default='results')
    parser.add_argument('--figures', default='figures')
    parser.add_argument('--id', default=None, help='example board id (default: rule above)')
    args = parser.parse_args()

    boards = load_boards(args.dataset)
    os.makedirs(os.path.join(args.results, 'summary'), exist_ok=True)
    rows = trace_all(boards, os.path.join(args.results, 'summary', 'trace_states.csv'))
    summary = summarise(rows)
    for row in summary:
        print(f"n={row['size']} {row['solver']}: to first {row['median_states_to_first_solution']}, "
              f"after first {row['median_states_after_first_solution']} "
              f"({row['median_share_after_first_solution_percent']:.0f}% median share)")

    if args.id:
        board = next(b for b in boards if b['id'] == args.id)
        rule = f'chosen with --id {args.id}'
    else:
        board, med = pick_example(boards, rows)
        rule = f'first {board["size"]}x{board["size"]} board with median state counts {med}'
    events = {}
    os.makedirs(os.path.join(args.results, 'trace'), exist_ok=True)
    for key, name, solver in SOLVERS:
        _, events[key] = collect_trace(solver, board['regions'], 2, None)
        write_jsonl(os.path.join(args.results, 'trace', f'{board["id"]}_{key}.jsonl'), events[key])
    os.makedirs(args.figures, exist_ok=True)
    draw_example(board, events, os.path.join(args.figures, 'fig5_worked_example'))
    with open(os.path.join(args.results, 'summary', 'trace_summary.json'), 'w') as f:
        json.dump({'example_board': board['id'], 'selection': rule, 'per_size': summary}, f, indent=2)
    print(f'Example board {board["id"]} ({rule}).')


if __name__ == '__main__':
    main()
