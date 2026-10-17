"""Generate and validate the frozen benchmark dataset.

Candidate boards come from the reference procedure: plant a random
non-touching placement, then grow one connected region from each token.
A candidate is accepted only if both solvers finish within the time limit,
both find exactly one solution, the solutions agree and pass the independent
checker, and the board is not a duplicate of an accepted board up to
rotation, reflection and relabelling of region IDs.

Example:
    python generator.py --seed 3079 --sizes 5 6 7 8 --per-size 100 --out dataset
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter

from solver import (SolverTimeout, check_solution, solve_dfs, solve_dlx,
                    time_limit, validate_grid)


# --- Reference generator (unchanged logic) -----------------------------------

def generate_non_touching_permutation(n, rng):
    """Find one column per row with no repeated or adjacent consecutive columns."""
    def search(row, used, placement):
        if row == n:
            return placement.copy()
        candidates = list(range(n))
        rng.shuffle(candidates)
        for column in candidates:
            if column in used or (row and abs(column - placement[-1]) == 1):
                continue
            used.add(column)
            placement.append(column)
            result = search(row + 1, used, placement)
            if result is not None:
                return result
            placement.pop()
            used.remove(column)
        return None
    return search(0, set(), [])


def grow_connected_regions(n, solution_columns, rng):
    """Grow n connected regions, each seeded by one cell of a known solution."""
    grid = [[-1] * n for _ in range(n)]
    region_cells = [[] for _ in range(n)]
    for region, column in enumerate(solution_columns):
        grid[region][column] = region
        region_cells[region].append((region, column))

    unassigned = n * n - n
    while unassigned:
        frontier = []
        for region, cells in enumerate(region_cells):
            for row, column in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = row + dr, column + dc
                    if 0 <= rr < n and 0 <= cc < n and grid[rr][cc] == -1:
                        frontier.append((region, rr, cc))
        if not frontier:
            raise RuntimeError('Region growth unexpectedly became disconnected.')
        region, row, column = rng.choice(frontier)
        if grid[row][column] == -1:
            grid[row][column] = region
            region_cells[region].append((row, column))
            unassigned -= 1
    return grid


def generate_candidate(n, rng=None):
    rng = rng or random.Random()
    placement = generate_non_touching_permutation(n, rng)
    if placement is None:
        raise ValueError(f'No valid placement found for n={n}.')
    return grow_connected_regions(n, placement, rng)


# --- Board normalisation and duplicate detection -----------------------------

def relabel_regions(grid):
    """Renumber regions 0..n-1 in order of first appearance (row-major scan).

    The reference generator gives region k the planted token of row k, so the
    raw labels reveal the solution. Relabelling removes that artefact.
    """
    mapping = {}
    for row in grid:
        for region in row:
            mapping.setdefault(region, len(mapping))
    return [[mapping[region] for region in row] for row in grid]


def symmetries(grid):
    """Yield the 8 rotations and reflections of a square grid."""
    current = [list(row) for row in grid]
    for _ in range(4):
        yield current
        yield [list(col) for col in zip(*current)]           # transpose
        current = [list(row) for row in zip(*current[::-1])]  # rotate 90 degrees


def canonical_key(grid):
    """Identical for boards that differ only by symmetry or region relabelling."""
    return min(tuple(tuple(row) for row in relabel_regions(g)) for g in symmetries(grid))


def regions_connected(grid):
    """True when every region is orthogonally connected."""
    n = len(grid)
    for region in range(n):
        cells = {(r, c) for r in range(n) for c in range(n) if grid[r][c] == region}
        if not cells:
            return False
        stack = [next(iter(cells))]
        seen = set(stack)
        while stack:
            r, c = stack.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if seen != cells:
            return False
    return True


# --- Dataset construction ----------------------------------------------------

def classify(grid, seconds):
    """Run both solvers (limit 2) and return (outcome, dlx_result, dfs_result)."""
    results = []
    for solver in (solve_dlx, solve_dfs):
        try:
            with time_limit(seconds):
                results.append(solver(grid, limit=2))
        except SolverTimeout:
            results.append(None)
    dlx, dfs = results
    if dlx is None or dfs is None:
        return 'undetermined', dlx, dfs
    if dlx[0] != dfs[0]:
        raise AssertionError(f'Solvers disagree on the solution count for {grid}')
    if dlx[0] == 0:
        return 'no_solution', dlx, dfs
    if dlx[0] >= 2:
        return 'multiple', dlx, dfs
    if sorted(dlx[2]) != sorted(dfs[2]) or not check_solution(grid, dlx[2]):
        raise AssertionError(f'Invalid or inconsistent unique solution for {grid}')
    return 'unique', dlx, dfs


def build_size(n, seed, per_size, max_attempts, seconds, log_writer):
    """Generate up to per_size accepted boards of size n within max_attempts."""
    rng = random.Random(f'{seed}:{n}')
    accepted, seen = [], set()
    outcomes = Counter()
    attempts = 0
    while len(accepted) < per_size and attempts < max_attempts:
        attempts += 1
        grid = relabel_regions(generate_candidate(n, rng))
        validate_grid(grid)
        if not regions_connected(grid):
            outcome, dlx, dfs = 'disconnected', None, None
        else:
            outcome, dlx, dfs = classify(grid, seconds)
        board_id = ''
        if outcome == 'unique':
            key = canonical_key(grid)
            if key in seen:
                outcome = 'duplicate'
            else:
                seen.add(key)
                board_id = f'n{n}-{len(accepted) + 1:03d}'
                accepted.append({
                    'id': board_id,
                    'size': n,
                    'regions': grid,
                    'solution': sorted(list(cell) for cell in dlx[2]),
                    'attempt': attempts,
                    'canonical_sha1': hashlib.sha1(repr(key).encode()).hexdigest()[:16],
                })
        outcomes[outcome] += 1
        log_writer.writerow([n, attempts, outcome,
                             '' if dlx is None else dlx[0],
                             '' if dfs is None else dfs[0], board_id])
    return accepted, attempts, outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seed', type=int, default=3079)
    parser.add_argument('--sizes', type=int, nargs='+', default=[5, 6, 7, 8])
    parser.add_argument('--per-size', type=int, default=100)
    parser.add_argument('--max-attempts', type=int, default=500_000,
                        help='upper bound on candidates generated per size')
    parser.add_argument('--time-limit', type=float, default=10.0,
                        help='seconds allowed per solver call')
    parser.add_argument('--out', default='dataset')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    boards_path = os.path.join(args.out, 'boards.jsonl')
    sizes_report = {}
    started = time.time()
    with open(os.path.join(args.out, 'generation_log.csv'), 'w', newline='') as log_file, \
            open(boards_path, 'w', encoding='utf-8') as boards_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(['size', 'attempt', 'outcome', 'dlx_solutions', 'dfs_solutions', 'board_id'])
        for n in args.sizes:
            accepted, attempts, outcomes = build_size(
                n, args.seed, args.per_size, args.max_attempts, args.time_limit, log_writer)
            for board in accepted:
                boards_file.write(json.dumps(board) + '\n')
            sizes_report[n] = {
                'accepted': len(accepted),
                'attempts': attempts,
                'outcomes': dict(outcomes),
                'shortfall': args.per_size - len(accepted),
            }
            print(f'n={n}: accepted {len(accepted)}/{args.per_size} '
                  f'after {attempts} attempts; outcomes {dict(outcomes)}', flush=True)

    with open(boards_path, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    manifest = {
        'command': ' '.join([os.path.basename(sys.executable), *sys.argv]),
        'seed': args.seed,
        'rng_per_size': "random.Random(f'{seed}:{n}')",
        'sizes': args.sizes,
        'target_per_size': args.per_size,
        'max_attempts_per_size': args.max_attempts,
        'time_limit_seconds': args.time_limit,
        'python': platform.python_version(),
        'generation_seconds': round(time.time() - started, 1),
        'results': sizes_report,
        'boards_file': 'boards.jsonl',
        'boards_sha256': digest,
    }
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Wrote {boards_path} (sha256 {digest[:12]}...)')


if __name__ == '__main__':
    main()
