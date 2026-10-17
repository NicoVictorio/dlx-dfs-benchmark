"""Time DLX and DFS uniqueness verification on a frozen dataset.

Every solve uses limit=2: the search stops after two solutions or when the
tree is exhausted. Each (board, solver) pair gets one untimed warm-up and
`--repetitions` timed runs. A timed run calls solve_dlx/solve_dfs on the board,
so it includes building the exact-cover matrix and linked nodes (DLX) or the
DFS sets, and starts from fresh solver state. File reading, result checking
and summaries are outside the timed region.

Examples:
    python benchmark.py run --dataset dataset --out results
    python benchmark.py construction --dataset dataset --out results
    python benchmark.py summarize --out results

The optional `construction` command is a diagnostic: it times only the DLX
set-up (building the exact-cover rows and the linked nodes, no search) on the
same boards, to show how much of the DLX solve time is set-up cost.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone

from solver import (DancingLinks, SolverTimeout, build_exact_cover, check_solution,
                    solve_dfs, solve_dlx, time_limit)

SOLVERS = {'dlx': solve_dlx, 'dfs': solve_dfs}
RAW_FIELDS = ['board_id', 'size', 'solver', 'repetition', 'position',
              'elapsed_ms', 'search_states', 'solutions', 'status']


def load_dataset(folder):
    """Read boards and refuse to run if the file differs from the frozen manifest."""
    with open(os.path.join(folder, 'manifest.json')) as f:
        manifest = json.load(f)
    path = os.path.join(folder, manifest['boards_file'])
    with open(path, 'rb') as f:
        data = f.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest['boards_sha256']:
        raise SystemExit(f'{path} does not match the frozen manifest hash.')
    boards = [json.loads(line) for line in data.decode('utf-8').splitlines()]
    return boards, manifest


def environment_info():
    cpu = platform.processor()
    try:
        with open('/proc/cpuinfo') as f:
            cpu = next(line.split(':', 1)[1].strip() for line in f if line.startswith('model name'))
    except (OSError, StopIteration):
        pass
    clock = time.get_clock_info('perf_counter')
    return {
        'cpu': cpu,
        'logical_cpus': os.cpu_count(),
        'os': platform.platform(),
        'python': sys.version.replace('\n', ' '),
        'implementation': platform.python_implementation(),
        'timer': f'time.perf_counter_ns ({clock.implementation}, resolution {clock.resolution} s)',
        'solver_dependencies': 'Python standard library only',
    }


def measure(solver, regions, seconds):
    """One timed solve. Returns (elapsed_ms, result or None on timeout)."""
    result = None
    start = end = 0
    gc.collect()  # collect earlier garbage outside the timed region; gc stays enabled
    try:
        with time_limit(seconds):
            start = time.perf_counter_ns()
            result = solver(regions, limit=2)
            end = time.perf_counter_ns()
    except SolverTimeout:
        end = end or time.perf_counter_ns()
    return (end - start) / 1e6, result


def run_status(board, result):
    if result is None:
        return 'timeout'
    count, _, first = result
    if count == 1 and sorted(map(list, first)) == board['solution'] \
            and check_solution(board['regions'], first):
        return 'unique'
    return 'wrong'


def run(args):
    boards, manifest = load_dataset(args.dataset)
    if args.sizes:
        boards = [b for b in boards if b['size'] in args.sizes]
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    rng = random.Random(f'{args.seed}:order')
    order = list(range(len(boards)))
    rng.shuffle(order)  # random board order, so drift over time is not tied to size

    os.makedirs(os.path.join(args.out, 'raw'), exist_ok=True)
    raw_path = os.path.join(args.out, 'raw', 'measurements.csv')
    started = datetime.now(timezone.utc).isoformat(timespec='seconds')
    t0 = time.time()
    with open(raw_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(RAW_FIELDS)
        for done, index in enumerate(order, 1):
            board = boards[index]
            # Repetition 0 is the untimed warm-up; it is saved but not summarised.
            for repetition in range(args.repetitions + 1):
                names = list(SOLVERS)
                rng.shuffle(names)
                for position, name in enumerate(names, 1):
                    elapsed, result = measure(SOLVERS[name], board['regions'], args.time_limit)
                    status = run_status(board, result)
                    writer.writerow([board['id'], board['size'], name, repetition, position,
                                     f'{elapsed:.6f}',
                                     '' if result is None else result[1],
                                     '' if result is None else result[0], status])
            if done % 50 == 0 or done == len(order):
                print(f'{done}/{len(order)} boards measured', flush=True)

    env = environment_info()
    env.update({
        'command': ' '.join([os.path.basename(sys.executable), *sys.argv]),
        'started_utc': started,
        'duration_seconds': round(time.time() - t0, 1),
        'dataset_sha256': manifest['boards_sha256'],
        'dataset_seed': manifest['seed'],
        'order_seed': f'{args.seed}:order',
        'warmup_runs': 1,
        'timed_repetitions': args.repetitions,
        'time_limit_seconds': args.time_limit,
        'solution_limit': 2,
        'cpu_affinity': args.cpu,
        'garbage_collection': 'enabled during runs; gc.collect() before each run',
    })
    with open(os.path.join(args.out, 'raw', 'environment.json'), 'w') as f:
        json.dump(env, f, indent=2)
    print(f'Wrote {raw_path}')
    summarize(args)


def construction(args):
    """Diagnostic: time DLX set-up alone (no search) with the same protocol."""
    boards, _ = load_dataset(args.dataset)
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    order = list(range(len(boards)))
    random.Random(f'{args.seed}:construction').shuffle(order)
    path = os.path.join(args.out, 'raw', 'dlx_construction.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['board_id', 'size', 'repetition', 'elapsed_ms'])
        for index in order:
            board = boards[index]
            for repetition in range(args.repetitions + 1):  # repetition 0 = warm-up
                gc.collect()
                start = time.perf_counter_ns()
                DancingLinks(*build_exact_cover(board['regions']))
                elapsed = (time.perf_counter_ns() - start) / 1e6
                writer.writerow([board['id'], board['size'], repetition, f'{elapsed:.6f}'])
    print(f'Wrote {path}')
    summarize(args)


# --- Summaries ---------------------------------------------------------------

def quartiles(values):
    if len(values) == 1:
        return values[0], values[0]
    q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
    return q1, q3


def rank_correlation(x, y):
    """Spearman rank correlation, or '' when undefined (e.g. constant values)."""
    try:
        return statistics.correlation(x, y, method='ranked')
    except statistics.StatisticsError:
        return ''


def bootstrap_median_ci(values, rng, resamples=5000):
    """Percentile 95% interval for the median, resampling boards."""
    medians = sorted(statistics.median(rng.choices(values, k=len(values)))
                     for _ in range(resamples))
    return medians[int(0.025 * resamples)], medians[int(0.975 * resamples) - 1]


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(args):
    with open(os.path.join(args.out, 'raw', 'environment.json')) as f:
        limit_ms = json.load(f)['time_limit_seconds'] * 1000
    with open(os.path.join(args.out, 'raw', 'measurements.csv')) as f:
        raw = [r for r in csv.DictReader(f) if r['repetition'] != '0']

    # Step 1: summarise the repetitions within each (board, solver).
    groups = {}
    for r in raw:
        groups.setdefault((r['board_id'], int(r['size']), r['solver']), []).append(r)
    board_rows = []
    for (board_id, size, solver), runs in sorted(groups.items()):
        # A timeout is recorded at the time limit, so the board median is a lower bound.
        times = [limit_ms if r['status'] == 'timeout' else float(r['elapsed_ms']) for r in runs]
        states = {r['search_states'] for r in runs if r['search_states']}
        board_rows.append({
            'board_id': board_id, 'size': size, 'solver': solver,
            'runs': len(runs),
            'completed_runs': sum(r['status'] == 'unique' for r in runs),
            'timeouts': sum(r['status'] == 'timeout' for r in runs),
            'wrong_results': sum(r['status'] == 'wrong' for r in runs),
            'median_ms': statistics.median(times),
            'min_ms': min(times), 'max_ms': max(times),
            'cv_percent': 100 * statistics.stdev(times) / statistics.mean(times) if len(times) > 1 else 0.0,
            'search_states': int(states.pop()) if len(states) == 1 else '',
        })
    os.makedirs(os.path.join(args.out, 'summary'), exist_ok=True)
    write_csv(os.path.join(args.out, 'summary', 'board_summary.csv'), board_rows)

    # Step 2: summarise the board-level values across boards of each size.
    sizes = sorted({row['size'] for row in board_rows})
    size_rows, paired_rows = [], []
    rng = random.Random(f'{args.seed}:bootstrap')
    for size in sizes:
        per_solver = {}
        for solver in SOLVERS:
            rows = [r for r in board_rows if r['size'] == size and r['solver'] == solver]
            per_solver[solver] = {r['board_id']: r for r in rows}
            medians = [r['median_ms'] for r in rows]
            states = [r['search_states'] for r in rows if r['search_states'] != '']
            complete = sum(r['completed_runs'] == r['runs'] for r in rows)
            q1, q3 = quartiles(medians)
            s1, s3 = quartiles(states)
            size_rows.append({
                'size': size, 'solver': solver, 'boards': len(rows),
                'boards_all_runs_completed': complete,
                'completion_rate': complete / len(rows),
                'timed_runs': sum(r['runs'] for r in rows),
                'timeouts': sum(r['timeouts'] for r in rows),
                'wrong_results': sum(r['wrong_results'] for r in rows),
                'median_ms': statistics.median(medians), 'q1_ms': q1, 'q3_ms': q3,
                'min_ms': min(medians), 'max_ms': max(medians),
                'median_states': statistics.median(states), 'q1_states': s1, 'q3_states': s3,
                'min_states': min(states), 'max_states': max(states),
                'median_within_board_cv_percent': statistics.median(r['cv_percent'] for r in rows),
                'spearman_time_vs_states': rank_correlation(
                    [r['median_ms'] for r in rows if r['search_states'] != ''], states),
            })
        # Paired comparison on the same boards: ratio > 1 means DLX was faster.
        ids = sorted(set(per_solver['dlx']) & set(per_solver['dfs']))
        ratios = [per_solver['dfs'][i]['median_ms'] / per_solver['dlx'][i]['median_ms'] for i in ids]
        state_ratios = [per_solver['dfs'][i]['search_states'] / per_solver['dlx'][i]['search_states']
                        for i in ids if '' not in (per_solver['dfs'][i]['search_states'],
                                                   per_solver['dlx'][i]['search_states'])]
        q1, q3 = quartiles(ratios)
        lo, hi = bootstrap_median_ci(ratios, rng)
        paired_rows.append({
            'size': size, 'paired_boards': len(ids),
            'median_time_ratio_dfs_over_dlx': statistics.median(ratios),
            'q1_ratio': q1, 'q3_ratio': q3, 'ci95_low': lo, 'ci95_high': hi,
            'boards_dlx_faster': sum(x > 1 for x in ratios),
            'boards_dfs_faster': sum(x < 1 for x in ratios),
            'median_state_ratio_dfs_over_dlx': statistics.median(state_ratios),
            'boards_dlx_fewer_states': sum(x > 1 for x in state_ratios),
        })
    write_csv(os.path.join(args.out, 'summary', 'size_summary.csv'), size_rows)
    write_csv(os.path.join(args.out, 'summary', 'paired_summary.csv'), paired_rows)
    for row in size_rows:
        print(f"n={row['size']} {row['solver']}: median {row['median_ms']:.3f} ms "
              f"[IQR {row['q1_ms']:.3f}-{row['q3_ms']:.3f}], states {row['median_states']}, "
              f"completion {row['completion_rate']:.0%}")
    for row in paired_rows:
        print(f"n={row['size']}: DFS/DLX time ratio {row['median_time_ratio_dfs_over_dlx']:.2f} "
              f"(95% CI {row['ci95_low']:.2f}-{row['ci95_high']:.2f}); "
              f"DLX faster on {row['boards_dlx_faster']}/{row['paired_boards']} boards")
    summarize_construction(args, board_rows)


def summarize_construction(args, board_rows):
    """Per-size DLX set-up time and its share of the full DLX solve time."""
    path = os.path.join(args.out, 'raw', 'dlx_construction.csv')
    if not os.path.exists(path):
        return
    with open(path) as f:
        raw = [r for r in csv.DictReader(f) if r['repetition'] != '0']
    per_board = {}
    for r in raw:
        per_board.setdefault(r['board_id'], []).append(float(r['elapsed_ms']))
    solve_ms = {r['board_id']: r['median_ms'] for r in board_rows if r['solver'] == 'dlx'}
    sizes = {r['board_id']: r['size'] for r in board_rows}
    rows = []
    for size in sorted(set(sizes.values())):
        ids = [i for i in per_board if sizes.get(i) == size]
        medians = [statistics.median(per_board[i]) for i in ids]
        shares = [100 * statistics.median(per_board[i]) / solve_ms[i] for i in ids]
        q1, q3 = quartiles(medians)
        s1, s3 = quartiles(shares)
        rows.append({'size': size, 'boards': len(ids),
                     'median_construction_ms': statistics.median(medians), 'q1_ms': q1, 'q3_ms': q3,
                     'median_share_of_dlx_solve_percent': statistics.median(shares),
                     'q1_share': s1, 'q3_share': s3})
        print(f"n={size}: DLX set-up {rows[-1]['median_construction_ms']:.3f} ms, "
              f"{rows[-1]['median_share_of_dlx_solve_percent']:.0f}% of the DLX solve time")
    write_csv(os.path.join(args.out, 'summary', 'construction_summary.csv'), rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p_run = sub.add_parser('run', help='measure and then summarise')
    p_run.add_argument('--dataset', default='dataset')
    p_run.add_argument('--sizes', type=int, nargs='+', help='subset of sizes (default: all)')
    p_run.add_argument('--repetitions', type=int, default=10)
    p_run.add_argument('--time-limit', type=float, default=10.0, help='seconds per solve')
    p_run.add_argument('--cpu', type=int, help='pin the process to this logical CPU (Linux)')
    p_con = sub.add_parser('construction', help='diagnostic: time DLX set-up only')
    p_con.add_argument('--dataset', default='dataset')
    p_con.add_argument('--repetitions', type=int, default=10)
    p_con.add_argument('--cpu', type=int, help='pin the process to this logical CPU (Linux)')
    p_sum = sub.add_parser('summarize', help='recompute summaries from saved raw data')
    for p in (p_run, p_con, p_sum):
        p.add_argument('--out', default='results')
        p.add_argument('--seed', type=int, default=3079)
    args = parser.parse_args()
    {'run': run, 'construction': construction, 'summarize': summarize}[args.command](args)


if __name__ == '__main__':
    main()
