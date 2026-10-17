import itertools
import random
import time
import unittest

from generator import (canonical_key, generate_candidate, regions_connected,
                       relabel_regions, symmetries)
from solver import (DancingLinks, SolverTimeout, build_exact_cover, check_solution,
                    search_outcome, solve_dfs, solve_dlx, time_limit, validate_grid)

SOLVERS = {'dlx': solve_dlx, 'dfs': solve_dfs}

# Connected 4x4 board with no solution (checked by brute force below).
ZERO_4 = [[0, 0, 0, 0],
          [0, 1, 1, 0],
          [1, 1, 2, 3],
          [3, 3, 3, 3]]

# 5x5 board with exactly one solution. Its tokens (2,4) and (4,2) lie on the same
# diagonal two steps apart, which the rules allow.
UNIQUE_5 = [[0, 1, 2, 2, 3],
            [0, 1, 3, 3, 3],
            [0, 3, 3, 3, 3],
            [0, 0, 3, 3, 3],
            [0, 0, 4, 3, 3]]
UNIQUE_5_SOLUTION = [(0, 3), (1, 1), (2, 4), (3, 0), (4, 2)]


def rows_as_regions(n):
    return [[r] * n for r in range(n)]


def brute_force_solutions(grid):
    """All solutions by checking every permutation with the independent checker."""
    n = len(grid)
    return [sorted(enumerate(p)) for p in itertools.permutations(range(n))
            if check_solution(grid, list(enumerate(p)))]


def brute_force_prefix_count(grid):
    """Number of rule-consistent placements in rows 0..k-1 for k = 0..n (DFS tree size)."""
    n = len(grid)
    count = 0
    for k in range(n + 1):
        for cols in itertools.permutations(range(n), k):
            regions = [grid[r][c] for r, c in enumerate(cols)]
            if len(set(regions)) == k and all(abs(cols[i] - cols[i + 1]) != 1 for i in range(k - 1)):
                count += 1
    return count


class SolverAgreementTest(unittest.TestCase):
    def test_generated_instances(self):
        """Reference test: DLX and DFS agree on generated boards (extended with the checker)."""
        rng = random.Random(12345)
        for n in range(5, 9):
            for _ in range(20):
                grid = generate_candidate(n, rng)
                validate_grid(grid)
                dlx_count, _, dlx_first = solve_dlx(grid, limit=3)
                dfs_count, _, dfs_first = solve_dfs(grid, limit=3)
                self.assertEqual(dlx_count, dfs_count)
                self.assertTrue(check_solution(grid, dlx_first))
                self.assertTrue(check_solution(grid, dfs_first))
                if dlx_count == 1:
                    self.assertEqual(set(dlx_first), set(dfs_first))

    def test_full_enumeration_matches_brute_force(self):
        """Exact solution counts on random small boards, including non-unique ones."""
        rng = random.Random(2024)
        for n in (4, 5, 6):
            for _ in range(15):
                grid = relabel_regions(generate_candidate(n, rng))
                expected = len(brute_force_solutions(grid))
                for name, solver in SOLVERS.items():
                    with self.subTest(n=n, solver=name):
                        self.assertEqual(solver(grid, limit=None)[0], expected)


class FocusedCasesTest(unittest.TestCase):
    def test_zero_solutions(self):
        self.assertEqual(brute_force_solutions(ZERO_4), [])
        for name, solver in SOLVERS.items():
            with self.subTest(solver=name):
                self.assertEqual(solver(ZERO_4, limit=2)[0], 0)
                self.assertIsNone(solver(ZERO_4, limit=2)[2])
                # No placement of three tokens avoids diagonal touching on 3x3.
                self.assertEqual(solver(rows_as_regions(3), limit=2)[0], 0)

    def test_exactly_one_solution(self):
        self.assertEqual(brute_force_solutions(UNIQUE_5), [UNIQUE_5_SOLUTION])
        for name, solver in SOLVERS.items():
            with self.subTest(solver=name):
                count, _, first = solver(UNIQUE_5, limit=2)
                self.assertEqual(count, 1)
                self.assertEqual(sorted(first), UNIQUE_5_SOLUTION)

    def test_multiple_solutions(self):
        # With one region per row, solutions are permutations without adjacent
        # consecutive values: 2, 14 and 90 for n = 4, 5, 6 (OEIS A002464).
        for n, expected in ((4, 2), (5, 14), (6, 90)):
            grid = rows_as_regions(n)
            for name, solver in SOLVERS.items():
                with self.subTest(n=n, solver=name):
                    self.assertEqual(solver(grid, limit=None)[0], expected)
                    self.assertEqual(solver(grid, limit=2)[0], 2)


class DiagonalRuleTest(unittest.TestCase):
    def test_checker_diagonal_rule(self):
        grid = rows_as_regions(4)
        self.assertTrue(check_solution(grid, [(0, 1), (1, 3), (2, 0), (3, 2)]))
        # (0,1) and (1,0) touch diagonally.
        self.assertFalse(check_solution(grid, [(0, 1), (1, 0), (2, 3), (3, 2)]))

    def test_encoding_uses_only_adjacent_pairs(self):
        n = 6
        primary, secondary, rows = build_exact_cover(rows_as_regions(n))
        self.assertEqual(len(primary), 3 * n)
        self.assertEqual(len(secondary), 2 * (n - 1) ** 2)
        for _, r1, c1, r2, c2 in secondary:
            self.assertEqual((r2 - r1, abs(c2 - c1)), (1, 1))
        self.assertLessEqual(max(len(cols) for cols in rows.values()), 7)

    def test_solvers_allow_distant_diagonal_tokens(self):
        # UNIQUE_5's only solution uses (2,4) and (4,2), two steps apart on a diagonal.
        for solver in SOLVERS.values():
            self.assertEqual(sorted(solver(UNIQUE_5)[2]), UNIQUE_5_SOLUTION)


class StateAndMetricTest(unittest.TestCase):
    def test_repeated_runs_are_identical(self):
        grid = generate_candidate(7, random.Random(99))
        for solver in SOLVERS.values():
            self.assertEqual(solver(grid), solver(grid))

    def test_dlx_instance_is_restored_and_reset(self):
        dlx = DancingLinks(*build_exact_cover(UNIQUE_5))
        sizes_before = {name: h.size for name, h in dlx.headers.items()}
        first = dlx.solve(limit=2)
        second = dlx.solve(limit=2)
        self.assertEqual(first, second)
        self.assertEqual(sizes_before, {name: h.size for name, h in dlx.headers.items()})

    def test_search_state_definition(self):
        # One-cell board: the root and the solution leaf.
        for solver in SOLVERS.values():
            self.assertEqual(solver([[0]])[1], 2)
        # A full DFS enumeration visits every rule-consistent row prefix once.
        for grid in (ZERO_4, UNIQUE_5, rows_as_regions(5)):
            self.assertEqual(solve_dfs(grid, limit=None)[1], brute_force_prefix_count(grid))

    def test_time_limit_interrupts_search(self):
        start = time.perf_counter()
        with self.assertRaises(SolverTimeout):
            with time_limit(0.05):
                solve_dfs(rows_as_regions(13), limit=None)
        self.assertLess(time.perf_counter() - start, 1.0)


class DatasetHelpersTest(unittest.TestCase):
    def test_canonical_key_ignores_symmetry_and_labels(self):
        grid = relabel_regions(generate_candidate(6, random.Random(5)))
        key = canonical_key(grid)
        for variant in symmetries(grid):
            swapped = [[(5 - v) for v in row] for row in variant]
            self.assertEqual(canonical_key(swapped), key)

    def test_generated_regions_are_connected(self):
        rng = random.Random(8)
        for n in (5, 8):
            self.assertTrue(regions_connected(generate_candidate(n, rng)))
        self.assertFalse(regions_connected([[0, 1], [1, 0]]))


def traced(solver, grid, limit=2):
    events = []
    result = solver(grid, limit=limit, trace=events.append)
    return result, events


class TraceTest(unittest.TestCase):
    def test_trace_does_not_change_results(self):
        cases = (
            (ZERO_4, 2),
            (UNIQUE_5, 2),
            (rows_as_regions(5), 2),
            (rows_as_regions(5), None),
        )
        for grid, limit in cases:
            for name, solver in SOLVERS.items():
                with self.subTest(solver=name, limit=limit, n=len(grid)):
                    silent = solver(grid, limit=limit)
                    result, events = traced(solver, grid, limit)
                    self.assertEqual(result, silent)
                    self.assertEqual(events[-1]['kind'], 'finish')
                    if result[2] is None:
                        self.assertIsNone(events[-1]['first_solution'])
                    else:
                        self.assertEqual(events[-1]['first_solution'],
                                         [list(cell) for cell in result[2]])
                        self.assertTrue(check_solution(grid, result[2]))

    def test_state_visits_match_search_states(self):
        for grid, limit in ((ZERO_4, 2), (UNIQUE_5, 2), (rows_as_regions(4), None)):
            for name, solver in SOLVERS.items():
                with self.subTest(solver=name, limit=limit, n=len(grid)):
                    result, events = traced(solver, grid, limit)
                    visited = [event for event in events if event['counts_as_state']]
                    self.assertEqual(len(visited), result[1])
                    self.assertEqual([event['search_states'] for event in visited],
                                     list(range(1, result[1] + 1)))
                    self.assertGreater(len(events), result[1])

    def test_rejections_are_not_search_states(self):
        result, events = traced(solve_dfs, ZERO_4, 2)
        rejects = [event for event in events if event['kind'] == 'reject']
        self.assertTrue(rejects)
        self.assertTrue(all(not event['counts_as_state'] for event in rejects))
        self.assertIn('region', {event['reason'] for event in rejects})
        self.assertIn('column', {event['reason'] for event in rejects})
        self.assertIn('diagonal', {event['reason'] for event in rejects})
        self.assertEqual(sum(event['counts_as_state'] for event in events), result[1])

    def test_placements_are_copies_stable_across_backtracking(self):
        for name, solver in SOLVERS.items():
            with self.subTest(solver=name):
                recorded = []

                def trace(event, recorded=recorded):
                    snapshot = [tuple(cell) for cell in event['placements']]
                    recorded.append((snapshot, event['placements']))
                    event['placements'].append([99, 99])

                traced_result = solver(UNIQUE_5, limit=2, trace=trace)
                self.assertEqual(traced_result, solver(UNIQUE_5, limit=2))
                self.assertGreater(len(recorded), 1)
                self.assertEqual(len({id(live) for _, live in recorded}), len(recorded))
                self.assertEqual(recorded[0][0], [])
                for snapshot, live in recorded:
                    self.assertEqual([tuple(cell) for cell in live[:-1]], snapshot)

    def test_dfs_one_cell_trace(self):
        result, events = traced(solve_dfs, [[0]], 2)
        self.assertEqual(result, (1, 2, [(0, 0)]))
        expected = [
            {'kind': 'visit', 'counts_as_state': True, 'search_states': 1,
             'solution_count': 0, 'placements': [], 'row': 0},
            {'kind': 'choose', 'counts_as_state': False, 'search_states': 1,
             'solution_count': 0, 'placements': [[0, 0]], 'cell': [0, 0]},
            {'kind': 'solution', 'counts_as_state': True, 'search_states': 2,
             'solution_count': 1, 'placements': [[0, 0]], 'solution_index': 1},
            {'kind': 'undo', 'counts_as_state': False, 'search_states': 2,
             'solution_count': 1, 'placements': [], 'cell': [0, 0]},
            {'kind': 'finish', 'counts_as_state': False, 'search_states': 2,
             'solution_count': 1, 'placements': [], 'status': 'unique', 'reason': None,
             'first_solution': [[0, 0]]},
        ]
        self.assertEqual(len(events), len(expected))
        for number, (event, want) in enumerate(zip(events, expected), start=1):
            self.assertEqual(event['event'], number)
            for key, value in want.items():
                self.assertEqual(event[key], value)

    def test_dfs_two_by_two_rejections_and_dead_ends(self):
        result, events = traced(solve_dfs, rows_as_regions(2), 2)
        self.assertEqual(result[0], 0)
        self.assertIsNone(result[2])
        expected = [
            {'kind': 'visit', 'counts_as_state': True, 'search_states': 1, 'placements': [], 'row': 0},
            {'kind': 'choose', 'counts_as_state': False, 'placements': [[0, 0]], 'cell': [0, 0]},
            {'kind': 'visit', 'counts_as_state': True, 'search_states': 2, 'placements': [[0, 0]], 'row': 1},
            {'kind': 'reject', 'counts_as_state': False, 'row': 1, 'column': 0, 'reason': 'column',
             'placements': [[0, 0]]},
            {'kind': 'reject', 'counts_as_state': False, 'row': 1, 'column': 1, 'reason': 'diagonal',
             'placements': [[0, 0]]},
            {'kind': 'dead_end', 'counts_as_state': False, 'search_states': 2, 'row': 1,
             'placements': [[0, 0]]},
            {'kind': 'undo', 'counts_as_state': False, 'placements': [], 'cell': [0, 0]},
            {'kind': 'choose', 'counts_as_state': False, 'placements': [[0, 1]], 'cell': [0, 1]},
            {'kind': 'visit', 'counts_as_state': True, 'search_states': 3, 'placements': [[0, 1]], 'row': 1},
            {'kind': 'reject', 'counts_as_state': False, 'row': 1, 'column': 0, 'reason': 'diagonal',
             'placements': [[0, 1]]},
            {'kind': 'reject', 'counts_as_state': False, 'row': 1, 'column': 1, 'reason': 'column',
             'placements': [[0, 1]]},
            {'kind': 'dead_end', 'counts_as_state': False, 'search_states': 3, 'row': 1,
             'placements': [[0, 1]]},
            {'kind': 'undo', 'counts_as_state': False, 'placements': [], 'cell': [0, 1]},
            {'kind': 'finish', 'counts_as_state': False, 'search_states': 3, 'solution_count': 0,
             'placements': [], 'status': 'no_solution', 'reason': None, 'first_solution': None},
        ]
        self.assertEqual(len(events), len(expected))
        for number, (event, want) in enumerate(zip(events, expected), start=1):
            self.assertEqual(event['event'], number)
            for key, value in want.items():
                self.assertEqual(event[key], value, msg=f'event {number} field {key}')
        self.assertEqual(sum(event['counts_as_state'] for event in events), result[1])

    def test_dlx_one_cell_lists_primary_rule_and_candidates(self):
        result, events = traced(solve_dlx, [[0]], 2)
        self.assertEqual(result, (1, 2, [(0, 0)]))
        self.assertEqual(events[0]['kind'], 'visit')
        self.assertEqual(events[0]['column'], ['row', 0])
        self.assertEqual(events[0]['candidates'], [[0, 0]])
        self.assertTrue(events[0]['counts_as_state'])
        self.assertEqual(events[1]['kind'], 'choose')
        self.assertEqual(events[1]['cell'], [0, 0])
        self.assertEqual(events[2]['kind'], 'solution')
        self.assertTrue(events[2]['counts_as_state'])
        self.assertEqual(events[-1]['status'], 'unique')
        self.assertIsNone(events[-1]['reason'])

    def test_dlx_choice_comes_from_listed_candidates(self):
        _, events = traced(solve_dlx, UNIQUE_5, 2)
        open_at = {}
        for event in events:
            if event['counts_as_state'] and event['kind'] in ('visit', 'dead_end'):
                open_at[event['search_states']] = {tuple(cell) for cell in event['candidates']}
                self.assertIsInstance(event['column'], list)
            if event['kind'] == 'choose':
                self.assertIn(tuple(event['cell']), open_at[event['search_states']])
        _, blocked = traced(solve_dlx, ZERO_4, 2)
        dead_ends = [event for event in blocked if event['kind'] == 'dead_end']
        self.assertTrue(dead_ends)
        self.assertTrue(all(event['counts_as_state'] and event['candidates'] == []
                            for event in dead_ends))

    def test_finish_status_is_separate_from_stop_reason(self):
        cases = (
            (UNIQUE_5, 2, 1, 'unique', None),
            (ZERO_4, 2, 0, 'no_solution', None),
            (rows_as_regions(4), 2, 2, 'non_unique', 'limit'),
            (rows_as_regions(4), None, 2, 'non_unique', None),
            (UNIQUE_5, 1, 1, 'incomplete', 'limit'),
            (rows_as_regions(4), 1, 1, 'incomplete', 'limit'),
        )
        for grid, limit, count, status, reason in cases:
            for name, solver in SOLVERS.items():
                with self.subTest(solver=name, limit=limit, status=status):
                    result, events = traced(solver, grid, limit)
                    self.assertEqual(result[0], count)
                    self.assertEqual((events[-1]['status'], events[-1]['reason']), (status, reason))
                    self.assertEqual((status, reason), search_outcome(count, limit))
        self.assertEqual(search_outcome(0, 2, interrupted='timeout'), ('incomplete', 'timeout'))
        self.assertEqual(search_outcome(1, 2, interrupted='timeout'), ('incomplete', 'timeout'))
        self.assertEqual(search_outcome(2, 2, interrupted='timeout'), ('non_unique', 'timeout'))
        self.assertEqual(search_outcome(2, None, interrupted='timeout'), ('non_unique', 'timeout'))


if __name__ == '__main__':
    unittest.main()
