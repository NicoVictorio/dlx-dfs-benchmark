"""Solvers for the one-token-per-row/column/region puzzle with no diagonal touching.

Both solvers return (solution_count, search_states, first_solution).
An optional trace callback records the same search and does not change that result.

search_states is the shared search-effort metric used in this study: the number
of nodes of the search tree, i.e. the number of times the recursive search
procedure is entered. Every entered node is a partial placement that satisfies
all puzzle rules so far (the root, interior nodes, dead ends and complete
solutions are all counted once). Rejected candidate positions are not states.
"""

import contextlib
import math
import signal


class SolverTimeout(Exception):
    """Raised inside a solver when the wall-clock time limit expires."""


def _raise_timeout(signum, frame):
    raise SolverTimeout()


@contextlib.contextmanager
def time_limit(seconds):
    """Interrupt the enclosed code after `seconds` of wall-clock time.

    Uses SIGALRM (Unix only), so the solvers need no time checks in their inner
    loops and pay no per-state overhead for the limit.
    """
    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def search_outcome(solution_count, limit, interrupted=None):
    """Return (status, reason) for one solver run.

    status is what the run proved about uniqueness: unique, no_solution,
    non_unique, or incomplete. reason is why the search stopped: None when the
    tree was exhausted, 'limit' when the solution cap stopped it, or 'timeout'
    when the time limit interrupted it. Two found solutions prove non_unique
    even if the reason is limit or timeout. incomplete means the run stopped
    before uniqueness could be decided.
    """
    hit_limit = limit is not None and solution_count >= limit
    if interrupted == 'timeout':
        if solution_count >= 2:
            return 'non_unique', 'timeout'
        return 'incomplete', 'timeout'
    if solution_count >= 2:
        return 'non_unique', ('limit' if hit_limit else None)
    if hit_limit:
        return 'incomplete', 'limit'
    if solution_count == 1:
        return 'unique', None
    return 'no_solution', None


class SearchTrace:
    """Event sink for one solve call. A missing callback records nothing."""

    def __init__(self, callback):
        self.callback = callback
        self.event_no = 0

    def __bool__(self):
        return self.callback is not None

    def emit(self, kind, counts_as_state, search_states, solution_count, placements, **info):
        if self.callback is None:
            return
        self.event_no += 1
        event = {
            'event': self.event_no,
            'kind': kind,
            'counts_as_state': counts_as_state,
            'search_states': search_states,
            'solution_count': solution_count,
            'placements': [list(cell) for cell in placements],
        }
        event.update(info)
        self.callback(event)

    def finish(self, search_states, solution_count, placements, limit, first_solution):
        status, reason = search_outcome(solution_count, limit)
        copied = None if first_solution is None else [list(cell) for cell in first_solution]
        self.emit('finish', False, search_states, solution_count, placements,
                  status=status, reason=reason, first_solution=copied)


class Node:
    def __init__(self):
        self.left = self.right = self.up = self.down = self
        self.column = None
        self.size = 0
        self.name = None
        self.row_id = None


class DancingLinks:
    """Algorithm X with primary (exactly-one) and secondary (at-most-one) columns."""

    def __init__(self, primary_columns, secondary_columns, candidate_rows):
        self.root = Node()
        self.headers = {}
        previous = self.root
        primary_set = set(primary_columns)

        for name in primary_columns + secondary_columns:
            header = Node()
            header.column = header
            header.name = name
            self.headers[name] = header
            # Only primary columns are linked into the root list. Secondary
            # headers stay self-linked, so the search never has to cover them,
            # but selecting a row still removes every row that shares one.
            if name in primary_set:
                header.left = previous
                header.right = self.root
                previous.right = header
                self.root.left = header
                previous = header

        for row_id, column_names in candidate_rows.items():
            first = previous_node = None
            for name in dict.fromkeys(column_names):
                header = self.headers[name]
                node = Node()
                node.column = header
                node.row_id = row_id
                node.up = header.up
                node.down = header
                header.up.down = node
                header.up = node
                header.size += 1
                if first is None:
                    first = node
                else:
                    node.left = previous_node
                    previous_node.right = node
                previous_node = node
            if first is not None:
                first.left = previous_node
                previous_node.right = first

        self.solution_count = 0
        self.search_states = 0
        self.first_solution = None
        self.partial_solution = []
        self._trace = SearchTrace(None)

    @staticmethod
    def _candidate_cells(column):
        cells = []
        row = column.down
        while row is not column:
            cells.append([row.row_id[0], row.row_id[1]])
            row = row.down
        return cells

    def _emit(self, kind, counts_as_state, **info):
        self._trace.emit(kind, counts_as_state, self.search_states, self.solution_count,
                         self.partial_solution, **info)

    @staticmethod
    def _cover(column):
        column.right.left = column.left
        column.left.right = column.right
        row = column.down
        while row is not column:
            node = row.right
            while node is not row:
                node.down.up = node.up
                node.up.down = node.down
                node.column.size -= 1
                node = node.right
            row = row.down

    @staticmethod
    def _uncover(column):
        row = column.up
        while row is not column:
            node = row.left
            while node is not row:
                node.column.size += 1
                node.down.up = node
                node.up.down = node
                node = node.left
            row = row.up
        column.right.left = column
        column.left.right = column

    def solve(self, limit=2, trace=None):
        """Return (solution_count, search_states, first_solution), stopping at limit.

        limit=None enumerates all solutions. Counters are reset on every call.
        trace, when given, receives one dict per recorded search event.
        """
        self.solution_count = 0
        self.search_states = 0
        self.first_solution = None
        self.partial_solution = []
        self._trace = SearchTrace(trace)
        try:
            self._search(math.inf if limit is None else limit)
            self._trace.finish(self.search_states, self.solution_count, self.partial_solution,
                               limit, self.first_solution)
        finally:
            self._trace = SearchTrace(None)
        return self.solution_count, self.search_states, self.first_solution

    def _search(self, limit):
        self.search_states += 1
        if self.root.right is self.root:
            self.solution_count += 1
            if self.first_solution is None:
                self.first_solution = list(self.partial_solution)
            self._emit('solution', True, solution_index=self.solution_count)
            return

        # Choose the primary column with the fewest remaining candidates.
        column = self.root.right
        cursor = column.right
        while cursor is not self.root:
            if cursor.size < column.size:
                column = cursor
            cursor = cursor.right
        if column.size == 0:
            if self._trace:
                self._emit('dead_end', True, column=list(column.name), candidates=[])
            return

        if self._trace:
            self._emit('visit', True, column=list(column.name),
                       candidates=self._candidate_cells(column))
        self._cover(column)
        row = column.down
        while row is not column and self.solution_count < limit:
            self.partial_solution.append(row.row_id)
            if self._trace:
                self._emit('choose', False, cell=list(row.row_id), column=list(column.name))
            node = row.right
            while node is not row:
                self._cover(node.column)
                node = node.right
            self._search(limit)
            node = row.left
            while node is not row:
                self._uncover(node.column)
                node = node.left
            self.partial_solution.pop()
            if self._trace:
                self._emit('undo', False, cell=list(row.row_id), column=list(column.name))
            row = row.down
        self._uncover(column)


def build_exact_cover(region_grid):
    """Construct generalized exact-cover columns and candidate rows."""
    n = len(region_grid)
    primary = ([('row', r) for r in range(n)]
               + [('col', c) for c in range(n)]
               + [('region', k) for k in range(n)])
    secondary = []
    for r in range(n - 1):
        for c in range(n):
            for dc in (-1, 1):
                if 0 <= c + dc < n:
                    secondary.append(('diag', r, c, r + 1, c + dc))

    rows = {}
    for r in range(n):
        for c in range(n):
            columns = [('row', r), ('col', c), ('region', region_grid[r][c])]
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    a, b = sorted(((r, c), (rr, cc)))
                    columns.append(('diag', a[0], a[1], b[0], b[1]))
            rows[(r, c)] = columns
    return primary, secondary, rows


def solve_dlx(region_grid, limit=2, trace=None):
    """Build the exact-cover matrix and the linked structure, then search."""
    primary, secondary, rows = build_exact_cover(region_grid)
    return DancingLinks(primary, secondary, rows).solve(limit, trace=trace)


def solve_dfs(region_grid, limit=2, trace=None):
    """Specialized row-wise DFS baseline (one token per row, columns tried left to right)."""
    n = len(region_grid)
    numeric_limit = math.inf if limit is None else limit
    used_columns, used_regions = set(), set()
    solution_count = search_states = 0
    first_solution = None
    partial = []
    recorder = SearchTrace(trace)

    def emit(kind, counts_as_state, **info):
        recorder.emit(kind, counts_as_state, search_states, solution_count, partial, **info)

    def search(row, previous_column):
        nonlocal solution_count, search_states, first_solution
        search_states += 1
        if row == n:
            solution_count += 1
            if first_solution is None:
                first_solution = list(partial)
            emit('solution', True, solution_index=solution_count)
            return
        emit('visit', True, row=row)
        placed_any = False
        for column in range(n):
            if solution_count >= numeric_limit:
                return
            region = region_grid[row][column]
            # Diagonal neighbours can only lie in the previous row, so checking
            # the previous token is enough for the no-touching rule.
            if (column in used_columns or region in used_regions
                    or (previous_column is not None and abs(column - previous_column) == 1)):
                if trace is not None:
                    if column in used_columns:
                        reason = 'column'
                    elif region in used_regions:
                        reason = 'region'
                    else:
                        reason = 'diagonal'
                    emit('reject', False, row=row, column=column, reason=reason)
                continue
            used_columns.add(column)
            used_regions.add(region)
            partial.append((row, column))
            emit('choose', False, cell=[row, column])
            placed_any = True
            search(row + 1, column)
            partial.pop()
            used_regions.remove(region)
            used_columns.remove(column)
            emit('undo', False, cell=[row, column])
        if not placed_any:
            emit('dead_end', False, row=row)

    search(0, None)
    recorder.finish(search_states, solution_count, partial, limit, first_solution)
    return solution_count, search_states, first_solution


def validate_grid(region_grid):
    n = len(region_grid)
    if n == 0 or any(len(row) != n for row in region_grid):
        raise ValueError('The region grid must be a nonempty square matrix.')
    if set(cell for row in region_grid for cell in row) != set(range(n)):
        raise ValueError(f'Region identifiers must be exactly 0 through {n - 1}.')


def check_solution(region_grid, cells):
    """Independent rule checker; shares no code with the solvers.

    Returns True when cells is a set of (row, column) positions with exactly one
    token in every row, column and region and no two tokens diagonally adjacent.
    Tokens farther apart on the same diagonal are allowed.
    """
    n = len(region_grid)
    cells = list(cells)
    if len(cells) != n or len(set(cells)) != n:
        return False
    if any(not (0 <= r < n and 0 <= c < n) for r, c in cells):
        return False
    if sorted(r for r, _ in cells) != list(range(n)):
        return False
    if sorted(c for _, c in cells) != list(range(n)):
        return False
    if sorted(region_grid[r][c] for r, c in cells) != list(range(n)):
        return False
    for i, (r1, c1) in enumerate(cells):
        for r2, c2 in cells[i + 1:]:
            if abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
                return False
    return True
