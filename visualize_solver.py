"""Draw one board and animate the real DFS and DLX searches.

The MP4 files are slowed down for study. Their duration is not the solver
runtime. Writing MP4 requires the ffmpeg program on PATH; Matplotlib calls it
through FFMpegWriter. On Debian or Ubuntu: sudo apt install ffmpeg.

Examples:
    python generator.py --sizes 5 --per-size 1 --seed 3079 --out demo/dataset
    python visualize_solver.py board --dataset demo/dataset --out demo
    python visualize_solver.py video --dataset demo/dataset --out demo --step-seconds 0.4
"""
import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle, Patch  # noqa: E402

from solver import (SolverTimeout, check_solution, search_outcome, solve_dfs,  # noqa: E402
                    solve_dlx, time_limit, validate_grid)

SOLVERS = (('dfs', 'DFS', solve_dfs), ('dlx', 'DLX', solve_dlx))
FOOTER = 'Video diperlambat untuk pembelajaran. Durasi ini bukan runtime algoritma.'
REJECT_TEXT = {
    'column': 'kolom sudah terpakai',
    'region': 'region sudah terpakai',
    'diagonal': 'bersentuhan diagonal dengan token di atasnya',
}


def load_board(dataset, board_id):
    path = os.path.join(dataset, 'boards.jsonl')
    if not os.path.isfile(path):
        raise SystemExit(f'{path} tidak ada. Buat board dengan generator.py lebih dulu.')
    chosen = None
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            board = json.loads(line)
            if board_id is None or board['id'] == board_id:
                chosen = board
                break
    if chosen is None:
        raise SystemExit(f'Board {board_id} tidak ada di {path}.')
    validate_grid(chosen['regions'])
    return chosen


def region_colors(n):
    samples = matplotlib.colormaps['tab20']
    return [samples(i % 20) for i in range(n)]


def constraint_label(column):
    kind, index = column[0], column[1]
    if kind == 'row':
        return f'baris {index}'
    if kind == 'col':
        return f'kolom {index}'
    if kind == 'region':
        return f'region {index}'
    return ' '.join(str(part) for part in column)


def cell_label(cell):
    return f'({cell[0]}, {cell[1]})'


def finish_text(status, reason):
    """Short on-screen status. The status and the stop reason stay separate."""
    if status == 'unique':
        return 'Status: unique.\nPohon pencarian selesai dan tepat satu solusi ditemukan.'
    if status == 'no_solution':
        return 'Status: no_solution.\nPohon pencarian selesai dan tidak ada solusi.'
    if status == 'non_unique' and reason == 'limit':
        return ('Status: non_unique. Alasan berhenti: limit.\n'
                'Minimal dua solusi sudah ditemukan, jadi papan tidak unik.\n'
                'Jumlah seluruh solusi belum dihitung.')
    if status == 'non_unique' and reason == 'timeout':
        return ('Status: non_unique. Alasan berhenti: timeout.\n'
                'Minimal dua solusi sudah ditemukan sebelum waktu habis.')
    if status == 'non_unique':
        return 'Status: non_unique.\nPencarian selesai dan menemukan lebih dari satu solusi.'
    if status == 'incomplete' and reason == 'limit':
        return ('Status: incomplete. Alasan berhenti: limit.\n'
                'Keunikan belum dapat dipastikan.')
    if status == 'incomplete' and reason == 'timeout':
        return ('Status: incomplete. Alasan berhenti: timeout.\n'
                'Pencarian berhenti sebelum keunikan diketahui.')
    return f'Status: {status}.'


def describe(event):
    kind = event['kind']
    if kind == 'visit' and 'row' in event:
        return f'Mengisi baris {event["row"]}.'
    if kind == 'visit':
        cells = ', '.join(cell_label(cell) for cell in event['candidates']) or 'tidak ada'
        return f'Aturan primary: {constraint_label(event["column"])}. Kandidat: {cells}.'
    if kind == 'reject':
        return (f'Tolak {cell_label((event["row"], event["column"]))}: '
                f'{REJECT_TEXT[event["reason"]]}.')
    if kind == 'choose':
        return f'Pilih token di {cell_label(event["cell"])}.'
    if kind == 'undo':
        return f'Batalkan token di {cell_label(event["cell"])}.'
    if kind == 'dead_end' and 'row' in event:
        return f'Buntu pada baris {event["row"]}: tidak ada kandidat yang sah.'
    if kind == 'dead_end':
        return f'Buntu: {constraint_label(event["column"])} tidak punya kandidat.'
    if kind == 'solution':
        return f'Solusi ke-{event["solution_index"]} ditemukan.'
    if kind == 'finish':
        return finish_text(event['status'], event['reason'])
    return kind


def shown_tokens(event):
    if event['kind'] == 'finish' and event.get('first_solution'):
        return event['first_solution']
    return event['placements']


def timeout_finish(events, limit):
    """The solver raises before it can emit finish, so the demo appends one."""
    last = events[-1] if events else None
    solution_count = last['solution_count'] if last else 0
    search_states = last['search_states'] if last else 0
    placements = [list(cell) for cell in last['placements']] if last else []
    first = None
    for event in events:
        if event['kind'] == 'solution':
            first = [list(cell) for cell in event['placements']]
            break
    status, reason = search_outcome(solution_count, limit, interrupted='timeout')
    return {
        'event': (last['event'] + 1) if last else 1,
        'kind': 'finish',
        'counts_as_state': False,
        'search_states': search_states,
        'solution_count': solution_count,
        'placements': placements,
        'status': status,
        'reason': reason,
        'first_solution': first,
    }


def collect_trace(solver, regions, limit, seconds):
    """Run one solver with a trace and check it against a silent run."""
    events = []
    try:
        if seconds is None:
            result = solver(regions, limit=limit, trace=events.append)
        else:
            with time_limit(seconds):
                result = solver(regions, limit=limit, trace=events.append)
    except SolverTimeout:
        events.append(timeout_finish(events, limit))
        return None, events

    silent = solver(regions, limit=limit)
    if result != silent:
        raise SystemExit('Hasil solver dengan jejak berbeda dari hasil tanpa jejak.')
    if result[2] is not None and not check_solution(regions, result[2]):
        raise SystemExit('check_solution menolak solusi yang ditemukan.')
    visited = [event['search_states'] for event in events if event['counts_as_state']]
    if visited != list(range(1, result[1] + 1)):
        raise SystemExit('Jumlah kunjungan keadaan tidak sama dengan search_states.')
    finish = events[-1]
    if (finish['status'], finish['reason']) != search_outcome(result[0], limit):
        raise SystemExit('Status akhir tidak sesuai dengan search_outcome.')
    return result, events


def annotate(events, algorithm):
    for event in events:
        event['algorithm'] = algorithm
        event['detail'] = describe(event)
    return events


def write_jsonl(path, events):
    with open(path, 'w', encoding='utf-8') as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + '\n')


def draw_board(ax, regions, colors, placements, candidates=None, rejected=None, focus=None):
    n = len(regions)
    ax.imshow(regions, cmap=ListedColormap(colors), vmin=-0.5, vmax=n - 0.5,
              interpolation='nearest')
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticks([i - 0.5 for i in range(n + 1)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(n + 1)], minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', length=0)
    ax.set_xlabel('kolom')
    ax.set_ylabel('baris')
    ax.set_aspect('equal')
    occupied = {tuple(cell) for cell in placements}
    for cell in candidates or []:
        if tuple(cell) in occupied:
            continue
        ax.add_patch(Circle((cell[1], cell[0]), 0.12, facecolor='none',
                            edgecolor='#222222', linewidth=1.2, zorder=3))
    for row, column in placements:
        ax.add_patch(Circle((column, row), 0.28, facecolor='white',
                            edgecolor='#111111', linewidth=1.4, zorder=4))
    if rejected is not None:
        row, column = rejected
        ax.plot(column, row, marker='x', color='#111111', markersize=16,
                markeredgewidth=2.2, zorder=5)
    if focus is not None:
        row, column = focus
        ax.add_patch(Circle((column, row), 0.38, facecolor='none', edgecolor='#111111',
                            linewidth=2.2, zorder=5))
    handles = [Patch(facecolor=colors[region], edgecolor='#333333', label=str(region))
               for region in range(n)]
    ax.legend(handles=handles, title='region', loc='upper left', bbox_to_anchor=(1.02, 1),
              frameon=False)


def new_figure():
    figure = plt.figure(figsize=(8.4, 8.2))
    axes = figure.add_axes([0.10, 0.24, 0.68, 0.62])
    header = figure.text(0.08, 0.95, '', ha='left', va='top', fontsize=13)
    detail = figure.text(0.08, 0.16, '', ha='left', va='top', fontsize=12)
    footer = figure.text(0.08, 0.03, '', ha='left', va='bottom', fontsize=9, color='#52514e')
    return figure, axes, header, detail, footer


def paint(axes, regions, colors, event):
    rejected = None
    focus = None
    candidates = None
    if event['kind'] == 'reject':
        rejected = (event['row'], event['column'])
    elif event['kind'] in ('choose', 'undo'):
        focus = tuple(event['cell'])
    elif event['kind'] == 'visit':
        candidates = event.get('candidates')
    draw_board(axes, regions, colors, shown_tokens(event), candidates, rejected, focus)


def save_still(path, regions, colors, placements, title, detail):
    figure, axes, header, detail_text, footer = new_figure()
    draw_board(axes, regions, colors, placements)
    header.set_text(title)
    detail_text.set_text(detail)
    footer.set_text('')
    figure.savefig(path, dpi=140)
    plt.close(figure)


def repeat_count(seconds, fps):
    if seconds <= 0 or fps <= 0:
        raise SystemExit('--step-seconds, --hold-seconds, dan --fps harus lebih dari 0.')
    return max(1, round(seconds * fps))


def write_video(path, regions, colors, events, board_id, algorithm, step_seconds, fps, hold_seconds):
    figure, axes, header, detail, footer = new_figure()
    footer.set_text(FOOTER)
    repeats = repeat_count(step_seconds, fps)
    hold = max(repeats, repeat_count(hold_seconds, fps))
    total = len(events)
    writer = FFMpegWriter(fps=fps, bitrate=1800,
                          extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    with writer.saving(figure, path, dpi=120):
        for index, event in enumerate(events):
            axes.clear()
            paint(axes, regions, colors, event)
            header.set_text(
                f'{board_id}    {algorithm}    search_states: {event["search_states"]}'
                f'    event {event["event"]}/{total}')
            detail.set_text(event['detail'])
            copies = hold if index == total - 1 else repeats
            for _ in range(copies):
                writer.grab_frame()
    plt.close(figure)


def solve_pair(regions, limit):
    results = {'DLX': solve_dlx(regions, limit=limit), 'DFS': solve_dfs(regions, limit=limit)}
    if results['DLX'][0] != results['DFS'][0]:
        raise SystemExit('DFS dan DLX tidak setuju pada jumlah solusi.')
    if results['DLX'][0] == 1 and set(map(tuple, results['DLX'][2])) != set(map(tuple, results['DFS'][2])):
        raise SystemExit('DFS dan DLX tidak setuju pada solusi unik.')
    for name, result in results.items():
        if result[2] is not None and not check_solution(regions, result[2]):
            raise SystemExit(f'check_solution menolak solusi {name}.')
    return results


def command_board(args, board):
    regions = board['regions']
    results = solve_pair(regions, args.limit)
    colors = region_colors(len(regions))
    os.makedirs(args.out, exist_ok=True)
    status, reason = search_outcome(results['DLX'][0], args.limit)
    title = f'{board["id"]}    {board["size"]}×{board["size"]}'
    save_still(os.path.join(args.out, 'board.png'), regions, colors, [], title,
               'Board tanpa token.')
    solution = [] if results['DLX'][2] is None else [list(cell) for cell in results['DLX'][2]]
    solution_detail = finish_text(status, reason)
    if results['DLX'][0] > 1:
        solution_detail += '\nGambar ini menunjukkan solusi pertama DLX.'
    save_still(os.path.join(args.out, 'solution.png'), regions, colors, solution, title,
               solution_detail)
    for name, result in results.items():
        print(f'{name}: solutions {result[0]}, search_states {result[1]}, '
              f'status {status}, reason {reason}')
    print(f'Wrote {os.path.join(args.out, "board.png")} and '
          f'{os.path.join(args.out, "solution.png")}')


def command_video(args, board):
    if not FFMpegWriter.isAvailable():
        raise SystemExit(
            'ffmpeg tidak ada di PATH. Matplotlib memakainya untuk menulis MP4. '
            'Pada Debian atau Ubuntu: sudo apt install ffmpeg')
    regions = board['regions']
    colors = region_colors(len(regions))
    os.makedirs(args.out, exist_ok=True)
    for key, label, solver in SOLVERS:
        result, events = collect_trace(solver, regions, args.limit, args.time_limit)
        annotate(events, label)
        trace_path = os.path.join(args.out, f'{key}_trace.jsonl')
        video_path = os.path.join(args.out, f'{key}.mp4')
        write_jsonl(trace_path, events)
        write_video(video_path, regions, colors, events, board['id'], label,
                    args.step_seconds, args.fps, args.hold_seconds)
        finish = events[-1]
        if result is None:
            print(f'{label}: interrupted, status {finish["status"]}, reason {finish["reason"]}, '
                  f'events {len(events)}')
        else:
            print(f'{label}: solutions {result[0]}, search_states {result[1]}, '
                  f'status {finish["status"]}, reason {finish["reason"]}, events {len(events)}')
        print(f'Wrote {trace_path} and {video_path}')


def add_common(parser):
    parser.add_argument('--dataset', default='demo/dataset')
    parser.add_argument('--out', default='demo')
    parser.add_argument('--id', default=None, help='board id; default is the first board')
    parser.add_argument('--limit', type=int, default=2,
                        help='stop after this many solutions (default: 2)')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    board = commands.add_parser('board', help='write board.png and solution.png')
    add_common(board)
    video = commands.add_parser('video', help='write both search traces and MP4 files')
    add_common(video)
    video.add_argument('--step-seconds', type=float, default=0.4,
                       help='seconds each search event stays on screen (default: 0.4)')
    video.add_argument('--fps', type=int, default=10, help='MP4 frame rate (default: 10)')
    video.add_argument('--hold-seconds', type=float, default=2.0,
                       help='seconds to hold the final status (default: 2)')
    video.add_argument('--time-limit', type=float, default=None,
                       help='optional wall-clock seconds for each solver')
    args = parser.parse_args()
    board_record = load_board(args.dataset, args.id)
    if args.command == 'board':
        command_board(args, board_record)
    else:
        command_video(args, board_record)


if __name__ == '__main__':
    main()
