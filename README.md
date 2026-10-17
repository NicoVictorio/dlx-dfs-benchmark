# DLX versus row-wise DFS for uniqueness verification

This repository compares two exact solvers for the puzzle in which an n×n
board is divided into n regions and the player places one token in every row,
every column and every region, with no two tokens touching diagonally. Tokens
farther apart on the same diagonal are allowed. This is the one-token form of
Star Battle. LinkedIn Queens and Meowdoku use the same rules.

The two solvers are:

* **DLX**: Knuth's Algorithm X on a generalized exact-cover matrix, implemented
  with the Dancing Links data structure.
* **DFS**: a specialized row-wise depth-first backtracking search. It fills one
  row at a time and tries columns from left to right.

The benchmark task is **uniqueness verification**: search until two solutions
are found or the search tree is exhausted. All boards are synthetic. They are
not levels taken from any app.

## Installation

Python 3.12 or later is required (`statistics.correlation(..., method='ranked')`).
The time limit uses `SIGALRM`, so the generator and benchmark run on Linux or
macOS (Windows users can use WSL).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Commands

Run every command from this folder.

```bash
# 1. Dataset (sizes 9 and 10 take many hours; the saved run took about 17 hours)
.venv/bin/python generator.py --seed 1 --sizes 5 6 7 8 9 10 --per-size 100 --max-attempts 500000000 --time-limit 60 --out dataset

# 2. Benchmark, then the DLX set-up diagnostic
.venv/bin/python benchmark.py run --dataset dataset --out results --repetitions 10 --time-limit 10
.venv/bin/python benchmark.py construction --dataset dataset --out results

# 3. Summaries from the raw CSV files, without re-running the solvers
.venv/bin/python benchmark.py summarize --out results
```

`dataset/boards.jsonl` and `dataset/manifest.json` are the dataset.
`dataset/generation_log.csv` is the log of every generation attempt. It is not
in the Git repository because it is about 440 MB, above GitHub's file size
limit. Regenerating the dataset writes it again.

`results/raw/measurements.csv` is one row per run.
`results/raw/dlx_construction.csv` is the set-up diagnostic.
`benchmark.py summarize` writes the files in `results/summary/`.

`--cpu N` pins the process to one logical CPU. The saved benchmark did not use
it. Leave it out on systems without `os.sched_setaffinity`.
