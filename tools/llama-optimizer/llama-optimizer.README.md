# llama-optimizer.py

> **📖 For `params.txt` syntax & configuration details, see:**  
> [`llama-optimizer.params.README.md`](./llama-optimizer.params.README.md)

## 🔍 Overview
`llama-optimizer.py` is a sequential parameter wrapper for `llama-bench` (from the `llama.cpp` project). It automates the process of finding optimal inference/generation parameters for GGUF models by running targeted benchmarks, parsing the results, and iteratively carrying forward the best values.

## 🧠 Core Philosophy
- **Zero Assumptions:** The script does not validate, map, or restrict `llama-bench` CLI flags. It passes them `1:1`.
- **Delegated Validation:** If a flag is misspelled or unsupported, `llama-bench` itself will report the error. The optimizer only sequences, executes, and evaluates.
- **Sequential Carry-Over:** Each optimization phase locks in the best-found value before moving to the next parameter, simulating real-world dependency tuning.

## ⚙️ How It Works
1. **Load Configuration:** Reads a `params.txt` file (recursively if `::params-file` is used).
2. **Extract Base Args:** All lines starting with `-` or `--` are collected as the baseline CLI command.
3. **Determine Order:** Uses `::optimize-order` if defined. Otherwise, follows the exact order of `::optimize` directives in the file.
4. **Execute Phases:** For each parameter:
   - Replaces its value in the CLI with the comma-separated test list.
   - Runs `llama-bench` and saves output to timestamped `bench_*` files (`.csv` and/or `.md`, see `--output-format`).
   - Parses `t/s` (tokens/sec) for `tg` (text generation) tests.
   - Selects the maximum `t/s` value and updates the baseline for the next phase.
5. **Output:** Prints a clear summary and recommended config lines.

## 🔲 Grid Search Mode
`--grid` CLI flag or `::grid` params directive: instead of phase-by-phase tuning, **every** combination of all `::optimize` targets is tested in a single `llama-bench` run (values are comma-joined per flag).
- Base values of optimized flags are ignored — only the `::optimize` lists count.
- The run prints the total combination count; above 50 tests it asks for confirmation, then counts down 3 seconds before starting.
- Result files use the same timestamp scheme with `grid` in the name (`bench_<model>_grid_<timestamp>.csv/.md`).
- **Weighted PP/TG ranking:** every combination is grouped by its parameters (best PP + best TG each; incomplete combinations without both are skipped), both metrics are min-max normalized over the grid dataset, and each combination is scored per TG/PP weighting. Scoring lives in [`model/ranking.py`](../../model/ranking.py) (see [`model/ranking.README.md`](../../model/ranking.README.md)); the result file holds a Markdown table with one 4-row block per rank (`par` = copy-paste-ready params, `pp`/`tg` = raw t/s, `score` = weighted normalized value). Columns run from pure TG to pure PP: `TG | 70/30 | 50/50 | 30/70 | PP` (stored TG-first: `1/0 … 0/1`).
- Use `--top N` for N ranks per weighting (default: `1`) and `--table-weights` for custom columns (default: `1/0,0.7/0.3,0.5/0.5,0.3/0.7,0/1`); both are also accepted as `::top` / `::table-weights` params directives (CLI wins).

## 🖥️ CLI Usage
```bash
python llama-optimizer.py [OPTIONS]
```

| Argument          | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `--params-file`   | Path to the `params.txt` configuration file.                                |
| `--llama-dir`     | Directory containing `llama-bench` executable (default: `./bin`).           |
| `--model-dir`     | Default directory for model file picker (default: `../models`).             |
| `-m`, `--model`   | Direct path to a `.gguf` model. Overrides interactive selection.            |
| `--output-dir`    | Directory for result CSV/TXT files (default: current directory, overrides `::output-dir`). Created if missing. |
| `--output-format` | `both` (default), `csv` or `md`. `both` writes `bench_*.csv` (for LlamaGraph) + `bench_*.md` (human-readable) per run; overrides `::output-format`. |
| `--top`           | Grid mode only: result rows per TG/PP weighting (default: `1`, overrides `::top`). |
| `--table-weights` | Grid mode only: comma-separated `TG/PP` weightings, e.g. `1/0,0.7/0.3,0.5/0.5,0.3/0.7,0/1` (default: `TG\|70/30\|50/50\|30/70\|PP`, overrides `::table-weights`). |

## 📁 Output & File Naming
All result files (`bench_*.csv`, `bench_*.md`, `opt_results_*.md`) are written to `--output-dir` (or `::output-dir` from the params file, CLI wins). Default is the current working directory. The directory is created if missing. `--output-format` (or `::output-format`, CLI wins, default `both`) controls the benchmark artifacts: `both` = CSV via `llama-bench -o csv` (stdout) plus Markdown via `-oe md` (stderr) in a single run; `csv`/`md` = only one variant. The CSV variant is required for LlamaGraph plots and for the script's own auto-evaluation (`md`-only runs skip the phase evaluation with a warning).

Each optimization phase generates a CSV file with the following naming schema:
```
bench_<model_stem>_<parameter>_phase<phase_number>_<unix_timestamp>.csv
```
**Example:** `bench_Qwen3.5-27B-UD-Q6_K_XL_ngl_phase1_1744023891.csv`

This ensures:
- Clear association with the model and optimized parameter.
- No file collisions on repeated runs.
- Easy manual inspection of intermediate results.

## 🖨️ Console Output
Every `llama-bench` invocation is printed in a distinct, copy-paste-friendly block:
```
══════════════════════════════════════════════════════════════════
║ 🚀 llama-bench run:                                            ║
║ .\bin\llama-bench.exe -m model.gguf -ngl 22,26,30 -b 1024 ... ║
══════════════════════════════════════════════════════════════════
```

## 📦 Requirements
- Python 3.10+ (the script uses `X | Y` union syntax in annotations)
- Compiled `llama-bench` binary (`llama-bench` on Linux/macOS, `llama-bench.exe` on Windows)
- At least one `.gguf` model file
- A properly formatted `params.txt` (see linked guide)

## ⚠️ Notes
- Sequential mode evaluates `tg` (text generation) performance (`t/s`) only; prompt processing (`pp`) metrics are ignored there. Grid mode ranks weighted `pp`+`tg` combinations (see above).
- If `llama-bench` exits with a non-zero status, the script stops immediately and prints the exact `stderr` from the binary.
