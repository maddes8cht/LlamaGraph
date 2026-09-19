# LlamaGraph — Agent Guide

## What it is

Desktop GUI (Tkinter + Matplotlib) that visualizes [llama-bench](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) CSV output. Plots prompt-processing (PP) and token-generation (TG) throughput as 2-D line or 3-D surface plots.

## Architecture: MVP

| Layer | Dir/File | Role |
|---|---|---|
| **Model** | `model/benchmark_model.py` | CSV loading, filtering, aggregation. Pure data — no Tk/Matplotlib. |
| **View** | `view/main_window.py` | Tkinter frame layout, toolbars, axis combos, level controls |
| | `view/plot_view.py` | Matplotlib canvas + `render_2d()`/`render_3d()` stateless funcs (merged 3-D surfaces, click tooltips) |
| | `view/left_sidebar.py` | File list, build/model comparison filter, per-series PP/TG toggles |
| | `view/right_sidebar.py` | Per-dimension filter listboxes |
| **Presenter** | `presenter/plotter_presenter.py` | Orchestrator — wires callbacks, owns app state |
| **Utils** | `utils/csv_parser.py`, `utils/colors.py` | CSV parsing, color math |
| **Tools** | `tools/<tool-name>/` | Standalone helpers (own subdir each); decoupled from MVP, exchange files only. See `tools/README.md` |

## Entry point

`python llamagraph.py [path] [--ns] [--no-md]`
- `path`: directory or specific `.csv`/`.md` file (default: `.`)
- `--ns`: start in latency (nanoseconds) view instead of tokens/s
- `--no-md`: list CSV files only (hide `.md` files and the `.md` toggle button)

## Key commands

```bash
pip install -r requirements.txt   # numpy, matplotlib, pytest
python -m pytest tests/           # all 352 tests (247 here; test_plot_view.py needs working native linalg)
python -m pytest tests/test_benchmark_model.py -v          # single file
python -m pytest tests/test_benchmark_model.py::test_load_files -v  # single test
# coverage:
python -m pytest tests/ --cov=utils --cov=model --cov=presenter --cov=view --cov=llamagraph.py --cov-report=term-missing
```

## Requirements

- Python >= 3.12 (uses `X | Y` union syntax, `list[X]`)
- `tkinter`: stdlib but needs OS package (`python3-tk` on Ubuntu, bundled on Windows)
- No formatter/linter config exists (`.ruff_cache/` in gitignore but no ruff config)

## Known issues

1. **CSV malformed rows only console-logged** (`utils/csv_parser.py`): rows with invalid n_prompt/n_gen are skipped with a print but no user feedback.

## CSV format

Must contain `avg_ts` (or `avg_ns`), `n_prompt`, and `n_gen` in the header. See `utils/csv_parser.py:132` for the probe function.

## Style conventions

- `from __future__ import annotations` in all files
- `Optional[X]` from `typing` (not `X | None` in function signatures)
- Dark-mode color constants in `utils/colors.py:12`
- Per-file hue rotation: `get_variant_color(base, idx, angle_deg=28)`
- Keyboard shortcuts: `Ctrl+T` toggle metric, `Ctrl+R` refresh, `Esc` quit
