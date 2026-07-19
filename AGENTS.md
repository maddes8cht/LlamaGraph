# LlamaGraph — Agent Guide

## What it is

Desktop GUI (Tkinter + Matplotlib) that visualizes [llama-bench](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) CSV output. Plots prompt-processing (PP) and token-generation (TG) throughput as 2-D line or 3-D surface plots.

## Architecture: MVP

| Layer | Dir/File | Role |
|---|---|---|
| **Model** | `model/benchmark_model.py` | CSV loading, filtering, aggregation. Pure data — no Tk/Matplotlib. |
| **View** | `view/main_window.py` | Tkinter frame layout, toolbar, axis combos |
| | `view/plot_view.py` | Matplotlib canvas + `render_2d()`/`render_3d()` stateless funcs |
| | `view/left_sidebar.py` | File list, per-series PP/TG toggles |
| | `view/right_sidebar.py` | Per-dimension filter listboxes |
| **Presenter** | `presenter/plotter_presenter.py` | Orchestrator — wires callbacks, owns app state |
| **Utils** | `utils/csv_parser.py`, `utils/colors.py` | CSV parsing, color math |

## Entry point

`python llamagraph.py [path] [--ns]`
- `path`: directory or specific `.csv` file (default: `.`)
- `--ns`: start in latency (nanoseconds) view instead of tokens/s

## Key commands

```bash
pip install -r requirements.txt   # numpy, matplotlib, pytest
python -m pytest tests/           # all 164 tests
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

1. **`_save_camera` dead code** (`presenter/plotter_presenter.py:256-257`): `except Exception` is unreachable — `hasattr` only catches `AttributeError`, and the `if` block short-circuits before the try.
2. **`get_3d_points` breaks on string params** (`model/benchmark_model.py:380`): `float()` conversion is attempted on all dimension values, so string params like `gpu_name` or `params` cause empty point lists.
3. **CSV malformed rows silently dropped** (`utils/csv_parser.py`): rows with invalid n_prompt/n_gen are skipped with a print but no user feedback.

## CSV format

Must contain `avg_ts` (or `avg_ns`), `n_prompt`, and `n_gen` in the header. See `utils/csv_parser.py:132` for the probe function.

## Style conventions

- `from __future__ import annotations` in all files
- `Optional[X]` from `typing` (not `X | None` in function signatures)
- Dark-mode color constants in `utils/colors.py:12`
- Per-file hue rotation: `get_variant_color(base, idx, angle_deg=28)`
- Keyboard shortcuts: `Ctrl+T` toggle metric, `Ctrl+R` refresh, `Esc` quit
