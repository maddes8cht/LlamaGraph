# LlamaGraph Test Suite

## Overview
This test suite provides comprehensive unit testing for the LlamaGraph project using pytest. Tests cover all layers of the MVC architecture:

- `utils/csv_parser.py`: CSV parsing and parameter detection
- `utils/colors.py`: Color manipulation and normalization helpers
- `model/benchmark_model.py`: Data model, filtering, and aggregation
- `presenter/plotter_presenter.py`: Controller orchestration layer
- `view/plot_view.py`: Matplotlib render functions and Tkinter canvas wrapper
- `view/main_window.py`: Tkinter window and toolbar logic
- `view/right_sidebar.py`: Dimension filter sidebar utilities
- `llamagraph.py`: Entry point with CLI argument parsing

## Test Coverage
Current coverage by module:

| Module | Coverage | Notes |
|---|---|---|
| `model/benchmark_model.py` | 100% | |
| `utils/colors.py` | 100% | |
| `utils/csv_parser.py` | 100% | |
| `llamagraph.py` | 100% | |
| `presenter/plotter_presenter.py` | 99% | Lines 256-257 dead code (see below) |
| `view/plot_view.py` | ~80% | Render functions covered; Tkinter widget class tested via mocks |
| `view/main_window.py` | ~15% | Core logic (`update_axis_choices`) tested; UI construction mocked |
| `view/right_sidebar.py` | ~10% | Static helpers and filter state readback tested |
| `view/left_sidebar.py` | 0% | Pure Tkinter delegation — low-value for unit tests |

### Test Count
**352 tests total**, all passing (247 without `test_plot_view.py`, which needs working native linalg — see AGENTS.md).

| Test File | Tests | What It Covers |
|---|---|---|
| `test_csv_parser.py` | 18 | CSV parsing, file metadata, validation, edge cases (empty, malformed, missing columns) |
| `test_benchmark_model.py` | 40 | Data model, filtering, 2D/3D aggregation, observer pattern, fallbacks |
| `test_colors.py` | 6 | Hue rotation, measurement combination, normalization edge cases |
| `test_presenter.py` | 89 | Scan/select/filter, comparison filter, metric toggle, 2D/3D routing, camera, pick tooltips, connector |
| `test_llamagraph.py` | 13 | CLI arg parsing, path resolution, error exit, `--ns`/`--no-md` flags |
| `test_md_parser.py` | 11 | Markdown table parsing, aliases, build-line handling |
| `test_plot_view.py` | 105 | Render functions (`render_2d`, `render_3d`, helpers), merged surfaces, tooltips, level plane, `PlotView` widget, `CustomNavigationToolbar` |
| `test_main_window.py` | 28 | `update_axis_choices`, metric button, level stepper/entry controls, key bindings |
| `test_right_sidebar.py` | 26 | `_fmt_value` formatting, `get_current_filters` filter state readback |
| `test_llama_optimizer.py` | 16 | Optimizer params parsing and helpers (see `tools/llama-optimizer/`) |

## Running Tests
```bash
# Install dependencies
pip install pytest pytest-cov

# Run all tests
python -m pytest tests/

# Run with coverage report
python -m pytest tests/ --cov=utils --cov=model --cov=presenter --cov=view --cov=llamagraph.py --cov-report=term-missing
```

## Known Issues

### CSV Parsing Edge Cases
**Issue**: Malformed rows in CSV files are only console-logged, with no user feedback in the GUI.

**Symptoms**: If a CSV contains rows with invalid n_prompt/n_gen values, those rows are skipped with a print.

**Solution**: Consider adding a warning or error collection for skipped rows.
