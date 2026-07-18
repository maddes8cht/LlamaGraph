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
**164 tests total**, all passing.

| Test File | Tests | What It Covers |
|---|---|---|
| `test_csv_parser.py` | 12 | CSV parsing, validation, edge cases (empty, malformed, missing columns) |
| `test_benchmark_model.py` | 34 | Data model, filtering, 2D/3D aggregation, observer pattern, fallbacks |
| `test_colors.py` | 6 | Hue rotation, measurement combination, normalization edge cases |
| `test_presenter.py` | 40 | Scan/select/filter, metric toggle, 2D/3D routing, camera, pick events |
| `test_llamagraph.py` | 9 | CLI arg parsing, path resolution, error exit, `--ns` flag |
| `test_plot_view.py` | 42 | Render functions (`render_2d`, `render_3d`, helpers), `PlotView` widget, `CustomNavigationToolbar`, `_draw_trisurf` (refined + direct paths) |
| `test_main_window.py` | 8 | `update_axis_choices`: combobox population, selection preservation, empty/single params |
| `test_right_sidebar.py` | 9 | `_fmt_value` formatting, `get_current_filters` filter state readback |

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

### Presenter: Dead Code in `_save_camera` (Lines 256-257)
**Issue**: The `except Exception` block inside `_save_camera()` can never be reached in Python 3.x.

**Root Cause**: `hasattr()` in Python 3.x catches only `AttributeError`. If `ax.roll` is a descriptor that raises a non-`AttributeError` exception, `hasattr()` would propagate that exception, crashing the method before the try block is entered. If `ax.roll` raises `AttributeError`, `hasattr()` returns `False` and the entire `if` block is skipped.

**Solution**: Remove the dead `try/except Exception` wrapping around `state['roll'] = ax.roll`, or replace `hasattr` with a different guard mechanism.

### Potential Bug in 3D Points Generation
**Issue**: The `get_3d_points` method requires numeric dimensions for x and y axes. When string-valued parameters (like "gpu_name" or "params" with values like "70B") are used, the method fails to convert them to floats, resulting in empty point lists.

**Symptoms**: 
- `get_3d_points("gpu_name", "params", ...)` returns empty lists
- Works correctly with numeric parameters like "n_prompt_batch", "n_gen_batch"

**Root Cause**: The method attempts `float()` conversion on all dimension values at line 380 without checking if they are numeric.

**Solution**: Add type or numeric check before float conversion, or skip non-numeric dimensions with a warning.

### CSV Parsing Edge Cases
**Issue**: Malformed rows in CSV files are silently skipped without notification.

**Symptoms**: If a CSV contains rows with invalid n_prompt/n_gen values, those rows are silently dropped.

**Solution**: Consider adding a warning or error collection for skipped rows.
