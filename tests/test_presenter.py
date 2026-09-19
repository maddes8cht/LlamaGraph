"""Tests for presenter/plotter_presenter.py - Orchestration layer."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model.benchmark_model import BenchmarkModel
from presenter.plotter_presenter import PlotterPresenter


# ── Mock View Components ──────────────────────────────────────────────────────


class MockSidebar:
    """Mock for LeftSidebar (file list, series toggles)."""

    def __init__(self):
        self.file_select_cb = None
        self.sort_cb = None
        self.refresh_cb = None
        self.series_toggle_cb = None
        self.select_all_cb = None
        self.deselect_all_cb = None
        self.md_toggle_cb = None
        self.md_visible: bool | None = None
        self.md_state: bool | None = None
        self.choose_directory_cb = None
        self.compat_filter_cb = None
        self.last_compat = None
        self._pp_flags: dict[int, bool] = {}
        self._tg_flags: dict[int, bool] = {}
        self.last_names: list[str] | None = None
        self.last_sort_label: str | None = None
        self.last_dir_label: str | None = None
        self.last_paths: list[Path] | None = None
        self.selected_indices: list[int] = []
        self.last_index: int | None = None

    def set_file_select_callback(self, cb):
        self.file_select_cb = cb

    def set_sort_callback(self, cb):
        self.sort_cb = cb

    def set_refresh_callback(self, cb):
        self.refresh_cb = cb

    def set_series_toggle_callback(self, cb):
        self.series_toggle_cb = cb

    def set_select_all_callback(self, cb):
        self.select_all_cb = cb

    def set_deselect_all_callback(self, cb):
        self.deselect_all_cb = cb

    def set_md_toggle_callback(self, cb):
        self.md_toggle_cb = cb

    def set_md_toggle_visible(self, visible: bool):
        self.md_visible = visible

    def set_md_toggle_state(self, active: bool):
        self.md_state = active

    def set_choose_directory_callback(self, cb):
        self.choose_directory_cb = cb

    def set_compat_filter_callback(self, cb):
        self.compat_filter_cb = cb

    def update_compat_filter(self, build_text, model_text,
                             build_active=False, model_active=False):
        self.last_compat = (build_text, model_text, build_active, model_active)

    def populate_file_list(self, names: list[str], sort_label: str):
        self.last_names = list(names)
        self.last_sort_label = sort_label

    def set_directory_label(self, path):
        self.last_dir_label = str(path)

    def update_series_toggles(self, dataset_paths: list[Path]):
        self.last_paths = list(dataset_paths)

    def select_index(self, idx: int):
        self.last_index = idx

    def select_all(self):
        pass

    def deselect_all(self):
        pass

    def get_pp_flag(self, i: int) -> bool:
        return self._pp_flags.get(i, True)

    def get_tg_flag(self, i: int) -> bool:
        return self._tg_flags.get(i, True)


class MockRightSidebar:
    """Mock for RightSidebar (dimension filters)."""

    def __init__(self):
        self.filter_change_cb = None
        self.last_update_args = None

    def set_filter_change_callback(self, cb):
        self.filter_change_cb = cb

    def update_filter_sections(self, dim_values, active_axes, current_filters):
        self.last_update_args = (dim_values, active_axes, current_filters)


class MockPlotView:
    """Mock for PlotView (Matplotlib canvas)."""

    def __init__(self):
        self.home_cb = None
        self.last_placeholder: str | None = None
        self.last_render_args = None
        self.redraw_count = 0

    def set_home_callback(self, cb):
        self.home_cb = cb

    def show_placeholder(self, text: str = ""):
        self.last_placeholder = text

    def render(self, fig, ax3d=None, on_pick_cb=None):
        self.last_render_args = (fig, ax3d, on_pick_cb)

    def redraw_idle(self):
        self.redraw_count += 1


class MockMainWindow:
    """Mock for MainWindow - no Tkinter dependencies."""

    def __init__(self):
        self.left_sidebar = MockSidebar()
        self.right_sidebar = MockRightSidebar()
        self.plot_view = MockPlotView()

        # Presenter callbacks
        self._render_cb = None
        self._toggle_3d_cb = None

        # State variables (backed by plain Python, not tk.IntVar/tk.StringVar)
        self._show_pp = 1
        self._show_tg = 1
        self._unify = 0
        self._normalize = 0
        self._mode_3d = 0
        self._show_surface = 1
        self._surface_style = "Solid"
        self._show_wireframe = 0
        self._show_projections = 0
        self._show_errors_3d = 1
        self._z_label_mode = "both-norm"
        self._show_level = 0
        self._level_val = 50
        self._subdiv = 0
        self._interp_method = "Cubic"
        self._mask_gaps = False
        self._axis_x = ""
        self._axis_y = ""

        # Sidebar tracking
        self.last_axis_choices = None
        self.last_unify_state = None
        self.last_metric_text = None
        self.last_key_bindings = None

    # Properties matching MainWindow interface
    @property
    def show_pp(self) -> bool:
        return bool(self._show_pp)

    @property
    def show_tg(self) -> bool:
        return bool(self._show_tg)

    @property
    def unify(self) -> bool:
        return bool(self._unify)

    @property
    def normalize(self) -> bool:
        return bool(self._normalize)

    @property
    def mode_3d(self) -> bool:
        return bool(self._mode_3d)

    @property
    def show_surface(self) -> bool:
        return bool(self._show_surface)

    @property
    def show_wireframe(self) -> bool:
        return bool(self._show_wireframe)

    @property
    def show_projections(self) -> bool:
        return bool(self._show_projections)

    @property
    def show_errors_3d(self) -> bool:
        return bool(self._show_errors_3d)

    @property
    def z_label_mode(self) -> str:
        return self._z_label_mode

    @property
    def show_level(self) -> bool:
        return bool(self._show_level)

    @property
    def level_val(self) -> int:
        return self._level_val

    @property
    def subdiv_level(self) -> int:
        return self._subdiv

    @property
    def interp_method(self) -> str:
        return self._interp_method

    @property
    def mask_gaps(self) -> bool:
        return self._mask_gaps

    @property
    def surface_style(self) -> str:
        return self._surface_style

    @property
    def axis_x(self) -> str:
        return self._axis_x

    @property
    def axis_y(self) -> str:
        return self._axis_y

    @property
    def root(self):
        return MagicMock()

    # Public methods
    def set_render_callback(self, cb):
        self._render_cb = cb

    def set_toggle_3d_callback(self, cb):
        self._toggle_3d_cb = cb

    def set_toggle_metric_callback(self, cb):
        self._metric_cb = cb

    def set_key_bindings(self, toggle_metric, refresh, quit_app):
        self.last_key_bindings = (toggle_metric, refresh, quit_app)

    def update_axis_choices(self, params: list[str]):
        self.last_axis_choices = list(params)

    def set_unify_state(self, enabled: bool):
        self.last_unify_state = enabled

    def set_metric_button_text(self, text: str):
        self.last_metric_text = text


# ── Helper ────────────────────────────────────────────────────────────────────


def create_bench_csv(path: Path, content: str = ""):
    """Create a minimal valid llama-bench CSV file."""
    if not content:
        content = (
            "n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params,n_gpu_layers\n"
            "1024,0,100.5,5.2,50250,2600,70B,32\n"
            "0,256,85.3,4.1,42650,2050,70B,32\n"
        )
    path.write_text(content)
    return path


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_initialization(tmp_path):
    """Test that init wires callbacks and scans files."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    assert presenter._pp_color == "#4ec9b0"
    assert presenter._tg_color == "#ce9178"
    assert presenter._show_ts is True
    assert presenter._sort_by_time is True
    assert presenter._available_csvs == []
    assert presenter._win is window
    assert presenter._cam_3d is None


def test_scan_files_no_csvs(tmp_path):
    """Test scanning a directory with no CSV files."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    assert presenter._available_csvs == []
    assert window.left_sidebar.last_names == []
    assert window.left_sidebar.last_dir_label == str(tmp_path)


def test_scan_files_finds_valid_csvs(tmp_path):
    """Test scanning a directory with valid CSV files."""
    csv1 = create_bench_csv(tmp_path / "bench1.csv")
    csv2 = create_bench_csv(tmp_path / "bench2.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    assert len(presenter._available_csvs) == 2
    assert csv1 in presenter._available_csvs
    assert csv2 in presenter._available_csvs
    assert "Sort: Time" in (window.left_sidebar.last_sort_label or "")


def test_scan_files_filters_non_bench_csv(tmp_path):
    """Test that non-llama-bench CSVs are filtered out."""
    (tmp_path / "valid.csv").write_text(
        "n_prompt,n_gen,avg_ts,avg_ns\n1024,0,100,50000"
    )
    (tmp_path / "invalid.csv").write_text("col1,col2\n1,2")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    assert len(presenter._available_csvs) == 1
    assert presenter._available_csvs[0].name == "valid.csv"


MD_TABLE = (
    "| model | backend | ngl | test | t/s |\n"
    "| ----- | ------- | --: | ----: | ---: |\n"
    "| m1 | CUDA | 22 | pp512 | 50.0 ± 1.0 |\n"
    "| m1 | CUDA | 22 | tg128 | 2.0 ± 0.1 |\n"
)


def test_scan_files_lists_md_by_default(tmp_path):
    """CSV + MD files are both listed; the .md button is visible and on."""
    (tmp_path / "bench.csv").write_text(
        "n_prompt,n_gen,avg_ts,avg_ns\n1024,0,100,50000"
    )
    (tmp_path / "bench.md").write_text(MD_TABLE)

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    names = sorted(p.name for p in presenter._available_csvs)
    assert names == ["bench.csv", "bench.md"]
    assert window.left_sidebar.md_visible is True
    assert window.left_sidebar.md_state is True


def test_no_md_hides_md_files_and_button(tmp_path):
    """show_md=False (from --no-md): no .md files, no toggle button."""
    (tmp_path / "bench.csv").write_text(
        "n_prompt,n_gen,avg_ts,avg_ns\n1024,0,100,50000"
    )
    (tmp_path / "bench.md").write_text(MD_TABLE)

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path, show_md=False)

    assert [p.name for p in presenter._available_csvs] == ["bench.csv"]
    assert window.left_sidebar.md_visible is False


def test_md_toggle_hides_and_restores(tmp_path):
    """The .md toggle filters the list and resets loaded data."""
    (tmp_path / "bench.csv").write_text(
        "n_prompt,n_gen,avg_ts,avg_ns\n1024,0,100,50000"
    )
    (tmp_path / "bench.md").write_text(MD_TABLE)

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    assert len(presenter._available_csvs) == 2

    presenter._on_md_toggle()
    assert presenter._show_md is False
    assert [p.name for p in presenter._available_csvs] == ["bench.csv"]
    assert window.left_sidebar.md_state is False

    presenter._on_md_toggle()
    assert presenter._show_md is True
    assert len(presenter._available_csvs) == 2
    assert window.left_sidebar.md_state is True


def test_md_toggle_resets_selection(tmp_path):
    """Toggling .md clears stale selection indices and loaded datasets."""
    csv = create_bench_csv(tmp_path / "bench.csv")
    (tmp_path / "bench.md").write_text(MD_TABLE)

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    presenter._on_file_select([0, 1])
    assert presenter._model.get_dataset_count() == 2

    presenter._on_md_toggle()
    assert presenter._current_selection == []
    assert presenter._model.get_dataset_count() == 0
    assert [p.name for p in presenter._available_csvs] == ["bench.csv"]


def test_md_toggle_keeps_data_when_list_unchanged(tmp_path):
    """Accidental toggle with no .md files present keeps loaded data."""
    create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    presenter._on_file_select([0])
    assert presenter._model.get_dataset_count() == 1

    presenter._on_md_toggle()  # hides .md, but there are none
    assert presenter._current_selection == [0]
    assert presenter._model.get_dataset_count() == 1
    assert [p.name for p in presenter._available_csvs] == ["bench.csv"]

    presenter._on_md_toggle()  # back on, still no .md files
    assert presenter._model.get_dataset_count() == 1


def test_scan_files_sort_by_name(tmp_path):
    """Test sorting by name after toggle."""
    create_bench_csv(tmp_path / "b.csv")
    create_bench_csv(tmp_path / "a.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    # Default is by time. Toggle to name sort.
    presenter._sort_by_time = False
    presenter.scan_files()

    assert presenter._available_csvs[0].name == "a.csv"
    assert presenter._available_csvs[1].name == "b.csv"
    assert window.left_sidebar.last_sort_label == "Sort: Name A-Z"


def test_on_sort_toggle(tmp_path):
    """Test sort toggle flips the flag and rescans."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    presenter._sort_by_time = True
    presenter._on_sort()
    assert presenter._sort_by_time is False

    presenter._on_sort()
    assert presenter._sort_by_time is True


def test_on_select_all(tmp_path):
    """Test select all delegates to sidebar."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    with patch.object(window.left_sidebar, 'select_all') as mock_select:
        presenter._on_select_all()
        mock_select.assert_called_once()


def test_on_deselect_all(tmp_path):
    """Test deselect all delegates to sidebar."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    with patch.object(window.left_sidebar, 'deselect_all') as mock_deselect:
        presenter._on_deselect_all()
        mock_deselect.assert_called_once()


def test_get_active_axes_2d(tmp_path):
    """Test active axes detection in 2D mode."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    window._axis_x = "params"
    window._mode_3d = 0
    axes = presenter._get_active_axes()
    assert axes == {"params"}


def test_get_active_axes_3d(tmp_path):
    """Test active axes detection in 3D mode."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    window._axis_x = "params"
    window._axis_y = "n_gpu_layers"
    window._mode_3d = 1
    axes = presenter._get_active_axes()
    assert axes == {"params", "n_gpu_layers"}


def test_get_active_axes_empty(tmp_path):
    """Test active axes when none are set."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    axes = presenter._get_active_axes()
    assert axes == set()


def test_toggle_metric(tmp_path):
    """Test metric toggle."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    assert presenter._show_ts is True
    presenter.toggle_metric()
    assert presenter._show_ts is False
    assert window.last_metric_text == "Switch: t/s"

    presenter.toggle_metric()
    assert presenter._show_ts is True
    assert window.last_metric_text == "Switch: ns"


def test_on_file_select(tmp_path):
    """Test file selection loads data into model."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    presenter._available_csvs = [csv1]

    presenter._on_file_select([0])

    assert presenter._model.has_data()
    assert presenter._model.get_dataset_count() == 1
    assert window.left_sidebar.last_paths == [csv1]


def test_on_file_select_invalid_index(tmp_path):
    """Test file selection with index out of range."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    # No CSVs loaded, attempt to select index 0
    presenter._on_file_select([0])

    assert not presenter._model.has_data()


def test_on_filter_change(tmp_path):
    """Test filter change propagates to model and triggers render."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    presenter._available_csvs = [csv1]
    presenter._on_file_select([0])

    filter_dict = {"params": {"70B"}}
    presenter._on_filter_change(filter_dict)

    state = presenter._model.get_filter_state()
    assert state.get("params") == {"70B"}


def test_update_right_sidebar(tmp_path):
    """Test right sidebar update."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    presenter._available_csvs = [csv1]
    presenter._on_file_select([0])

    presenter._update_right_sidebar()

    assert window.right_sidebar.last_update_args is not None
    dim_values, active_axes, filters = window.right_sidebar.last_update_args
    assert "params" in dim_values
    assert "n_gpu_layers" in dim_values


def test_on_toggle_3d(tmp_path):
    """Test 3D mode toggle triggers update and render."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    presenter._available_csvs = [csv1]
    presenter._on_file_select([0])

    with patch.object(presenter, '_update_right_sidebar') as mock_update:
        with patch.object(presenter, '_render_plot') as mock_render:
            presenter._on_toggle_3d()
            mock_update.assert_called_once()
            mock_render.assert_called_once()


def test_render_plot_no_data(tmp_path):
    """Test render plot shows placeholder when no data."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    presenter._render_plot()

    assert window.plot_view.last_placeholder is not None


def test_render_2d_called(tmp_path):
    """Test that 2D render is called with correct parameters."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    presenter._available_csvs = [csv1]
    presenter._on_file_select([0])

    # Set 2D mode
    window._mode_3d = 0
    window._axis_x = "params"

    with patch('presenter.plotter_presenter.render_2d', return_value=MagicMock()) as mock_render_2d:
        with patch.object(window.plot_view, 'render') as mock_view_render:
            presenter._render_plot()
            mock_render_2d.assert_called_once()
            mock_view_render.assert_called_once()


def test_render_3d_called(tmp_path):
    """Test that 3D render is called with correct parameters."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    presenter._available_csvs = [csv1]
    presenter._on_file_select([0])

    # Set 3D mode with two different axes
    window._mode_3d = 1
    window._axis_x = "params"
    window._axis_y = "n_gpu_layers"

    with patch('presenter.plotter_presenter.render_3d', return_value=(MagicMock(), MagicMock())) as mock_render_3d:
        with patch.object(window.plot_view, 'render') as mock_view_render:
            presenter._render_plot()
            mock_render_3d.assert_called_once()
            mock_view_render.assert_called_once()


def test_render_3d_missing_axes(tmp_path):
    """Test 3D render shows placeholder when axes are not set."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    presenter._available_csvs = [csv1]
    presenter._on_file_select([0])

    window._mode_3d = 1
    window._axis_x = ""  # No X axis set
    window._axis_y = ""

    presenter._render_plot()
    assert "Select two different axes" in (window.plot_view.last_placeholder or "")


def test_render_3d_same_axes(tmp_path):
    """Test 3D render with same X and Y shows placeholder."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    presenter._available_csvs = [csv1]
    presenter._on_file_select([0])

    window._mode_3d = 1
    window._axis_x = "params"
    window._axis_y = "params"

    presenter._render_plot()
    assert "Select two different axes" in (window.plot_view.last_placeholder or "")


def test_on_home_3d_not_3d(tmp_path):
    """Test home button returns False in 2D mode."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    window._mode_3d = 0
    result = presenter._on_home_3d()
    assert result is False


def test_handle_initial_selection(tmp_path):
    """Test initial file selection from CLI argument."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path, initial_selection_file=csv1)

    # Should have selected the file
    assert presenter._model.has_data()
    assert window.left_sidebar.last_index == 0


def test_handle_initial_selection_not_found(tmp_path):
    """Test initial selection with file not in scanned list."""
    window = MockMainWindow()
    presenter = PlotterPresenter(
        window, tmp_path,
        initial_selection_file=tmp_path / "nonexistent.csv"
    )

    # Should not crash, just ignore
    assert not presenter._model.has_data()


def test_save_camera(tmp_path):
    """Test camera state saving."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    mock_ax = MagicMock()
    mock_ax.elev = 30.0
    mock_ax.azim = -45.0
    mock_ax.get_xlim3d.return_value = (0, 10)
    mock_ax.get_ylim3d.return_value = (0, 20)
    mock_ax.get_zlim3d.return_value = (0, 30)

    state = presenter._save_camera(mock_ax)
    assert state['elev'] == 30.0
    assert state['azim'] == -45.0
    assert 'xlim' in state
    assert 'ylim' in state
    assert 'zlim' in state


def test_restore_camera(tmp_path):
    """Test camera state restoring."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    mock_ax = MagicMock()
    state = {
        'elev': 30.0,
        'azim': -45.0,
        'xlim': (0, 10),
        'ylim': (0, 20),
        'zlim': (0, 30),
    }

    presenter._restore_camera(mock_ax, state)
    mock_ax.view_init.assert_called_once_with(elev=30.0, azim=-45.0)
    mock_ax.set_xlim3d.assert_called_once_with((0, 10))
    mock_ax.set_ylim3d.assert_called_once_with((0, 20))
    mock_ax.set_zlim3d.assert_called_once_with((0, 30))


def test_camera_persistence_in_3d_render(tmp_path):
    """Test that camera is saved and restored across 3D renders."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    presenter._available_csvs = [csv1]
    presenter._on_file_select([0])

    window._mode_3d = 1
    window._axis_x = "params"
    window._axis_y = "n_gpu_layers"

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax.elev = 45.0
    mock_ax.azim = -60.0

    with patch('presenter.plotter_presenter.render_3d', return_value=(mock_fig, mock_ax)):
        with patch.object(window.plot_view, 'render'):
            # First render → camera should be saved as home
            presenter._render_plot()

            assert presenter._home_cam_3d is not None
            assert presenter._home_cam_3d['elev'] == 45.0
            assert presenter._home_cam_3d['azim'] == -60.0

            # Change camera and render again
            mock_ax.elev = 60.0
            presenter._render_plot()

            # Camera should have been saved before re-render
            # Then restore should have been called
            # (verify by checking redraw was triggered)
            assert window.plot_view.redraw_count >= 1


def test_on_choose_directory(tmp_path, monkeypatch):
    """Test directory change clears state and rescans."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    create_bench_csv(subdir / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    # Simulate choosing a new directory
    presenter._on_choose_directory(subdir)

    assert presenter._start_dir == subdir
    assert presenter._current_selection == []
    assert window.left_sidebar.last_dir_label == str(subdir)
    assert len(presenter._available_csvs) == 1
    assert window.plot_view.last_placeholder is not None


def test_toggle_metric_with_data(tmp_path):
    """Test metric toggle when data is loaded."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    presenter._available_csvs = [csv1]
    presenter._on_file_select([0])

    assert presenter._model.has_data()
    presenter.toggle_metric()
    assert presenter._show_ts is False


def test_on_file_select_with_errors(tmp_path):
    """Test file selection with a file that fails to parse."""
    # Create a file that exists but is not a valid llama-bench CSV
    invalid_file = tmp_path / "invalid.csv"
    invalid_file.write_text("not,a,valid,csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    # Manually add invalid file to available list
    presenter._available_csvs = [invalid_file]
    presenter._on_file_select([0])

    # Should not crash, just log errors
    assert not presenter._model.has_data()


def test_save_camera_with_roll(tmp_path):
    """Test _save_camera includes roll when available."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    mock_ax = MagicMock()
    mock_ax.elev = 30.0
    mock_ax.azim = -45.0
    mock_ax.roll = 15.0  # roll is available
    mock_ax.get_xlim3d.return_value = (0, 10)
    mock_ax.get_ylim3d.return_value = (0, 20)
    mock_ax.get_zlim3d.return_value = (0, 30)

    state = presenter._save_camera(mock_ax)
    assert state['elev'] == 30.0
    assert state['azim'] == -45.0
    assert state['roll'] == 15.0
    assert state['xlim'] == (0, 10)


def test_save_camera_exception_on_limits(tmp_path):
    """Test _save_camera when get_xlim3d raises."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    mock_ax = MagicMock()
    mock_ax.elev = 30.0
    mock_ax.azim = -45.0
    mock_ax.get_xlim3d.side_effect = Exception("xlim failed")
    mock_ax.get_ylim3d.side_effect = Exception("ylim failed")
    mock_ax.get_zlim3d.side_effect = Exception("zlim failed")

    state = presenter._save_camera(mock_ax)
    # Should have captured elev/azim but no limits
    assert state['elev'] == 30.0
    assert state['azim'] == -45.0
    assert 'xlim' not in state
    assert 'ylim' not in state
    assert 'zlim' not in state


def test_restore_camera_exception(tmp_path):
    """Test _restore_camera when view_init raises."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    mock_ax = MagicMock()
    mock_ax.view_init.side_effect = Exception("view_init failed")

    state = {'elev': 30.0, 'azim': -45.0}
    # Should not raise
    presenter._restore_camera(mock_ax, state)


def test_on_home_3d_true(tmp_path):
    """Test home button returns True in 3D mode with saved camera."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    presenter._available_csvs = [csv1]
    presenter._on_file_select([0])

    window._mode_3d = 1
    window._axis_x = "params"
    window._axis_y = "n_gpu_layers"
    mock_ax = MagicMock()
    mock_ax.elev = 30.0
    mock_ax.azim = -45.0

    # Set up 3D state via a full render cycle
    with patch('presenter.plotter_presenter.render_3d', return_value=(MagicMock(), mock_ax)):
        with patch.object(window.plot_view, 'render'):
            presenter._render_plot()

    assert presenter._home_cam_3d is not None
    assert presenter._current_3d_ax is not None

    # Now test home button
    result = presenter._on_home_3d()
    assert result is True


def test_on_pick(tmp_path):
    """Test pick event creates a tooltip."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    # Create a mock event with all required attributes
    mock_event = MagicMock()
    mock_event.ind = [0]
    mock_artist = MagicMock()
    mock_artist.get_label.return_value = "Test Series"
    mock_artist.get_xdata.return_value = [1.0, 2.0, 3.0]
    mock_artist.get_ydata.return_value = [10.0, 20.0, 30.0]
    mock_event.artist = mock_artist
    mock_event.mouseevent.x = 100
    mock_event.mouseevent.y = 200

    mock_toplevel = MagicMock()
    mock_label_widget = MagicMock()

    with patch('tkinter.Toplevel', return_value=mock_toplevel) as mock_tl:
        with patch('tkinter.Label', return_value=mock_label_widget) as mock_lbl:
            presenter._on_pick(mock_event)
            mock_tl.assert_called_once()
            mock_lbl.assert_called_once()
            # Verify tooltip content includes label and values
            kwargs = mock_lbl.call_args.kwargs
            assert 'Test Series' in kwargs.get('text', '')


def test_on_pick_no_ind(tmp_path):
    """Test pick event with no ind points (should return early)."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    mock_event = MagicMock()
    mock_event.ind = []

    presenter._on_pick(mock_event)
    # Should not raise, just return early


def test_on_pick_no_ind_attr(tmp_path):
    """Test pick event without 'ind' attribute (should return early)."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    mock_event = MagicMock()
    del mock_event.ind

    presenter._on_pick(mock_event)
    # Should not raise, just return early


def test_on_pick_unified_label(tmp_path):
    """Test pick event with 'Unified' in label (appends combined note)."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    mock_event = MagicMock()
    mock_event.ind = [0]
    mock_artist = MagicMock()
    mock_artist.get_label.return_value = "Unified Series (PP)"
    mock_artist.get_xdata.return_value = [1.0, 2.0]
    mock_artist.get_ydata.return_value = [10.0, 20.0]
    mock_event.artist = mock_artist
    mock_event.mouseevent.x = 100
    mock_event.mouseevent.y = 200

    with patch('tkinter.Toplevel') as mock_tl, \
         patch('tkinter.Label') as mock_lbl:
        presenter._on_pick(mock_event)
        # Verify the label text contains "Combined"
        kwargs = mock_lbl.call_args.kwargs
        assert 'Combined' in kwargs.get('text', ''), f"Expected 'Combined' in {kwargs}"


def test_save_camera_without_roll(tmp_path):
    """Test _save_camera when ax has no roll attribute (fallback path)."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    mock_ax = MagicMock(spec=['elev', 'azim', 'get_xlim3d', 'get_ylim3d', 'get_zlim3d'])
    mock_ax.elev = 30.0
    mock_ax.azim = -45.0
    del mock_ax.roll  # ensure no roll
    mock_ax.get_xlim3d.return_value = (0, 10)
    mock_ax.get_ylim3d.return_value = (0, 20)
    mock_ax.get_zlim3d.return_value = (0, 30)

    state = presenter._save_camera(mock_ax)
    assert state['elev'] == 30.0
    assert state['azim'] == -45.0
    assert 'roll' not in state


def test_save_camera_with_roll(tmp_path):
    """Test _save_camera when ax has a roll attribute."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    mock_ax = MagicMock(spec=['elev', 'azim', 'roll',
                              'get_xlim3d', 'get_ylim3d', 'get_zlim3d'])
    mock_ax.elev = 30.0
    mock_ax.azim = -45.0
    mock_ax.roll = 15.0
    mock_ax.get_xlim3d.return_value = (0, 10)
    mock_ax.get_ylim3d.return_value = (0, 20)
    mock_ax.get_zlim3d.return_value = (0, 30)

    state = presenter._save_camera(mock_ax)
    assert state['elev'] == 30.0
    assert state['azim'] == -45.0
    assert state['roll'] == 15.0
    assert state['xlim'] == (0, 10)


def test_save_camera_exception_path(tmp_path):
    """Test _save_camera when get_xlim3d raises (covers dead except block)."""
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)

    mock_ax = MagicMock()
    mock_ax.elev = 30.0
    mock_ax.azim = -45.0
    mock_ax.get_xlim3d.side_effect = RuntimeError("No 3D support")

    state = presenter._save_camera(mock_ax)
    assert state['elev'] == 30.0
    assert state['azim'] == -45.0
    assert 'xlim' not in state  # exception caught and ignored


# ── Comparison filter (same build / same model) ─────────────────────────────


def _create_meta_csv(path: Path, build_number: str, model_filename: str,
                     build_commit: str = "abc1234") -> Path:
    """Create a valid llama-bench CSV carrying build/model metadata."""
    path.write_text(
        "build_commit,build_number,model_filename,model_type,"
        "n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,n_gpu_layers\n"
        f"{build_commit},{build_number},{model_filename},TestModel,"
        "1024,0,100.5,5.2,50250,2600,32\n"
        f"{build_commit},{build_number},{model_filename},TestModel,"
        "0,256,85.3,4.1,42650,2050,32\n"
    )
    return path


def test_compat_box_hidden_without_selection(tmp_path):
    """No file selected → the comparison-filter box stays hidden."""
    _create_meta_csv(tmp_path / "a.csv", "11028", "model-A.gguf")
    window = MockMainWindow()
    PlotterPresenter(window, tmp_path)

    assert window.left_sidebar.last_compat[0] is None
    assert window.left_sidebar.last_compat[1] is None


def test_compat_box_shows_reference_after_select(tmp_path):
    """Selecting a file offers its build number and model as filter options."""
    _create_meta_csv(tmp_path / "a.csv", "11028", "model-A.gguf")
    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path)
    presenter._on_file_select([0])

    build_text, model_text, build_active, model_active = \
        window.left_sidebar.last_compat
    assert "11028" in (build_text or "")
    assert "model-A.gguf" in (model_text or "")
    assert build_active is False
    assert model_active is False


def test_build_filter_narrows_visible_list(tmp_path):
    """'Only this build' hides files with a different build number."""
    _create_meta_csv(tmp_path / "a.csv", "11028", "same.gguf")
    _create_meta_csv(tmp_path / "b.csv", "99999", "same.gguf")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path, show_md=False)
    assert len(presenter._visible_csvs) == 2

    # Select one file, then lock to its build
    names = window.left_sidebar.last_names
    idx_a = names.index("a.csv")
    presenter._on_file_select([idx_a])
    assert presenter._ref_build == "11028"

    presenter._on_compat_filter('build', True)
    visible = [p.name for p in presenter._visible_csvs]
    assert visible == ["a.csv"]
    assert window.left_sidebar.last_names == ["a.csv"]
    assert window.left_sidebar.last_compat[2] is True


def test_model_filter_uses_basename(tmp_path):
    """Model comparison uses the filename, not the absolute CSV path."""
    _create_meta_csv(tmp_path / "a.csv", "11028", "G:\\models\\foo.gguf")
    _create_meta_csv(tmp_path / "b.csv", "11028", "/other/dir/foo.gguf")
    _create_meta_csv(tmp_path / "c.csv", "11028", "bar.gguf")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path, show_md=False)
    names = window.left_sidebar.last_names
    presenter._on_file_select([names.index("a.csv")])

    presenter._on_compat_filter('model', True)
    visible = sorted(p.name for p in presenter._visible_csvs)
    assert visible == ["a.csv", "b.csv"]


def test_deselect_all_resets_filter_and_restores_list(tmp_path):
    """Empty selection clears the filters and shows all files again."""
    _create_meta_csv(tmp_path / "a.csv", "11028", "same.gguf")
    _create_meta_csv(tmp_path / "b.csv", "99999", "same.gguf")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path, show_md=False)
    names = window.left_sidebar.last_names
    presenter._on_file_select([names.index("a.csv")])
    presenter._on_compat_filter('build', True)
    assert len(presenter._visible_csvs) == 1

    presenter._on_file_select([])  # last file deselected
    assert presenter._build_only is False
    assert presenter._model_only is False
    assert len(presenter._visible_csvs) == 2
    assert window.left_sidebar.last_compat[0] is None


def test_select_all_covers_only_visible_files(tmp_path):
    """With an active filter, indices map to the visible (filtered) list."""
    _create_meta_csv(tmp_path / "a.csv", "11028", "same.gguf")
    _create_meta_csv(tmp_path / "b.csv", "99999", "same.gguf")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path, show_md=False)
    names = window.left_sidebar.last_names
    presenter._on_file_select([names.index("a.csv")])
    presenter._on_compat_filter('build', True)

    # "Select All" on the single visible file loads exactly that file
    presenter._on_file_select([0])
    assert presenter._model.get_dataset_count() == 1
    assert presenter._selected_paths == presenter._visible_csvs == \
        [p for p in presenter._available_csvs if p.name == "a.csv"]


def test_build_filter_drops_hidden_loaded_file(tmp_path):
    """Mixed-build multi-select + filter: the hidden file leaves the model."""
    _create_meta_csv(tmp_path / "a.csv", "11028", "same.gguf")
    _create_meta_csv(tmp_path / "b.csv", "99999", "same.gguf")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path, show_md=False)
    names = window.left_sidebar.last_names
    presenter._on_file_select([names.index("a.csv"), names.index("b.csv")])
    assert presenter._model.get_dataset_count() == 2

    presenter._on_compat_filter('build', True)

    assert [p.name for p in presenter._visible_csvs] == ["a.csv"]
    assert presenter._selected_paths == [
        p for p in presenter._available_csvs if p.name == "a.csv"]
    assert presenter._model.get_dataset_count() == 1
    assert window.left_sidebar.last_paths == presenter._selected_paths


def test_scan_skips_files_without_valid_rows(tmp_path):
    """Header-valid files with zero parsable rows are not listed."""
    (tmp_path / "empty.csv").write_text(
        "n_prompt,n_gen,avg_ts,avg_ns\n0,0,100,50000")
    _create_meta_csv(tmp_path / "good.csv", "11028", "same.gguf")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path, show_md=False)

    assert [p.name for p in presenter._available_csvs] == ["good.csv"]


def test_scan_prunes_deleted_selection(tmp_path):
    """Files deleted from disk leave the selection and the model."""
    csv_a = _create_meta_csv(tmp_path / "a.csv", "11028", "same.gguf")
    _create_meta_csv(tmp_path / "b.csv", "11028", "same.gguf")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path, show_md=False)
    names = window.left_sidebar.last_names
    presenter._on_file_select([names.index("a.csv"), names.index("b.csv")])
    assert presenter._model.get_dataset_count() == 2

    csv_a.unlink()
    presenter.scan_files()

    assert [p.name for p in presenter._available_csvs] == ["b.csv"]
    assert presenter._selected_paths == [
        p for p in presenter._available_csvs if p.name == "b.csv"]
    assert presenter._model.get_dataset_count() == 1


def _mock_3d_ax(elev: float = 30.0, azim: float = -45.0) -> MagicMock:
    """Axes3D stand-in with usable camera state (no real 3D required)."""
    ax = MagicMock()
    ax.elev = elev
    ax.azim = azim
    ax.get_xlim3d.return_value = (0, 10)
    ax.get_ylim3d.return_value = (0, 20)
    ax.get_zlim3d.return_value = (0, 120)
    return ax


def test_3d_norm_toggle_restores_rotation_only(tmp_path):
    """A Norm toggle re-records home and must not reapply the stale zlim."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path, show_md=False)
    presenter._available_csvs = [csv1]
    presenter._visible_csvs = [csv1]
    presenter._on_file_select([0])

    window._mode_3d = 1
    window._axis_x = "params"
    window._axis_y = "n_gpu_layers"
    window._normalize = 0

    mock_fig = MagicMock()
    ax1, ax2, ax3 = _mock_3d_ax(), _mock_3d_ax(elev=60.0), _mock_3d_ax()

    with patch('presenter.plotter_presenter.render_3d',
               side_effect=[(mock_fig, ax1), (mock_fig, ax2), (mock_fig, ax3)]):
        with patch.object(window.plot_view, 'render'):
            presenter._render_plot()  # absolute scale
            home_abs = presenter._home_cam_3d
            assert home_abs['zlim'] == (0, 120)

            window._normalize = 1
            presenter._render_plot()  # normalized scale
            assert presenter._home_cam_3d is not home_abs
            assert presenter._home_cam_3d['elev'] == 60.0
            # Rotation carried over, stale absolute zlim NOT reapplied
            ax2.view_init.assert_called_once_with(elev=30.0, azim=-45.0)
            ax2.set_zlim3d.assert_not_called()
            ax2.set_xlim3d.assert_not_called()

            presenter._render_plot()  # same scale → full camera restore
            ax3.view_init.assert_called_once_with(elev=60.0, azim=-45.0)
            ax3.set_zlim3d.assert_called_once()


def test_3d_render_passes_interp_method(tmp_path):
    """The toolbar interpolation choice reaches render_3d."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path, show_md=False)
    presenter._available_csvs = [csv1]
    presenter._visible_csvs = [csv1]
    presenter._on_file_select([0])

    window._mode_3d = 1
    window._axis_x = "params"
    window._axis_y = "n_gpu_layers"
    window._interp_method = "Linear"

    with patch('presenter.plotter_presenter.render_3d',
               return_value=(MagicMock(), _mock_3d_ax())) as mock_render:
        with patch.object(window.plot_view, 'render'):
            presenter._render_plot()
            _, kwargs = mock_render.call_args
            assert kwargs.get('interp_method') == "Linear"


def test_3d_render_passes_clamp_and_mask(tmp_path):
    """The '+Clamp' suffix and Mask checkbox reach render_3d parsed."""
    csv1 = create_bench_csv(tmp_path / "bench.csv")

    window = MockMainWindow()
    presenter = PlotterPresenter(window, tmp_path, show_md=False)
    presenter._available_csvs = [csv1]
    presenter._visible_csvs = [csv1]
    presenter._on_file_select([0])

    window._mode_3d = 1
    window._axis_x = "params"
    window._axis_y = "n_gpu_layers"
    window._interp_method = "Cubic+Clamp"
    window._mask_gaps = True

    with patch('presenter.plotter_presenter.render_3d',
               return_value=(MagicMock(), _mock_3d_ax())) as mock_render:
        with patch.object(window.plot_view, 'render'):
            presenter._render_plot()
            _, kwargs = mock_render.call_args
            assert kwargs.get('interp_method') == "Cubic"
            assert kwargs.get('clamp_surface') is True
            assert kwargs.get('mask_gaps') is True
