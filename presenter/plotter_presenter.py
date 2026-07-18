"""
presenter/plotter_presenter.py

PlotterPresenter — the single "smart" controller in llamagraph's MVP design.

Responsibilities:
  - Hold references to the Model and all View instances
  - Wire every callback between View events and Model/View actions
  - Own the application state: show_ts (metric), sort order, 3-D camera
  - Drive the right sidebar filter updates whenever axes or data change
  - Invoke the correct plot-engine function and hand the Figure to PlotView

The Presenter is the only place that imports from both model/ and view/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from model.benchmark_model import BenchmarkModel
from utils.colors import DEFAULT_PP_COLOR, DEFAULT_TG_COLOR
from utils.csv_parser import is_llama_bench_csv
from view.main_window import MainWindow
from view.plot_view import render_2d, render_3d


class PlotterPresenter:
    """
    Wires Model - View and drives all application logic.

    Lifecycle:
      1. __init__ receives a configured MainWindow and a search directory.
      2. _wire_callbacks() connects every UI event to a handler here.
      3. scan_files() populates the file list on first launch.
    """

    def __init__(
        self,
        window: MainWindow,
        start_dir: Path,
        pp_color: str = DEFAULT_PP_COLOR,
        tg_color: str = DEFAULT_TG_COLOR,
        default_ts: bool = True,
        initial_selection_file: Optional[Path] = None,
    ) -> None:
        self._win = window
        self._model = BenchmarkModel()
        self._start_dir = start_dir
        self._pp_color = pp_color
        self._tg_color = tg_color

        # Application state
        self._show_ts: bool = default_ts          # True = tokens/s, False = ns
        self._sort_by_time: bool = True           # True = mtime desc, False = name
        self._available_csvs: list[Path] = []     # All valid CSV paths in dir
        self._current_selection: list[int] = []   # Currently selected indices

        # 3-D camera persistence
        self._cam_3d: Optional[dict] = None
        self._home_cam_3d: Optional[dict] = None
        self._current_3d_ax = None
        self._last_3d_signature: Optional[tuple] = None

        # Connect everything
        self._wire_callbacks()
        self.scan_files()
        self._update_metric_button()

        # Handle initial file selection from CLI
        if initial_selection_file:
            self._handle_initial_selection(initial_selection_file)

    def _handle_initial_selection(self, file_path: Path) -> None:
        """Finds the index of the provided file and selects it."""
        try:
            idx = self._available_csvs.index(file_path)
            self._win.left_sidebar.select_index(idx)
            self._on_file_select([idx])
        except ValueError:
            # File not in the scanned list, ignore silently as requested
            pass

    def _wire_callbacks(self) -> None:
        """Inject all callback references into View components."""
        win = self._win
        ls = win.left_sidebar
        rs = win.right_sidebar
        pv = win.plot_view

        # Left sidebar
        ls.set_file_select_callback(self._on_file_select)
        ls.set_sort_callback(self._on_sort)
        ls.set_refresh_callback(self.scan_files)
        ls.set_series_toggle_callback(self._render_plot)
        ls.set_select_all_callback(self._on_select_all)
        ls.set_deselect_all_callback(self._on_deselect_all)
        ls.set_choose_directory_callback(self._on_choose_directory)

        # Right sidebar (dimension filters)
        rs.set_filter_change_callback(self._on_filter_change)

        # Main window toolbar callbacks
        win.set_render_callback(self._render_plot)
        win.set_toggle_3d_callback(self._on_toggle_3d)
        win.set_toggle_metric_callback(self.toggle_metric)

        # Plot view home-button override
        pv.set_home_callback(self._on_home_3d)

        # Key bindings
        win.set_key_bindings(
            toggle_metric=self.toggle_metric,
            refresh=self.scan_files,
            quit_app=win.root.quit,
        )

    # ── File management ───────────────────────────────────────────────────────

    def _on_choose_directory(self, chosen_path) -> None:
        """
        Called when the user picks a directory via the Browse button.
        Updates the working directory and rescans for CSV files.
        """
        self._start_dir = chosen_path
        self._win.left_sidebar.set_directory_label(chosen_path)
        # Clear current selection and model data before rescanning
        self._current_selection = []
        self._model.clear()
        self._win.left_sidebar.update_series_toggles([])
        self._win.plot_view.show_placeholder(
            "📊 Select CSV file(s) with Ctrl+Click to display"
        )
        self.scan_files()

    def scan_files(self) -> None:
        """Scan *start_dir* for valid llama-bench CSV files and populate the list."""
        self._available_csvs = []
        files = list(self._start_dir.glob("*.csv"))

        if self._sort_by_time:
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            sort_label = "Sort: Time ↓"
        else:
            files.sort(key=lambda f: f.name)
            sort_label = "Sort: Name A-Z"

        for f in files:
            if is_llama_bench_csv(f):
                self._available_csvs.append(f)

        names = [f.name for f in self._available_csvs]
        self._win.left_sidebar.populate_file_list(names, sort_label)
        self._win.left_sidebar.set_directory_label(self._start_dir)

    def _on_sort(self) -> None:
        self._sort_by_time = not self._sort_by_time
        self.scan_files()

    def _on_select_all(self) -> None:
        self._win.left_sidebar.select_all()

    def _on_deselect_all(self) -> None:
        self._win.left_sidebar.deselect_all()

    # ── File selection → Model load ───────────────────────────────────────────

    def _on_file_select(self, selected_indices: list[int]) -> None:
        """Called by LeftSidebar when the user changes the file selection."""
        self._current_selection = selected_indices

        paths = [self._available_csvs[i] for i in selected_indices
                 if i < len(self._available_csvs)]

        errors = self._model.load_files(paths)
        for err in errors:
            print(f"[Presenter] {err}")

        self._refresh_ui_after_load()

    def _refresh_ui_after_load(self) -> None:
        """Update axis comboboxes, series toggles, right sidebar after a load."""
        datasets = self._model.get_datasets_raw()
        paths = [ds['path'] for ds in datasets]

        # Rebuild left sidebar series toggles
        self._win.left_sidebar.update_series_toggles(paths)

        # Unify button availability
        self._win.set_unify_state(len(datasets) >= 2)

        # Update axis combobox options
        dims = self._model.get_dimensions()
        self._win.update_axis_choices(dims)

        # Rebuild right sidebar filter sections
        self._update_right_sidebar()

        # Redraw
        self._render_plot()

    # ── Filter change → Model → re-render ────────────────────────────────────

    def _on_filter_change(self, filter_dict: dict[str, set]) -> None:
        """Called by RightSidebar when the user changes a dimension filter."""
        self._model.apply_filters(filter_dict)
        self._render_plot()

    def _update_right_sidebar(self) -> None:
        """Tell the right sidebar which filter sections to show."""
        dim_values = self._model.get_all_dim_values()
        active_axes = self._get_active_axes()
        current_filters = self._model.get_filter_state()

        self._win.right_sidebar.update_filter_sections(
            dim_values, active_axes, current_filters
        )

    def _get_active_axes(self) -> set[str]:
        """Return the set of dimension names currently used as plot axes."""
        axes = set()
        x = self._win.axis_x
        y = self._win.axis_y
        if x:
            axes.add(x)
        if self._win.mode_3d and y:
            axes.add(y)
        return axes

    # ── 3-D mode toggle ───────────────────────────────────────────────────────

    def _on_toggle_3d(self) -> None:
        """Called when the user clicks the 3D checkbox."""
        self._update_right_sidebar()
        self._render_plot()

    # ── Metric toggle (t/s vs ns) ─────────────────────────────────────────────

    def toggle_metric(self) -> None:
        self._show_ts = not self._show_ts
        self._update_metric_button()
        if self._model.has_data():
            self._render_plot()

    def _update_metric_button(self) -> None:
        text = "Switch: ns" if self._show_ts else "Switch: t/s"
        self._win.set_metric_button_text(text)

    # ── 3-D camera ────────────────────────────────────────────────────────────

    def _save_camera(self, ax) -> dict:
        state: dict = {'elev': ax.elev, 'azim': ax.azim}
        if hasattr(ax, 'roll'):
            state['roll'] = ax.roll
        try:
            state['xlim'] = ax.get_xlim3d()
            state['ylim'] = ax.get_ylim3d()
            state['zlim'] = ax.get_zlim3d()
        except Exception:
            pass
        return state

    def _restore_camera(self, ax, state: dict) -> None:
        try:
            ax.view_init(elev=state['elev'], azim=state['azim'])
            if 'roll' in state and hasattr(ax, 'roll'):
                ax.roll = state['roll']
            if 'xlim' in state:
                ax.set_xlim3d(state['xlim'])
            if 'ylim' in state:
                ax.set_ylim3d(state['ylim'])
            if 'zlim' in state:
                ax.set_zlim3d(state['zlim'])
        except Exception as exc:
            print(f"[Presenter] Camera restore warning: {exc}")

    def _on_home_3d(self) -> bool:
        """
        Called by CustomNavigationToolbar.home().
        Returns True if we handled it (prevents Matplotlib default).
        """
        if self._win.mode_3d and self._current_3d_ax and self._home_cam_3d:
            self._restore_camera(self._current_3d_ax, self._home_cam_3d)
            self._win.plot_view.redraw_idle()
            return True
        return False

    # ── Core render ───────────────────────────────────────────────────────────

    def _render_plot(self) -> None:
        """
        Build and display the plot.  This is the central orchestration method.
        Called whenever any control changes.
        """
        if not self._model.has_data():
            self._win.plot_view.show_placeholder(
                "📊 Select CSV file(s) with Ctrl+Click to display"
            )
            return

        # Read all relevant state from the window
        x_param = self._win.axis_x
        normalize = self._win.normalize
        scale_pct = (self._win.z_label_mode == "%")
        show_pp = self._win.show_pp
        show_tg = self._win.show_tg

        if self._win.mode_3d:
            self._render_3d(x_param, normalize, scale_pct, show_pp, show_tg)
        else:
            self._render_2d(x_param, normalize, scale_pct, show_pp, show_tg)

        # Keep right sidebar in sync whenever axes may have changed
        self._update_right_sidebar()

    def _render_2d(
        self, x_param: str, normalize: bool, scale_pct: bool,
        show_pp: bool, show_tg: bool
    ) -> None:
        if not x_param:
            self._win.plot_view.show_placeholder("⚠ No parameter axis available.")
            return

        ls = self._win.left_sidebar
        n = self._model.get_dataset_count()
        show_pp_flags = [ls.get_pp_flag(i) and show_pp for i in range(n)]
        show_tg_flags = [ls.get_tg_flag(i) and show_tg for i in range(n)]

        # Pull aggregated data from model based on current filters and axis choices
        series_data = self._model.get_2d_series(
            x_dim=x_param,
            show_ts=self._show_ts,
            show_pp=show_pp,
            show_tg=show_tg,
            normalize=normalize,
            scale_pct=scale_pct,
        )

        fig = render_2d(
            datasets_raw=self._model.get_datasets_raw(),
            series_data=series_data,
            x_param=x_param,
            pp_base=self._pp_color,
            tg_base=self._tg_color,
            show_pp_flags=show_pp_flags,
            show_tg_flags=show_tg_flags,
            do_unify=self._win.unify,
            normalize=normalize,
            z_label_mode=self._win.z_label_mode,
        )

        self._current_3d_ax = None
        self._win.plot_view.render(fig, ax3d=None, on_pick_cb=self._on_pick)

    def _render_3d(
        self, x_param: str, normalize: bool, scale_pct: bool,
        show_pp: bool, show_tg: bool
    ) -> None:
        y_param = self._win.axis_y
        if not x_param or not y_param or x_param == y_param:
            self._win.plot_view.show_placeholder(
                "⚠ Select two different axes for 3D plot."
            )
            return

        # Detect axis change — reset camera on new axis combo
        sig = (x_param, y_param)
        if sig != self._last_3d_signature:
            self._cam_3d = None
            self._home_cam_3d = None
            self._last_3d_signature = sig

        # Save user camera before rebuilding canvas
        saved_cam: Optional[dict] = None
        if self._current_3d_ax is not None:
            saved_cam = self._save_camera(self._current_3d_ax)

        points_pp, points_tg = self._model.get_3d_points(
            x_dim=x_param,
            y_dim=y_param,
            show_ts=self._show_ts,
            show_pp=show_pp,
            show_tg=show_tg,
            normalize=normalize,
            scale_pct=scale_pct,
        )

        fig, ax = render_3d(
            points_pp=points_pp,
            points_tg=points_tg,
            x_param=x_param,
            y_param=y_param,
            pp_color=self._pp_color,
            tg_color=self._tg_color,
            z_label_mode=self._win.z_label_mode,
            show_surface=self._win.show_surface,
            show_wireframe=self._win.show_wireframe,
            show_projections=self._win.show_projections,
            show_errors_3d=self._win.show_errors_3d,
            show_level=self._win.show_level,
            level_val=self._win.level_val,
            surface_style=self._win.surface_style,
            subdiv_level=self._win.subdiv_level,
        )

        self._current_3d_ax = ax
        self._win.plot_view.render(fig, ax3d=ax)

        # Record home camera at first render for this axis combo
        if self._home_cam_3d is None:
            self._home_cam_3d = self._save_camera(ax)

        # Restore the user's last camera position
        if saved_cam is not None:
            self._restore_camera(ax, saved_cam)
            self._win.plot_view.redraw_idle()

    # ── Pick event (2-D tooltip) ──────────────────────────────────────────────

    def _on_pick(self, event) -> None:
        if not hasattr(event, 'ind') or len(event.ind) == 0:
            return
        ind = event.ind[0]
        label = event.artist.get_label()
        xdata = event.artist.get_xdata()
        ydata = event.artist.get_ydata()

        tip = __import__('tkinter').Toplevel(self._win.root)
        tip.wm_overrideredirect(True)
        mx = event.mouseevent.x + 20
        my = event.mouseevent.y + 20
        tip.geometry(f"+{mx}+{my}")
        tip.configure(bg='#1e1e1e', bd=1, relief='solid')

        txt = f"{label}\nX: {xdata[ind]}\nY: {ydata[ind]:.2f}"
        if "Unified" in label:
            txt += "\n🔗 Combined"

        __import__('tkinter').Label(
            tip, text=txt,
            bg='#1e1e1e', fg='#d4d4d4',
            font=('Consolas', 9), padx=5, pady=3,
        ).pack()
        tip.after(2000, tip.destroy)
