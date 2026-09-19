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
from utils.csv_parser import get_bench_file_meta, parse_bench_file
from view.main_window import MainWindow
from view.plot_view import thin_value_ticks, render_2d, render_3d


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
        show_md: bool = True,
    ) -> None:
        self._win = window
        self._model = BenchmarkModel()
        self._start_dir = start_dir
        self._pp_color = pp_color
        self._tg_color = tg_color

        # Application state
        self._show_ts: bool = default_ts          # True = tokens/s, False = ns
        self._sort_by_time: bool = True           # True = mtime desc, False = name
        self._md_allowed: bool = show_md          # False with --no-md: no .md ever
        self._show_md: bool = show_md             # .md toggle state (sidebar button)
        self._available_csvs: list[Path] = []     # All valid CSV/MD paths in dir
        self._visible_csvs: list[Path] = []       # Filtered subset shown in sidebar
        self._current_selection: list[int] = []   # Selected indices into _visible_csvs
        self._selected_paths: list[Path] = []     # Loaded paths (survive list rebuilds)

        # Comparison filter (same build / same model): reference values come
        # from the first selected file; flags are user-toggled in the sidebar.
        self._file_meta: dict[Path, dict] = {}
        self._ref_build: Optional[str] = None
        self._ref_build_commit: Optional[str] = None
        self._ref_model_key: Optional[str] = None
        self._ref_model_label: Optional[str] = None
        self._build_only: bool = False
        self._model_only: bool = False

        # 3-D camera persistence
        self._cam_3d: Optional[dict] = None
        self._home_cam_3d: Optional[dict] = None
        self._current_3d_ax = None
        self._last_3d_signature: Optional[tuple] = None

        # Connect everything
        self._wire_callbacks()
        self._win.left_sidebar.set_md_toggle_visible(self._md_allowed)
        self._win.left_sidebar.set_md_toggle_state(self._show_md)
        self.scan_files()
        self._update_metric_button()

        # Handle initial file selection from CLI
        if initial_selection_file:
            self._handle_initial_selection(initial_selection_file)

    def _handle_initial_selection(self, file_path: Path) -> None:
        """Finds the index of the provided file and selects it."""
        try:
            idx = self._visible_csvs.index(file_path)
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
        ls.set_md_toggle_callback(self._on_md_toggle)
        ls.set_choose_directory_callback(self._on_choose_directory)
        ls.set_compat_filter_callback(self._on_compat_filter)

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
        self._reset_compat_filter()
        self._clear_loaded_data()
        self.scan_files()

    def _reset_compat_filter(self) -> None:
        """Clear reference values and flags of the comparison filter."""
        self._ref_build = None
        self._ref_build_commit = None
        self._ref_model_key = None
        self._ref_model_label = None
        self._build_only = False
        self._model_only = False

    def _clear_loaded_data(self) -> None:
        """Clear current selection and model data (list indices become stale)."""
        self._current_selection = []
        self._selected_paths = []
        self._model.clear()
        self._win.left_sidebar.update_series_toggles([])
        self._win.plot_view.show_placeholder(
            "📊 Select CSV/MD file(s) with Ctrl+Click to display"
        )

    def _get_meta(self, path: Path) -> dict:
        """Cached file-level metadata (build/model) for the compat filter."""
        meta = self._file_meta.get(path)
        if meta is None:
            meta = get_bench_file_meta(path)
            self._file_meta[path] = meta
        return meta

    def _matches_compat_filter(self, path: Path) -> bool:
        """True if *path* passes the active build/model restrictions."""
        if self._build_only:
            if self._ref_build is None:
                return True  # no reference → cannot restrict
            if self._get_meta(path).get('build_number') != self._ref_build:
                return False
        if self._model_only:
            if self._ref_model_key is None:
                return True
            if self._get_meta(path).get('model_key') != self._ref_model_key:
                return False
        return True

    def _refresh_visible_files(self) -> None:
        """
        Recompute the filtered file list, repopulate the sidebar, restore the
        highlight for still-visible loaded paths, and update the filter box.
        Never touches the model — pure list/bookkeeping refresh.
        """
        self._visible_csvs = [
            f for f in self._available_csvs if self._matches_compat_filter(f)
        ]
        if self._sort_by_time:
            sort_label = "Sort: Time ↓"
        else:
            sort_label = "Sort: Name A-Z"
        names = [f.name for f in self._visible_csvs]
        self._win.left_sidebar.populate_file_list(names, sort_label)

        # Restore highlight for loaded paths that are still visible.
        # populate_file_list() cleared the widget selection; select_index()
        # is silent (fires no callback), so no model reload is triggered.
        self._current_selection = [
            i for i, f in enumerate(self._visible_csvs)
            if f in self._selected_paths
        ]
        for i in self._current_selection:
            try:
                self._win.left_sidebar.select_index(i)
            except Exception as exc:
                print(f"[Presenter] Reselect warning: {exc}")

        self._update_compat_box()

    def _update_compat_box(self) -> None:
        """Show/hide the sidebar comparison-filter box from current state."""
        update = self._win.left_sidebar.update_compat_filter
        if not self._selected_paths:
            update(None, None)
            return
        if self._ref_build is not None:
            build_text = f"Build {self._ref_build}"
            if self._ref_build_commit:
                build_text += f" ({self._ref_build_commit})"
        else:
            build_text = None
        update(build_text, self._ref_model_label,
               self._build_only, self._model_only)

    def _update_compat_reference(self) -> None:
        """
        Point the comparison filter at the first selected file.
        Keeps the user's on/off flags; drops a flag whose reference vanished.
        """
        if not self._selected_paths:
            self._reset_compat_filter()
            return
        meta = self._get_meta(self._selected_paths[0])
        self._ref_build = meta.get('build_number')
        self._ref_build_commit = meta.get('build_commit')
        self._ref_model_key = meta.get('model_key')
        self._ref_model_label = meta.get('model_label')
        if self._ref_build is None:
            self._build_only = False
        if self._ref_model_key is None:
            self._model_only = False

    def _on_compat_filter(self, kind: str, active: bool) -> None:
        """Called by the sidebar when a comparison checkbox is toggled."""
        if kind == 'build':
            # Only lock to a build when we actually know the reference.
            self._build_only = bool(active) and self._ref_build is not None
        elif kind == 'model':
            self._model_only = bool(active) and self._ref_model_key is not None
        else:
            return
        if self._build_only or self._model_only:
            kept = [p for p in self._selected_paths
                    if self._matches_compat_filter(p)]
            if len(kept) != len(self._selected_paths):
                # A loaded file is now hidden by the filter: drop it from
                # the model so list, highlight, and plot stay consistent.
                # (The first selected file defines the reference, so `kept`
                # is never empty here.)
                self._selected_paths = kept
                self._reload_selection()
        self._refresh_visible_files()

    def _reload_selection(self) -> None:
        """(Re)load the model from _selected_paths and refresh dependent UI."""
        errors = self._model.load_files(list(self._selected_paths))
        for err in errors:
            print(f"[Presenter] {err}")
        self._refresh_ui_after_load()

    def _find_bench_files(self) -> list[Path]:
        """Collect candidate benchmark files (side-effect free, for scan + compare)."""
        files = list(self._start_dir.glob("*.csv"))
        if self._md_allowed and self._show_md:
            files += list(self._start_dir.glob("*.md"))
            files += list(self._start_dir.glob("*.markdown"))
        return files

    def scan_files(self) -> None:
        """
        Scan *start_dir* for valid llama-bench files (CSV or MD) and populate
        the list. Each candidate is parsed once; files that fail to parse are
        skipped (so unlike a header-only probe, files with zero valid rows
        are not listed).
        """
        self._available_csvs = []
        self._file_meta = {}
        files = self._find_bench_files()

        if self._sort_by_time:
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        else:
            files.sort(key=lambda f: f.name)

        for f in files:
            parsed = parse_bench_file(f)
            if parsed is None:
                continue
            self._available_csvs.append(f)
            meta = parsed.get('file_meta')
            self._file_meta[f] = meta if isinstance(meta, dict) else {}

        if self._selected_paths:
            # Drop loaded paths that no longer exist (or parse) on disk and
            # reload so the model never holds data for unlisted files.
            still_here = set(self._available_csvs)
            kept = [p for p in self._selected_paths if p in still_here]
            if len(kept) != len(self._selected_paths):
                self._selected_paths = kept
                self._update_compat_reference()
                self._reload_selection()

        self._refresh_visible_files()
        self._win.left_sidebar.set_directory_label(self._start_dir)

    def _on_sort(self) -> None:
        self._sort_by_time = not self._sort_by_time
        self.scan_files()

    def _on_md_toggle(self) -> None:
        """Show/hide `.md` files in the sidebar (unavailable with `--no-md`)."""
        if not self._md_allowed:
            return
        self._show_md = not self._show_md
        self._win.left_sidebar.set_md_toggle_state(self._show_md)
        before = {f.name for f in self._available_csvs}
        self.scan_files()
        if {f.name for f in self._available_csvs} != before:
            # List indices shift when .md files appear/disappear → reset
            # (skipped when the visible list is unchanged, e.g. no .md files).
            # No re-scan needed: the list above is already up to date.
            self._reset_compat_filter()
            self._clear_loaded_data()
            self._refresh_visible_files()

    def _on_select_all(self) -> None:
        self._win.left_sidebar.select_all()

    def _on_deselect_all(self) -> None:
        self._win.left_sidebar.deselect_all()

    # ── File selection → Model load ───────────────────────────────────────────

    def _on_file_select(self, selected_indices: list[int]) -> None:
        """Called by LeftSidebar when the user changes the file selection."""
        valid = [i for i in selected_indices if i < len(self._visible_csvs)]
        paths = [self._visible_csvs[i] for i in valid]
        self._current_selection = valid
        self._selected_paths = list(paths)

        if not paths:
            # Last file deselected → filters reset, full list returns.
            self._reset_compat_filter()
            errors = self._model.load_files([])
            for err in errors:
                print(f"[Presenter] {err}")
            self._refresh_ui_after_load()
            self._refresh_visible_files()
            return

        errors = self._model.load_files(paths)
        for err in errors:
            print(f"[Presenter] {err}")

        self._update_compat_reference()
        self._refresh_ui_after_load()
        if self._build_only or self._model_only:
            # Narrow the list to same-build/same-model files; the just-loaded
            # paths always match their own reference, so they stay visible.
            # (Select All then naturally covers only the filtered files.)
            self._refresh_visible_files()
        else:
            self._update_compat_box()

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

    def _restore_rotation(self, ax, state: dict) -> None:
        """Restore only the viewing angle (elev/azim/roll), not the limits."""
        try:
            ax.view_init(elev=state['elev'], azim=state['azim'])
            if 'roll' in state and hasattr(ax, 'roll'):
                ax.roll = state['roll']
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
                "📊 Select CSV/MD file(s) with Ctrl+Click to display"
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
            x_ticks=thin_value_ticks(
                [p['x'] for p in series_data['pp']
                 if show_pp_flags[p['file_idx']]]
                + [p['x'] for p in series_data['tg']
                   if show_tg_flags[p['file_idx']]]
            ),
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

        # Detect scale-relevant changes — a new axis combo, metric,
        # normalization, series visibility, interpolation, or gap mask
        # means the old axis limits (especially zlim) no longer fit the
        # data. The home view is then re-recorded and only the viewing
        # angle is carried over, so a stale zlim can never squash or blow
        # out rescaled surfaces. Pure rotations and dimension-filter
        # changes keep the full camera.
        interp_raw = self._win.interp_method
        mask_gaps = self._win.mask_gaps
        clamp_surface = interp_raw.endswith("+Clamp")
        interp_method = interp_raw[:-len("+Clamp")] if clamp_surface else interp_raw
        sig = (x_param, y_param, normalize, scale_pct,
               self._show_ts, show_pp, show_tg, interp_raw, mask_gaps,
               self._win.subdiv_level)
        scale_changed = (sig != self._last_3d_signature)
        if scale_changed:
            self._cam_3d = None
            self._home_cam_3d = None
            self._last_3d_signature = sig

        # Save user camera before rebuilding canvas
        saved_cam: Optional[dict] = None
        if self._current_3d_ax is not None:
            saved_cam = self._save_camera(self._current_3d_ax)

        points_pp, points_tg, stats = self._model.get_3d_points(
            x_dim=x_param,
            y_dim=y_param,
            show_ts=self._show_ts,
            show_pp=show_pp,
            show_tg=show_tg,
            normalize=normalize,
            scale_pct=scale_pct,
        )

        # Measured value ticks, unless a dimension is string-valued
        # (then the categorical labels below take precedence).
        x_labels = self._model.get_dim_labels(x_param)
        y_labels = self._model.get_dim_labels(y_param)
        x_ticks = None if x_labels is not None else thin_value_ticks(
            [p[0] for p in points_pp] + [p[0] for p in points_tg])
        y_ticks = None if y_labels is not None else thin_value_ticks(
            [p[1] for p in points_pp] + [p[1] for p in points_tg])

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
            interp_method=interp_method,
            clamp_surface=clamp_surface,
            mask_gaps=mask_gaps,
            normalized=normalize,
            pp_min=stats['pp_min'],
            pp_max=stats['pp_max'],
            tg_min=stats['tg_min'],
            tg_max=stats['tg_max'],
            x_ticks=x_ticks,
            y_ticks=y_ticks,
        )

        self._current_3d_ax = ax
        self._win.plot_view.render(fig, ax3d=ax)

        # Apply categorical tick labels if dimensions are string-valued
        if x_labels is not None:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels)
        if y_labels is not None:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)

        # Record home camera at first render for this scale signature
        if self._home_cam_3d is None:
            self._home_cam_3d = self._save_camera(ax)

        # Restore the user's last camera position (angle only when the
        # data scale changed, full camera otherwise)
        if saved_cam is not None:
            if scale_changed:
                self._restore_rotation(ax, saved_cam)
            else:
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
