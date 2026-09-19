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

import math
import tkinter as tk
from pathlib import Path
from typing import Optional

from model.benchmark_model import BenchmarkModel
from utils.colors import DEFAULT_PP_COLOR, DEFAULT_TG_COLOR
from utils.csv_parser import get_bench_file_meta, parse_bench_file
from view.main_window import MainWindow
from view.plot_view import (
    average_bucket,
    build_2d_title,
    build_3d_title,
    interp_surface_z,
    thin_value_ticks,
    render_2d,
    render_3d,
)

# Tooltip appearance/behavior (shared by the 2-D and 3-D pick handlers)
TOOLTIP_ALPHA = 0.85
TOOLTIP_BORDER = '#888888'
TOOLTIP_TTL_MS = 6000
TOOLTIP_SEPARATOR = "─" * 26


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

        # Tooltip context (axis names + per-point payloads)
        self._last_2d: dict = {}
        self._last_3d: dict = {}
        self._active_tip = None
        self._connector_line = None

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
        self._win.set_graph_title("")
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
        text = "t/s → time" if self._show_ts else "time → t/s"
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
            self._win.set_graph_title("")
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
            self._win.set_graph_title("")
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
        self._last_2d = {'series': series_data, 'x_param': x_param}

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
            show_ts=self._show_ts,
            x_ticks=thin_value_ticks(
                [p['x'] for p in series_data['pp']
                 if show_pp_flags[p['file_idx']]]
                + [p['x'] for p in series_data['tg']
                   if show_tg_flags[p['file_idx']]]
            ),
        )

        self._current_3d_ax = None
        self._win.set_graph_title(
            build_2d_title(x_param, self._win.unify, normalize))
        self._win.plot_view.render(fig, ax3d=None, on_pick_cb=self._on_pick)

    def _render_3d(
        self, x_param: str, normalize: bool, scale_pct: bool,
        show_pp: bool, show_tg: bool
    ) -> None:
        y_param = self._win.axis_y
        if not x_param or not y_param or x_param == y_param:
            self._win.set_graph_title("")
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

        points_pp, points_tg, stats, infos = self._model.get_3d_points(
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
            infos_pp=infos['pp'],
            infos_tg=infos['tg'],
        )

        self._current_3d_ax = ax
        self._last_3d = {'infos': infos, 'x_param': x_param, 'y_param': y_param,
                         'points': {'pp': points_pp, 'tg': points_tg}}
        self._connector_line = None  # new canvas drops the old connector
        self._win.set_graph_title(build_3d_title(x_param, y_param))
        self._win.plot_view.render(fig, ax3d=ax, on_pick_cb=self._on_pick_3d)

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

    # ── Pick events (2-D/3-D tooltips) ─────────────────────────────────────

    @staticmethod
    def _fmt_axis_value(value) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, str):
            return value
        return f"{value:g}"

    @staticmethod
    def _fmt_uncertain(value, err, sig=2, grouping=False) -> tuple:
        """
        Value ± error strings with precision derived from the error.

        The error is rounded to *sig* significant digits and the value
        to the same decimal position, so the value never claims more
        precision than its uncertainty supports (and the error never
        carries a meaningless digit tail).
        """
        if value is None:
            return "n/a", None
        if not err or not math.isfinite(err) or err <= 0 \
                or not math.isfinite(value):
            plain = f"{value:,.0f}" if grouping else f"{value:g}"
            return plain, None
        exp = math.floor(math.log10(abs(err))) - sig + 1
        if exp >= 0:
            if grouping:
                return (f"{round(value, -exp):,.0f}",
                        f"{round(err, -exp):,.0f}")
            return f"{round(value, -exp):.0f}", f"{round(err, -exp):.0f}"
        return (f"{round(value, -exp):.{-exp}f}",
                f"{round(err, -exp):.{-exp}f}")

    @staticmethod
    def _ts_parts(record: dict) -> tuple:
        """(value, error) strings for t/s; error None when absent."""
        return PlotterPresenter._fmt_uncertain(
            record.get('ts'), record.get('ts_err'))

    @staticmethod
    def _ns_unit(value: float) -> tuple:
        """(factor, unit) scaling nanoseconds to a readable unit."""
        magnitude = abs(value)
        if magnitude >= 1e9:
            return 1e-9, "s"
        if magnitude >= 1e6:
            return 1e-6, "ms"
        if magnitude >= 1e3:
            return 1e-3, "µs"
        return 1.0, "ns"

    @staticmethod
    def _ns_parts(record: dict) -> tuple:
        """(value, error) strings for latency, auto-scaled ns/µs/ms/s."""
        value, err = record.get('ns'), record.get('ns_err')
        if value is None:
            return "n/a", None
        factor, unit = PlotterPresenter._ns_unit(value)
        val_str, err_str = PlotterPresenter._fmt_uncertain(
            value * factor, err * factor if err else 0.0)
        if err_str is None:
            return f"{val_str} {unit}", None
        return val_str, f"{err_str} {unit}"

    @staticmethod
    def _metric_rows(record: dict) -> list:
        """Shared (name, value, error) rows for t/s and ns."""
        ts_v, ts_e = PlotterPresenter._ts_parts(record)
        ns_v, ns_e = PlotterPresenter._ns_parts(record)
        return [("t/s:", ts_v, ts_e), ("time:", ns_v, ns_e)]

    @staticmethod
    def _tooltip_table(blocks: list) -> str:
        """
        Table-aligned tooltip text (monospace assumed).

        *blocks* are {'header': str|None, 'rows': [(name, value,
        error|None), ...], 'tag': str|None} dicts. Names share one
        left-aligned column, values one right-aligned column, and
        errors one right-aligned column after "±", across all blocks —
        so ± signs sit exactly below each other.
        """
        names = [n for b in blocks for n, _, _ in b['rows']]
        vals = [v for b in blocks for _, v, _ in b['rows']]
        errs = [e for b in blocks for _, _, e in b['rows'] if e is not None]
        name_w = max([len(n) for n in names] or [0])
        val_w = max([len(v) for v in vals] or [0])
        err_w = max([len(e) for e in errs] or [0])
        out = []
        for i, block in enumerate(blocks):
            if i > 0:
                out.append(TOOLTIP_SEPARATOR)
            if block.get('header'):
                out.append(block['header'])
            for name, value, err in block['rows']:
                line = f"{name:<{name_w}}  {value:>{val_w}}"
                if err is not None:
                    line += f"  ± {err:>{err_w}}"
                out.append(line)
            if block.get('tag'):
                out.append(block['tag'])
        return "\n".join(out)

    @staticmethod
    def _record_block(first, coord_rows: list, record: dict) -> dict:
        """One tooltip block: header, coordinate rows, metric rows."""
        rows = list(coord_rows) + PlotterPresenter._metric_rows(record)
        tag = None
        if isinstance(first, str) and "Unified" in first:
            tag = "🔗 Combined"
        return {'header': first, 'rows': rows, 'tag': tag}

    @staticmethod
    def _clean_label(label) -> Optional[str]:
        """Usable label line or None for missing/default labels."""
        if not isinstance(label, str) or not label or label.startswith("_"):
            return None
        return label

    def _counterpart_label_2d(self, series: str, file_idx) -> str:
        """Counterpart label matching render_2d's naming."""
        if file_idx is None:
            return "Unified PP" if series == 'pp' else "Unified TG"
        try:
            stem = Path(
                self._model.get_datasets_raw()[file_idx]['path']).stem
        except (IndexError, KeyError, TypeError):
            stem = "?"
        return f"PP: {stem}" if series == 'pp' else f"TG: {stem}"

    def _counterpart_visible_2d(self, series: str, file_idx) -> bool:
        """Whether the counterpart series is currently displayed."""
        ls = self._win.left_sidebar
        if series == 'pp':
            per_file = ls.get_pp_flag(file_idx) if file_idx is not None else True
            return bool(self._win.show_pp and per_file)
        per_file = ls.get_tg_flag(file_idx) if file_idx is not None else True
        return bool(self._win.show_tg and per_file)

    def _find_counterpart_2d(self, rec: dict):
        """
        Counterpart point of the other series at the same x value.

        Returns (label, record) or None. Per-file points match within
        the same file; unified lines (no file_idx) average the other
        series across all *displayed* files with the same math the
        unified line itself uses. Only displayed series qualify, and
        only on exact x match — no nearest-point guessing.
        """
        series = rec.get('series')
        other = 'tg' if series == 'pp' else 'pp' if series == 'tg' else None
        if other is None:
            return None
        file_idx = rec.get('file_idx')
        cands = [p for p in (self._last_2d.get('series') or {}).get(other, [])
                 if p.get('x') == rec.get('x')]
        if file_idx is not None:
            cands = [p for p in cands if p.get('file_idx') == file_idx]
            if not cands or not self._counterpart_visible_2d(other, file_idx):
                return None
            return self._counterpart_label_2d(other, file_idx), cands[0]
        # Unified: average across displayed files only (mirrors the line).
        visible = [p for p in cands
                   if self._counterpart_visible_2d(other, p.get('file_idx'))]
        if not visible:
            return None
        _, _, avg_rec = average_bucket(rec.get('x'), visible)
        avg_rec['label'] = self._counterpart_label_2d(other, None)
        avg_rec['series'] = other
        return avg_rec['label'], avg_rec

    def _find_counterpart_3d(self, label: str, ind: int):
        """
        Counterpart info of the other series at the same (x, y).

        Returns (label, info) or None when the other series is hidden
        or has no point at exactly that position. Points carry no file
        identity, so with several loaded files sharing an (x, y) the
        first hit wins even across files — accepted: combined 3-D
        tooltips compare positions, not file provenance.
        """
        series = 'pp' if label == 'PP' else 'tg' if label == 'TG' else None
        if series is None:
            return None
        other = 'tg' if series == 'pp' else 'pp'
        if other == 'pp' and not self._win.show_pp:
            return None
        if other == 'tg' and not self._win.show_tg:
            return None
        points = self._last_3d.get('points', {}) or {}
        infos = self._last_3d.get('infos', {}) or {}
        own = points.get(series) or []
        if not (0 <= ind < len(own)):
            return None
        x, y = own[ind][0], own[ind][1]
        others = points.get(other) or []
        other_infos = infos.get(other) or []
        for j, q in enumerate(others):
            if q[0] == x and q[1] == y and j < len(other_infos):
                return ('PP' if other == 'pp' else 'TG'), other_infos[j]
        return None

    def _show_tooltip(self, text: str) -> None:
        # One tooltip at a time: drop the previous one so rapid picks
        # do not stack overlapping windows.
        old_tip = self._active_tip
        if old_tip is not None:
            try:
                old_tip.destroy()
            except Exception:
                pass
            self._active_tip = None
        tip = tk.Toplevel(self._win.root)
        tip.wm_overrideredirect(True)
        # Anchor at the real pointer position: mouseevent coordinates
        # are canvas-relative, which parked old tooltips far off target.
        try:
            px, py = tip.winfo_pointerx() + 16, tip.winfo_pointery() + 12
        except Exception:
            px, py = 100, 100
        tip.geometry(f"+{px}+{py}")
        tip.configure(bg='#1e1e1e', bd=1, relief='solid',
                      highlightbackground=TOOLTIP_BORDER,
                      highlightthickness=1)
        try:
            tip.attributes('-alpha', TOOLTIP_ALPHA)
        except Exception:
            pass  # translucency unsupported on this platform

        tk.Label(
            tip, text=text,
            bg='#1e1e1e', fg='#d4d4d4',
            font=('Consolas', 9), padx=5, pady=3,
        ).pack()
        self._active_tip = tip
        tip.after(TOOLTIP_TTL_MS, lambda: self._hide_tooltip(tip))

    def _hide_tooltip(self, tip) -> None:
        """Deferred tooltip cleanup that survives app teardown."""
        try:
            tip.destroy()
        except Exception:
            pass
        if self._active_tip is tip:
            self._active_tip = None

    def _on_pick(self, event) -> None:
        """2-D tooltip: axis name, both metrics with errors."""
        if event is None:
            return  # empty click: nothing transient in 2-D
        if not hasattr(event, 'ind') or len(event.ind) == 0:
            return
        ind = event.ind[0]
        label = event.artist.get_label()
        records = getattr(event.artist, '_llama_records', None)

        if isinstance(records, list) and 0 <= ind < len(records):
            rec = records[ind]
            # Series label rides in the record: errorbar() keeps it on
            # the container (for the legend) while the pickable data
            # line itself stays at the default "_no_legend_".
            first = self._clean_label(rec.get('label') or label)
            x_title = (self._last_2d.get('x_param') or 'x').replace('_', ' ').title()
            coord = [(x_title + ":", self._fmt_axis_value(rec.get('x')), None)]
            blocks = [self._record_block(first, coord, rec)]
            counter = self._find_counterpart_2d(rec)
            if counter is not None:
                clabel, crec = counter
                ccoord = [(x_title + ":", self._fmt_axis_value(crec.get('x')), None)]
                blocks.append(self._record_block(clabel, ccoord, crec))
            self._show_tooltip(self._tooltip_table(blocks))
            return

        # Fallback for artists without attached records (e.g. a stray
        # whisker pick): unnamed artists show values without a label line.
        xdata = event.artist.get_xdata()
        ydata = event.artist.get_ydata()
        x_title = (self._last_2d.get('x_param') or 'x').replace('_', ' ').title()
        fblock = {'header': self._clean_label(label),
                  'rows': [(x_title + ":",
                            self._fmt_axis_value(xdata[ind]), None),
                           ("Y:", f"{ydata[ind]:.2f}", None)],
                  'tag': None}
        if fblock['header'] is not None and "Unified" in fblock['header']:
            fblock['tag'] = "🔗 Combined"
        self._show_tooltip(self._tooltip_table([fblock]))

    def _on_pick_3d(self, event) -> None:
        """3-D tooltip: both axis names/values plus both metrics."""
        if event is None:
            if self._clear_connector():
                self._win.plot_view.redraw_idle()
            return
        if not hasattr(event, 'ind') or len(event.ind) == 0:
            return
        ind = event.ind[0]
        label = event.artist.get_label()
        if label not in ("PP", "TG"):
            return
        records = getattr(event.artist, '_llama_records', None)
        if isinstance(records, list) and 0 <= ind < len(records):
            info = records[ind]
        else:
            infos = self._last_3d.get('infos', {}).get(
                'pp' if label == 'PP' else 'tg')
            if not isinstance(infos, list) or not (0 <= ind < len(infos)):
                return
            info = infos[ind]
        x_title = (self._last_3d.get('x_param') or 'x').replace('_', ' ').title()
        y_title = (self._last_3d.get('y_param') or 'y').replace('_', ' ').title()

        def block(slave_label, slave_info):
            return self._record_block(
                slave_label,
                [(x_title + ":", self._fmt_axis_value(slave_info.get('x')), None),
                 (y_title + ":", self._fmt_axis_value(slave_info.get('y')), None)],
                slave_info)

        # PP block always on top, TG below; single block without match.
        counter = self._find_counterpart_3d(label, ind)
        if counter is not None and label == 'TG':
            blocks = [block(*counter), block(label, info)]
        elif counter is not None:
            blocks = [block(label, info), block(*counter)]
        else:
            blocks = [block(label, info)]
        self._show_tooltip(self._tooltip_table(blocks))
        self._update_connector(label, ind)

    def _update_connector(self, label: str, ind: int) -> None:
        """
        Vertical dashed line from the picked point to the other surface.

        The counterpart height is linearly interpolated on the other
        series at the picked (x, y); without coverage there (outside
        its hull) no line is drawn. Exactly one connector exists at a
        time — each new pick moves it. Never raises.
        """
        removed = self._clear_connector()
        drawn = False
        try:
            series = 'pp' if label == 'PP' else 'tg'
            all_points = self._last_3d.get('points', {}) or {}
            pts = all_points.get(series) or []
            other = all_points.get('tg' if series == 'pp' else 'pp') or []
            if not (0 <= ind < len(pts)) or not other:
                return
            x, y, z = pts[ind][0], pts[ind][1], pts[ind][2]
            z_other = interp_surface_z(
                [p[0] for p in other], [p[1] for p in other],
                [p[2] for p in other], x, y)
            if z_other is None:
                return
            ax = self._current_3d_ax
            lines = ax.plot([x, x], [y, y], [z, z_other],
                            color='#cccccc', linestyle='--', linewidth=1.5)
            self._connector_line = lines[0] if lines else None
            drawn = self._connector_line is not None
        except Exception as exc:
            print(f"[Presenter] Connector warning: {exc}")
        finally:
            if removed or drawn:
                self._win.plot_view.redraw_idle()

    def _clear_connector(self) -> bool:
        """Remove the current connector line; True when one existed."""
        line, self._connector_line = self._connector_line, None
        if line is None:
            return False
        try:
            line.remove()
        except Exception:
            pass
        return True
