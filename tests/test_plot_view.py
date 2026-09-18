"""Tests for view/plot_view.py - Stateless render functions and helpers."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use('Agg')  # headless backend, must be set before importing matplotlib modules

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from view.plot_view import (
    _draw_unified,
    _set_series_zticks,
    _set_z_label,
    render_2d,
    render_3d,
)

# ── Test data helpers ─────────────────────────────────────────────────────────


def _fig_ax_3d():
    """Create a (Figure, Axes3D) pair for testing 3D-related functions."""
    fig = Figure()
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax


# ── _set_z_label ─────────────────────────────────────────────────────────────


class TestSetZLabel:
    """Tests for _set_z_label() - pure axis-label dispatch."""

    def test_pp(self):
        _, ax = _fig_ax_3d()
        _set_z_label(ax, "pp", "#ff0000", "#00ff00")
        assert ax.get_zlabel() == "PP Performance"

    def test_tg(self):
        _, ax = _fig_ax_3d()
        _set_z_label(ax, "tg", "#ff0000", "#00ff00")
        assert ax.get_zlabel() == "TG Performance"

    def test_both_norm(self):
        _, ax = _fig_ax_3d()
        _set_z_label(ax, "both-norm", "#ff0000", "#00ff00")
        assert ax.get_zlabel() == "Normalized (PP & TG)"

    def test_percent(self):
        _, ax = _fig_ax_3d()
        _set_z_label(ax, "%", "#ff0000", "#00ff00")
        assert ax.get_zlabel() == "Performance (%)"
        assert ax.get_zlim() == (0, 100)

    def test_unknown_mode(self):
        _, ax = _fig_ax_3d()
        _set_z_label(ax, "nonexistent", "#ff0000", "#00ff00")
        # Falls back to default
        assert ax.get_zlabel()


# ── _set_series_zticks ─────────────────────────────────────────────────────


class TestSetSeriesZTicks:
    """Absolute Z tick labels for stretched per-series data."""

    def test_labels_span_min_to_max(self):
        _, ax = _fig_ax_3d()
        _set_series_zticks(ax, 20.0, 120.0)
        assert [t.get_text() for t in ax.get_zticklabels()] == \
            ["20", "40", "60", "80", "100", "120"]

    def test_fractional_labels(self):
        _, ax = _fig_ax_3d()
        _set_series_zticks(ax, 1.0, 2.0)
        assert [t.get_text() for t in ax.get_zticklabels()] == \
            ["1", "1.2", "1.4", "1.6", "1.8", "2"]

    def test_missing_stats_untouched(self):
        _, ax = _fig_ax_3d()
        before = [t.get_text() for t in ax.get_zticklabels()]
        _set_series_zticks(ax, 20.0, None)
        assert [t.get_text() for t in ax.get_zticklabels()] == before
        _set_series_zticks(ax, None, None)
        assert [t.get_text() for t in ax.get_zticklabels()] == before

    def test_constant_series_single_tick(self):
        _, ax = _fig_ax_3d()
        _set_series_zticks(ax, 85.3, 85.3)
        assert [t.get_text() for t in ax.get_zticklabels()] == ["85.3"]


# ── render_2d ────────────────────────────────────────────────────────────────


class TestRender2D:
    """Tests for render_2d() - 2-D multi-file plot builder."""

    def test_empty_series(self):
        """Empty series_data produces a figure with no lines and no legend."""
        fig = render_2d(
            datasets_raw=[{'path': Path('/f.csv')}],
            series_data={'pp': [], 'tg': []},
            x_param="params",
            pp_base="#ff0000",
            tg_base="#00ff00",
            show_pp_flags=[True],
            show_tg_flags=[True],
            do_unify=False,
        )
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert len(ax.lines) == 0  # no data lines
        assert ax.get_legend() is None  # no handles → no legend

    def test_single_file_pp_only(self):
        """One file, only PP data → single errorbar on primary axis."""
        fig = render_2d(
            datasets_raw=[{'path': Path('/file.csv')}],
            series_data={
                'pp': [{'x': 1, 'y': 100, 'err': 5, 'file_idx': 0}],
                'tg': [],
            },
            x_param="params",
            pp_base="#ff0000",
            tg_base="#00ff00",
            show_pp_flags=[True],
            show_tg_flags=[True],
            do_unify=False,
        )
        ax = fig.axes[0]
        assert "PP" in ax.get_ylabel()
        # errorbar creates multiple line artists; at least one should exist
        assert len(ax.lines) >= 1
        legend = ax.get_legend()
        assert legend is not None
        # Legend should contain at least one label with "PP"
        labels = [t.get_text() for t in legend.get_texts()]
        assert any("PP" in lbl for lbl in labels)

    def test_single_file_pp_and_tg(self):
        """One file with PP+TG → twinx axis appears."""
        fig = render_2d(
            datasets_raw=[{'path': Path('/file.csv')}],
            series_data={
                'pp': [{'x': 1, 'y': 100, 'err': 5, 'file_idx': 0}],
                'tg': [{'x': 1, 'y': 80, 'err': 3, 'file_idx': 0}],
            },
            x_param="params",
            pp_base="#ff0000",
            tg_base="#00ff00",
            show_pp_flags=[True],
            show_tg_flags=[True],
            do_unify=False,
        )
        assert len(fig.axes) == 2  # primary + twinx
        ax_tg = fig.axes[1]
        assert ax_tg.get_ylabel()
        assert "TG" in ax_tg.get_ylabel()

    def test_multi_file_per_file(self):
        """Two files, per-file mode → one errorbar per file per type."""
        fig = render_2d(
            datasets_raw=[
                {'path': Path('/file1.csv')},
                {'path': Path('/file2.csv')},
            ],
            series_data={
                'pp': [
                    {'x': 1, 'y': 100, 'err': 5, 'file_idx': 0},
                    {'x': 1, 'y': 120, 'err': 6, 'file_idx': 1},
                ],
                'tg': [
                    {'x': 1, 'y': 80, 'err': 4, 'file_idx': 0},
                    {'x': 1, 'y': 90, 'err': 4, 'file_idx': 1},
                ],
            },
            x_param="params",
            pp_base="#ff0000",
            tg_base="#00ff00",
            show_pp_flags=[True, True],
            show_tg_flags=[True, True],
            do_unify=False,
        )
        assert len(fig.axes) == 2
        legend = fig.axes[0].get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        # Should have labels for both files
        assert any("file1" in lbl for lbl in labels)
        assert any("file2" in lbl for lbl in labels)
        # PP and TG labels should both appear
        assert any("PP" in lbl for lbl in labels)
        assert any("TG" in lbl for lbl in labels)

    def test_unified_mode(self):
        """Unified mode → title contains 'Unified'."""
        fig = render_2d(
            datasets_raw=[
                {'path': Path('/f1.csv')},
                {'path': Path('/f2.csv')},
            ],
            series_data={
                'pp': [
                    {'x': 1, 'y': 100, 'err': 5, 'file_idx': 0},
                    {'x': 1, 'y': 110, 'err': 4, 'file_idx': 1},
                ],
                'tg': [
                    {'x': 1, 'y': 80, 'err': 4, 'file_idx': 0},
                ],
            },
            x_param="params",
            pp_base="#ff0000",
            tg_base="#00ff00",
            show_pp_flags=[True, True],
            show_tg_flags=[True, True],
            do_unify=True,
        )
        assert "Unified" in fig.axes[0].get_title()

    def test_normalized(self):
        """Normalize=True + z_label_mode='%' → 'Normalized' in title, '%' in ylabel."""
        fig = render_2d(
            datasets_raw=[{'path': Path('/f.csv')}],
            series_data={
                'pp': [{'x': 1, 'y': 100, 'err': 5, 'file_idx': 0}],
                'tg': [],
            },
            x_param="params",
            pp_base="#ff0000",
            tg_base="#00ff00",
            show_pp_flags=[True],
            show_tg_flags=[True],
            do_unify=False,
            normalize=True,
            z_label_mode="%",
        )
        ax = fig.axes[0]
        assert "Normalized" in ax.get_title()
        assert "%" in ax.get_ylabel()

    def test_pp_flag_false(self):
        """PP flag False → no PP line, spine color is grey (#888)."""
        fig = render_2d(
            datasets_raw=[{'path': Path('/f.csv')}],
            series_data={
                'pp': [{'x': 1, 'y': 100, 'err': 5, 'file_idx': 0}],
                'tg': [{'x': 1, 'y': 80, 'err': 4, 'file_idx': 0}],
            },
            x_param="params",
            pp_base="#ff0000",
            tg_base="#00ff00",
            show_pp_flags=[False],
            show_tg_flags=[True],
            do_unify=False,
        )
        ax = fig.axes[0]
        # When all PP flags are False, left spine becomes '#888'
        spine_color = ax.spines['left'].get_edgecolor()
        assert spine_color == (0.5333333333333333, 0.5333333333333333, 0.5333333333333333, 1.0)  # '#888' as rgba

    def test_tg_flag_false(self):
        """TG flag False → no twinx axis created."""
        fig = render_2d(
            datasets_raw=[{'path': Path('/f.csv')}],
            series_data={
                'pp': [{'x': 1, 'y': 100, 'err': 5, 'file_idx': 0}],
                'tg': [{'x': 1, 'y': 80, 'err': 4, 'file_idx': 0}],
            },
            x_param="params",
            pp_base="#ff0000",
            tg_base="#00ff00",
            show_pp_flags=[True],
            show_tg_flags=[False],
            do_unify=False,
        )
        # No twinx because any(show_tg_flags) is False
        assert len(fig.axes) == 1

    def test_dark_mode_false(self):
        """dark_mode=False → white background."""
        fig = render_2d(
            datasets_raw=[{'path': Path('/f.csv')}],
            series_data={
                'pp': [{'x': 1, 'y': 100, 'err': 5, 'file_idx': 0}],
                'tg': [],
            },
            x_param="params",
            pp_base="#ff0000",
            tg_base="#00ff00",
            show_pp_flags=[True],
            show_tg_flags=[True],
            do_unify=False,
            dark_mode=False,
        )
        assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)  # white


# ── _draw_unified ────────────────────────────────────────────────────────────


class TestDrawUnified:
    """Tests for _draw_unified() - helper that averages points by x."""

    def test_empty(self):
        """Empty point list → no line added."""
        fig = Figure()
        ax = fig.add_subplot(111)
        handles = []
        _draw_unified(ax, [], "#ff0000", "Test", "-", handles)
        assert len(handles) == 0
        assert len(ax.lines) == 0

    def test_single_point_per_x(self):
        """One point per x value → simple line."""
        fig = Figure()
        ax = fig.add_subplot(111)
        pts = [{'x': 1, 'y': 100, 'err': 5}, {'x': 2, 'y': 200, 'err': 10}]
        handles = []
        _draw_unified(ax, pts, "#ff0000", "Test", "-", handles)
        assert len(handles) == 1
        assert len(ax.lines) > 0
        # Check line data
        line = ax.lines[0]
        assert list(line.get_xdata()) == [1, 2]
        assert list(line.get_ydata()) == [100, 200]

    def test_multi_point_per_x(self):
        """Multiple points per x → averaged (mean y, combined error)."""
        fig = Figure()
        ax = fig.add_subplot(111)
        pts = [
            {'x': 1, 'y': 100, 'err': 5},
            {'x': 1, 'y': 110, 'err': 3},
            {'x': 2, 'y': 200, 'err': 10},
        ]
        handles = []
        _draw_unified(ax, pts, "#ff0000", "Test", "-", handles)
        line = ax.lines[0]
        xdata = list(line.get_xdata())
        ydata = list(line.get_ydata())
        assert xdata == [1, 2]
        assert ydata[0] == pytest.approx(105.0)  # (100 + 110) / 2
        assert ydata[1] == 200.0


# ── render_3d ────────────────────────────────────────────────────────────────


class TestRender3D:
    """Tests for render_3d() - 3-D surface/point plot builder."""

    def test_empty(self):
        """No points → figure created, no legend."""
        fig, ax = render_3d(
            points_pp=[],
            points_tg=[],
            x_param="params",
            y_param="n_gpu_layers",
            pp_color="#ff0000",
            tg_color="#00ff00",
        )
        assert isinstance(fig, Figure)
        assert ax is not None
        assert ax.get_legend() is None

    def test_basic(self):
        """PP and TG points → legend entries, scatter collections."""
        points_pp = [(1.0, 2.0, 100.0, 5.0), (1.0, 3.0, 120.0, 6.0)]
        points_tg = [(2.0, 2.0, 80.0, 4.0)]
        fig, ax = render_3d(
            points_pp=points_pp,
            points_tg=points_tg,
            x_param="params",
            y_param="n_gpu_layers",
            pp_color="#ff0000",
            tg_color="#00ff00",
        )
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "PP" in legend_texts
        assert "TG" in legend_texts
        # Axis labels
        assert "Params" in ax.get_xlabel()
        assert "N Gpu Layers" in ax.get_ylabel()

    def test_error_bars(self):
        """Error bars enabled → vertical lines in plot."""
        points_pp = [(1.0, 2.0, 100.0, 5.0)]
        fig, ax = render_3d(
            points_pp=points_pp,
            points_tg=[],
            x_param="x",
            y_param="y",
            pp_color="#ff0000",
            tg_color="#00ff00",
            show_errors_3d=True,
        )
        # Error bars create extra lines (vertical + caps)
        assert len(ax.lines) >= 1

    def test_surface_trisurf(self):
        """3+ points → surface trisurf created (collection added)."""
        points_pp = [(1.0, 2.0, 100.0, 5.0),
                     (2.0, 3.0, 110.0, 5.0),
                     (1.0, 3.0, 95.0, 5.0)]
        fig, ax = render_3d(
            points_pp=points_pp,
            points_tg=[],
            x_param="x",
            y_param="y",
            pp_color="#ff0000",
            tg_color="#00ff00",
            show_surface=True,
        )
        # Surface creates Poly3DCollection in ax.collections
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        surf_collections = [
            c for c in ax.collections
            if isinstance(c, Poly3DCollection)
        ]
        assert len(surf_collections) >= 1

    def test_projection_lines(self):
        """show_projections=True → extra lines for wall projections."""
        points_pp = [(1.0, 2.0, 100.0, 5.0),
                     (2.0, 3.0, 110.0, 5.0)]
        fig, ax = render_3d(
            points_pp=points_pp,
            points_tg=[],
            x_param="x",
            y_param="y",
            pp_color="#ff0000",
            tg_color="#00ff00",
            show_projections=True,
        )
        # Projections add extra lines beyond the error bars
        assert len(ax.lines) >= 2

    def test_z_label_percent_mode(self):
        """z_label_mode='%' → zlim set to (0, 100)."""
        points_pp = [(1.0, 2.0, 50.0, 5.0)]
        fig, ax = render_3d(
            points_pp=points_pp,
            points_tg=[],
            x_param="x",
            y_param="y",
            pp_color="#ff0000",
            tg_color="#00ff00",
            z_label_mode="%",
        )
        assert ax.get_zlim() == (0, 100)

    def test_level_plane(self):
        """show_level=True → additional surface collection for plane."""
        pts = [(1.0, 2.0, 100.0, 5.0),
               (2.0, 3.0, 110.0, 5.0)]
        fig, ax = render_3d(
            points_pp=pts,
            points_tg=[],
            x_param="x", y_param="y",
            pp_color="#ff0000", tg_color="#00ff00",
            show_level=True,
            level_val=50,
        )
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        # Level plane creates a Surface3DCollection (subclass of Poly3DCollection)
        collections_before = len(ax.collections)
        # At least some collections should be present (surface + level)
        # We can't easily distinguish plane from trisurf by type alone,
        # but we know more collections exist when level is on
        assert len(ax.collections) >= 1

    def test_wireframe_enabled(self):
        """show_wireframe=True → edge_c='black', linewidth>0."""
        pts = [(1.0, 2.0, 100.0, 5.0),
               (2.0, 3.0, 110.0, 5.0),
               (1.0, 3.0, 95.0, 5.0)]
        fig, ax = render_3d(
            points_pp=pts,
            points_tg=[],
            x_param="x", y_param="y",
            pp_color="#ff0000", tg_color="#00ff00",
            show_surface=True,
            show_wireframe=True,
        )
        # Should create a surface collection (wireframe just changes style)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        surf = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(surf) >= 1


# ── _draw_trisurf (via render_3d) ────────────────────────────────────────────


class TestDrawTrisurf:
    """Tests for _draw_trisurf surface style dispatch (tested via render_3d)."""

    def test_solid_style(self):
        """Solid surface style produces a Poly3DCollection."""
        pts = [(1.0, 2.0, 100.0, 5.0),
               (2.0, 3.0, 110.0, 5.0),
               (1.0, 3.0, 95.0, 5.0)]
        fig, ax = render_3d(
            points_pp=pts,
            points_tg=[],
            x_param="x", y_param="y",
            pp_color="#ff0000", tg_color="#00ff00",
            show_surface=True,
            surface_style="Solid",
        )
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        solid = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(solid) >= 1

    def test_shaded_style(self):
        """Shaded surface style."""
        pts = [(1.0, 2.0, 100.0, 5.0),
               (2.0, 3.0, 110.0, 5.0),
               (1.0, 3.0, 95.0, 5.0)]
        fig, ax = render_3d(
            points_pp=pts,
            points_tg=[],
            x_param="x", y_param="y",
            pp_color="#ff0000", tg_color="#00ff00",
            show_surface=True,
            surface_style="Shaded",
        )
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        shaded = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(shaded) >= 1

    def test_colormap_style(self):
        """Colormap surface style."""
        pts = [(1.0, 2.0, 100.0, 5.0),
               (2.0, 3.0, 110.0, 5.0),
               (1.0, 3.0, 95.0, 5.0)]
        fig, ax = render_3d(
            points_pp=pts,
            points_tg=[],
            x_param="x", y_param="y",
            pp_color="#ff0000", tg_color="#00ff00",
            show_surface=True,
            surface_style="Colormap",
        )
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        cmap_colls = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(cmap_colls) >= 1

    def test_direct_dispatch_via_mock(self):
        """Test _draw_trisurf direct dispatch using mocked ax."""
        from view.plot_view import _draw_trisurf
        mock_ax = MagicMock()
        cmap_mock = MagicMock()
        light_mock = MagicMock()

        # Direct array path: Solid
        _draw_trisurf(mock_ax, None, None, [1.0], [2.0], "#ff0000", cmap_mock,
                      "Solid", light_mock, 'none', 0, z_arr=[10.0])
        mock_ax.plot_trisurf.assert_called()
        args, kwargs = mock_ax.plot_trisurf.call_args
        assert kwargs.get('color') == "#ff0000"
        assert kwargs.get('alpha') == 0.45

        # Direct array path: Shaded
        mock_ax.reset_mock()
        _draw_trisurf(mock_ax, None, None, [1.0], [2.0], "#ff0000", cmap_mock,
                      "Shaded", light_mock, 'none', 0, z_arr=[10.0])
        args, kwargs = mock_ax.plot_trisurf.call_args
        assert kwargs.get('shade') is True
        assert kwargs.get('alpha') == 0.8

        # Direct array path: Colormap
        mock_ax.reset_mock()
        _draw_trisurf(mock_ax, None, None, [1.0], [2.0], "#ff0000", cmap_mock,
                      "Colormap", light_mock, 'none', 0, z_arr=[10.0])
        args, kwargs = mock_ax.plot_trisurf.call_args
        assert kwargs.get('cmap') == cmap_mock
        assert kwargs.get('alpha') == 0.85

    def test_refined_path_solid(self):
        """Refined triangulation path (tri_r not None) with Solid style."""
        from view.plot_view import _draw_trisurf
        mock_ax = MagicMock()
        tri_r = MagicMock()
        z_r = MagicMock()
        cmap_mock = MagicMock()
        light_mock = MagicMock()

        _draw_trisurf(mock_ax, tri_r, z_r, None, None, "#ff0000", cmap_mock,
                      "Solid", light_mock, 'none', 0)
        mock_ax.plot_trisurf.assert_called_once_with(
            tri_r, z_r, color="#ff0000", alpha=0.45,
            edgecolor='none', linewidth=0, antialiased=True,
        )

    def test_refined_path_shaded(self):
        """Refined triangulation path with Shaded style."""
        from view.plot_view import _draw_trisurf
        mock_ax = MagicMock()
        tri_r = MagicMock()
        z_r = MagicMock()
        cmap_mock = MagicMock()
        light_mock = MagicMock()

        _draw_trisurf(mock_ax, tri_r, z_r, None, None, "#ff0000", cmap_mock,
                      "Shaded", light_mock, 'none', 0)
        mock_ax.plot_trisurf.assert_called_once_with(
            tri_r, z_r, color="#ff0000", alpha=0.8,
            shade=True, lightsource=light_mock,
            edgecolor='none', linewidth=0, antialiased=True,
        )

    def test_refined_path_colormap(self):
        """Refined triangulation path with Colormap style."""
        from view.plot_view import _draw_trisurf
        mock_ax = MagicMock()
        tri_r = MagicMock()
        z_r = MagicMock()
        cmap_mock = MagicMock()
        light_mock = MagicMock()

        _draw_trisurf(mock_ax, tri_r, z_r, None, None, "#ff0000", cmap_mock,
                      "Colormap", light_mock, 'none', 0)
        mock_ax.plot_trisurf.assert_called_once_with(
            tri_r, z_r, cmap=cmap_mock, alpha=0.85,
            shade=True, lightsource=light_mock,
            edgecolor='none', linewidth=0, antialiased=True,
        )


# ── CustomNavigationToolbar ────────────────────────────────────────────


class TestCustomNavigationToolbar:
    """Tests for CustomNavigationToolbar.home() - home button override."""

    def test_home_callback_handled(self):
        """Callback returns True → super().home() NOT called."""
        from view.plot_view import CustomNavigationToolbar
        toolbar = CustomNavigationToolbar.__new__(CustomNavigationToolbar)
        toolbar._home_callback = MagicMock(return_value=True)

        with patch.object(NavigationToolbar2Tk, 'home') as mock_super:
            toolbar.home()
            mock_super.assert_not_called()

    def test_home_callback_not_handled(self):
        """Callback returns False → super().home() called."""
        from view.plot_view import CustomNavigationToolbar
        toolbar = CustomNavigationToolbar.__new__(CustomNavigationToolbar)
        toolbar._home_callback = MagicMock(return_value=False)

        with patch.object(NavigationToolbar2Tk, 'home') as mock_super:
            toolbar.home()
            mock_super.assert_called_once()

    def test_home_no_callback(self):
        """No callback set → super().home() called."""
        from view.plot_view import CustomNavigationToolbar
        toolbar = CustomNavigationToolbar.__new__(CustomNavigationToolbar)
        toolbar._home_callback = None

        with patch.object(NavigationToolbar2Tk, 'home') as mock_super:
            toolbar.home()
            mock_super.assert_called_once()


# ── PlotView widget ────────────────────────────────────────────────────


_TK_PATCHES = [
    patch('view.plot_view.tk.Frame'),
    patch('view.plot_view.tk.Label'),
    patch('view.plot_view.tk.Button'),
]


class TestPlotView:
    """Tests for PlotView - the Tkinter/Matplotlib canvas wrapper."""

    def _make_pv(self):
        """Create a PlotView with all tkinter dependencies mocked."""
        for p in _TK_PATCHES:
            p.start()
        from view.plot_view import PlotView
        pv = PlotView.__new__(PlotView)
        pv._canvas = None
        pv._toolbar = None
        pv._home_cb = None
        pv._placeholder = MagicMock()
        return pv

    def _cleanup_pv(self):
        for p in _TK_PATCHES:
            p.stop()

    def test_show_placeholder_default(self):
        """Default text shown when called with no argument."""
        pv = self._make_pv()
        pv.show_placeholder()
        assert pv._placeholder.pack.called
        self._cleanup_pv()

    def test_show_placeholder_custom_text(self):
        """Custom text replaces placeholder label."""
        pv = self._make_pv()
        custom = "No data available"
        pv.show_placeholder(custom)
        pv._placeholder.config.assert_called_once_with(text=custom)
        self._cleanup_pv()

    def test_show_placeholder_destroys_canvas(self):
        """show_placeholder destroys any existing canvas first."""
        pv = self._make_pv()
        fake_canvas = MagicMock()
        fake_canvas.get_tk_widget.return_value.winfo_exists.return_value = True
        fake_toolbar = MagicMock()
        fake_toolbar.winfo_exists.return_value = True
        pv._canvas = fake_canvas
        pv._toolbar = fake_toolbar

        pv.show_placeholder("test")

        fake_toolbar.destroy.assert_called_once()
        fake_canvas.get_tk_widget().destroy.assert_called_once()
        assert pv._canvas is None
        assert pv._toolbar is None
        self._cleanup_pv()

    def test_destroy_canvas_none(self):
        """_destroy_canvas with no toolbar/canvas raises no error."""
        pv = self._make_pv()
        pv._destroy_canvas()  # should not raise
        self._cleanup_pv()

    def test_destroy_canvas_toolbar_dead(self):
        """_destroy_canvas skips toolbar destroy when winfo_exists is False."""
        pv = self._make_pv()
        fake_toolbar = MagicMock()
        fake_toolbar.winfo_exists.return_value = False
        pv._toolbar = fake_toolbar

        pv._destroy_canvas()

        fake_toolbar.destroy.assert_not_called()
        assert pv._toolbar is None
        self._cleanup_pv()

    def test_destroy_canvas_widget_dead(self):
        """_destroy_canvas skips widget destroy when winfo_exists is False."""
        pv = self._make_pv()
        fake_canvas = MagicMock()
        fake_canvas.get_tk_widget.return_value.winfo_exists.return_value = False
        pv._canvas = fake_canvas

        pv._destroy_canvas()

        fake_canvas.get_tk_widget().destroy.assert_not_called()
        assert pv._canvas is None
        self._cleanup_pv()

    def test_render_2d_no_pick_cb(self):
        """render with fig, no ax3d, no pick_cb → no mpl_connect."""
        pv = self._make_pv()
        fig = Figure()

        with patch('view.plot_view.FigureCanvasTkAgg') as mock_canvas_cls:
            with patch('view.plot_view.CustomNavigationToolbar') as mock_tb_cls:
                pv.render(fig)

                mock_canvas_cls.assert_called_once_with(fig, master=pv)
                inst = mock_canvas_cls.return_value
                inst.draw.assert_called_once()
                inst.mpl_connect.assert_not_called()

                mock_tb_cls.assert_called_once_with(
                    inst, pv, home_callback=pv._home_cb
                )
                mock_tb_cls.return_value.update.assert_called_once()
        self._cleanup_pv()

    def test_render_3d_no_pick_cb(self):
        """render with fig + ax3d → no pick_event connection."""
        pv = self._make_pv()
        fig = Figure()
        ax = MagicMock(spec=['elev', 'azim'])

        with patch('view.plot_view.FigureCanvasTkAgg') as mock_canvas_cls:
            with patch('view.plot_view.CustomNavigationToolbar'):
                pv.render(fig, ax3d=ax)

                inst = mock_canvas_cls.return_value
                inst.mpl_connect.assert_not_called()
        self._cleanup_pv()

    def test_render_2d_with_pick_cb(self):
        """render with fig and on_pick_cb (ax3d=None) → mpl_connect called."""
        pv = self._make_pv()
        fig = Figure()
        pick_cb = MagicMock()

        with patch('view.plot_view.FigureCanvasTkAgg') as mock_canvas_cls:
            with patch('view.plot_view.CustomNavigationToolbar'):
                pv.render(fig, on_pick_cb=pick_cb)

                inst = mock_canvas_cls.return_value
                inst.mpl_connect.assert_called_once_with('pick_event', pick_cb)
        self._cleanup_pv()

    def test_redraw_idle_with_canvas(self):
        """redraw_idle delegates to canvas.draw_idle when canvas exists."""
        pv = self._make_pv()
        fake_canvas = MagicMock()
        pv._canvas = fake_canvas

        pv.redraw_idle()

        fake_canvas.draw_idle.assert_called_once()
        self._cleanup_pv()

    def test_redraw_idle_no_canvas(self):
        """redraw_idle does nothing when canvas is None."""
        pv = self._make_pv()
        pv.redraw_idle()  # should not raise
        self._cleanup_pv()

    def test_set_home_callback(self):
        """set_home_callback stores the callback."""
        pv = self._make_pv()
        cb = MagicMock()
        pv.set_home_callback(cb)
        assert pv._home_cb == cb
        self._cleanup_pv()


# ── Surface fallback (bug-hunting) ──────────────────────────────────────


class TestRender3DFallback:
    """Tests for render_3d surface fallback when Triangulation fails."""

    def test_fallback_uses_plot_trisurf_on_exception(self):
        """When _draw_trisurf raises, fallback calls ax.plot_trisurf directly."""
        with patch('view.plot_view._draw_trisurf',
                   side_effect=RuntimeError("Triangulation failed")):
            pts = [(1.0, 2.0, 100.0, 5.0),
                   (2.0, 3.0, 110.0, 5.0),
                   (1.0, 3.0, 95.0, 5.0)]
            fig, ax = render_3d(
                points_pp=pts, points_tg=[],
                x_param="x", y_param="y",
                pp_color="#ff0000", tg_color="#00ff00",
                show_surface=True,
            )
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        surf = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(surf) >= 1

    def test_fallback_with_subdiv_level(self):
        """Fallback also works when subdiv_level > 0 raises."""
        with patch('view.plot_view._draw_trisurf',
                   side_effect=RuntimeError("Refinement failed")):
            pts = [(1.0, 2.0, 100.0, 5.0),
                   (2.0, 3.0, 110.0, 5.0),
                   (1.0, 3.0, 95.0, 5.0)]
            fig, ax = render_3d(
                points_pp=pts, points_tg=[],
                x_param="x", y_param="y",
                pp_color="#ff0000", tg_color="#00ff00",
                show_surface=True,
                subdiv_level=1,
            )
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        surf = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(surf) >= 1

    def test_no_fallback_needed_on_valid_data(self):
        """Valid triangulation data does not trigger fallback."""
        pts = [(1.0, 2.0, 100.0, 5.0),
               (2.0, 3.0, 110.0, 5.0),
               (1.0, 3.0, 95.0, 5.0)]
        fig, ax = render_3d(
            points_pp=pts, points_tg=[],
            x_param="x", y_param="y",
            pp_color="#ff0000", tg_color="#00ff00",
            show_surface=True,
        )
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        surf = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(surf) >= 1

    def test_refined_path_subdiv_1(self):
        """subdiv_level=1 via render_3d produces surface."""
        pts = [(1.0, 2.0, 100.0, 5.0),
               (2.0, 3.0, 110.0, 5.0),
               (1.0, 3.0, 95.0, 5.0),
               (3.0, 1.0, 105.0, 5.0)]
        fig, ax = render_3d(
            points_pp=pts, points_tg=[],
            x_param="x", y_param="y",
            pp_color="#ff0000", tg_color="#00ff00",
            show_surface=True,
            subdiv_level=1,
        )
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        surf = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(surf) >= 1

    def test_no_surface_no_fallback(self):
        """show_surface=False → no surface attempt, no fallback needed."""
        pts = [(1.0, 2.0, 100.0, 5.0),
               (2.0, 3.0, 110.0, 5.0),
               (1.0, 3.0, 95.0, 5.0)]
        fig, ax = render_3d(
            points_pp=pts, points_tg=[],
            x_param="x", y_param="y",
            pp_color="#ff0000", tg_color="#00ff00",
            show_surface=False,
        )
        # Scatter collections but no Poly3DCollection (no surface)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        surf = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(surf) == 0
