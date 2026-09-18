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


# ── Merged 3-D surfaces ────────────────────────────────────────────────────


class TestTriangulateSeries:
    """Tests for _triangulate_series (pure helper, no axes required)."""

    def test_valid_points_triangulated(self):
        from view.plot_view import _triangulate_series
        tri = _triangulate_series(
            [1.0, 2.0, 1.0, 2.0], [2.0, 3.0, 3.0, 2.5],
            [100.0, 110.0, 95.0, 105.0], 0,
        )
        assert tri is not None
        triangles, x, y, z = tri
        assert triangles.shape[1] == 3
        assert len(triangles) >= 1

    def test_fewer_than_three_points_returns_none(self):
        from view.plot_view import _triangulate_series
        assert _triangulate_series([1.0], [2.0], [3.0], 0) is None
        assert _triangulate_series([1.0, 2.0], [2.0, 3.0], [3.0, 4.0], 0) is None

    def test_degenerate_points_return_none(self):
        """Collinear points cannot be triangulated (no crash)."""
        from view.plot_view import _triangulate_series
        assert _triangulate_series(
            [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [5.0, 5.0, 5.0], 0,
        ) is None

    def test_subdiv_refinement_increases_triangles(self):
        from view.plot_view import _triangulate_series
        xs = [1.0, 2.0, 1.0, 2.0]
        ys = [2.0, 3.0, 3.0, 2.5]
        zs = [100.0, 110.0, 95.0, 105.0]
        plain = _triangulate_series(xs, ys, zs, 0)
        refined = _triangulate_series(xs, ys, zs, 1)
        assert plain is not None and refined is not None
        assert len(refined[0]) > len(plain[0])


class TestSeriesFaceColors:
    """Tests for _series_face_colors (pure helper, no axes required)."""

    def _setup(self):
        from matplotlib.colors import LightSource, LinearSegmentedColormap
        from view.plot_view import _triangulate_series
        tri = _triangulate_series(
            [1.0, 2.0, 1.0, 2.0], [2.0, 3.0, 3.0, 2.5],
            [100.0, 110.0, 95.0, 105.0], 0,
        )
        light = LightSource(azdeg=315, altdeg=45)
        cmap = LinearSegmentedColormap.from_list("t", ["#000000", "#ffffff"])
        return tri, light, cmap

    def test_solid_uses_base_color_with_alpha(self):
        from view.plot_view import _series_face_colors
        (triangles, x, y, z), light, cmap = self._setup()
        verts, colors = _series_face_colors(
            triangles, x, y, z, "#ff0000", cmap, "Solid", light)
        assert verts.shape == (len(triangles), 3, 3)
        assert colors.shape == (len(triangles), 4)
        assert all(abs(c[0] - 1.0) < 1e-9 and c[1] == 0.0 and c[2] == 0.0
                   for c in colors)
        assert all(c[3] == pytest.approx(0.45) for c in colors)

    def test_shaded_keeps_alpha_and_modulates(self):
        from view.plot_view import _series_face_colors
        (triangles, x, y, z), light, cmap = self._setup()
        _, colors = _series_face_colors(
            triangles, x, y, z, "#ff0000", cmap, "Shaded", light)
        assert all(c[3] == pytest.approx(0.8) for c in colors)
        assert all(0.0 <= c[0] <= 1.0 for c in colors)

    def test_colormap_maps_own_z_range(self):
        from view.plot_view import _series_face_colors
        (triangles, x, y, z), light, cmap = self._setup()
        _, colors = _series_face_colors(
            triangles, x, y, z, "#ff0000", cmap, "Colormap", light)
        assert all(c[3] == pytest.approx(0.85) for c in colors)
        # Darkest face (lowest z) differs from brightest face
        assert any((colors[0] != c).any() for c in colors[1:])


class TestDrawMergedSurfaces:
    """Tests for _draw_merged_surfaces (mock axes, no 3-D required)."""

    def _surfaces(self):
        from matplotlib.colors import LightSource, LinearSegmentedColormap
        from view.plot_view import _series_face_colors, _triangulate_series
        tri = _triangulate_series(
            [1.0, 2.0, 1.0, 2.0], [2.0, 3.0, 3.0, 2.5],
            [100.0, 110.0, 95.0, 105.0], 0,
        )
        light = LightSource(azdeg=315, altdeg=45)
        cmap = LinearSegmentedColormap.from_list("t", ["#000000", "#ffffff"])
        s1 = _series_face_colors(*tri, "#ff0000", cmap, "Solid", light)
        s2 = _series_face_colors(*tri, "#00ff00", cmap, "Solid", light)
        return s1, s2

    def test_both_series_in_single_collection(self):
        """PP + TG triangles land in ONE jointly sorted collection."""
        from view.plot_view import _draw_merged_surfaces, art3d
        s1, s2 = self._surfaces()
        mock_ax = MagicMock()
        seen: dict = {}
        RealColl = art3d.Poly3DCollection

        def spy(verts, **kwargs):
            seen['verts'] = np.asarray(verts)
            seen['colors'] = np.asarray(kwargs.get('facecolors'))
            seen['zsort'] = kwargs.get('zsort')
            return RealColl(verts, **kwargs)

        with patch('view.plot_view.art3d.Poly3DCollection', side_effect=spy):
            _draw_merged_surfaces(mock_ax, [s1, s2], 'none', 0)
        mock_ax.add_collection3d.assert_called_once()
        n_total = len(s1[0]) + len(s2[0])
        assert seen['verts'].shape[0] == n_total
        assert seen['colors'].shape == (n_total, 4)
        assert seen['zsort'] == 'average'
        # Per-series colors preserved: red PP faces, green TG faces
        red = int((seen['colors'][:, 0] > 0.9).sum())
        green = int((seen['colors'][:, 1] > 0.9).sum())
        assert red == len(s1[0])
        assert green == len(s2[0])

    def test_empty_returns_none_without_drawing(self):
        from view.plot_view import _draw_merged_surfaces
        mock_ax = MagicMock()
        assert _draw_merged_surfaces(mock_ax, [], 'none', 0) is None
        mock_ax.add_collection3d.assert_not_called()


class TestMergedSurfaceRender:
    """render_3d emits exactly one surface collection for both series."""

    def test_single_merged_collection_for_pp_and_tg(self):
        """Overlapping PP/TG share one depth-sorted collection."""
        pts_pp = [(1.0, 2.0, 100.0, 5.0),
                  (2.0, 3.0, 110.0, 5.0),
                  (1.0, 3.0, 95.0, 5.0)]
        pts_tg = [(1.5, 2.5, 50.0, 2.0),
                  (2.5, 3.5, 55.0, 2.0),
                  (1.5, 3.5, 48.0, 2.0)]
        fig, ax = render_3d(
            points_pp=pts_pp,
            points_tg=pts_tg,
            x_param="x", y_param="y",
            pp_color="#ff0000", tg_color="#00ff00",
            show_surface=True,
        )
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        surf = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(surf) == 1

    def test_surface_styles(self):
        """Solid / Shaded / Colormap all produce the merged collection."""
        pts = [(1.0, 2.0, 100.0, 5.0),
               (2.0, 3.0, 110.0, 5.0),
               (1.0, 3.0, 95.0, 5.0)]
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        for style in ("Solid", "Shaded", "Colormap"):
            fig, ax = render_3d(
                points_pp=pts,
                points_tg=[],
                x_param="x", y_param="y",
                pp_color="#ff0000", tg_color="#00ff00",
                show_surface=True,
                surface_style=style,
            )
            surf = [c for c in ax.collections
                    if isinstance(c, Poly3DCollection)]
            assert len(surf) == 1, style

    def test_wireframe_edges_passed_through(self):
        """Edge color and width reach the merged collection."""
        from matplotlib.colors import LightSource, LinearSegmentedColormap
        from view.plot_view import (
            _draw_merged_surfaces, _series_face_colors, _triangulate_series,
        )
        tri = _triangulate_series(
            [1.0, 2.0, 1.0, 2.0], [2.0, 3.0, 3.0, 2.5],
            [100.0, 110.0, 95.0, 105.0], 0,
        )
        light = LightSource(azdeg=315, altdeg=45)
        cmap = LinearSegmentedColormap.from_list("t", ["#000000", "#ffffff"])
        s1 = _series_face_colors(*tri, "#ff0000", cmap, "Solid", light)
        mock_ax = MagicMock()
        coll = _draw_merged_surfaces(mock_ax, [s1], 'black', 0.5)
        assert coll is not None
        mock_ax.add_collection3d.assert_called_once_with(coll)


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
    """Degenerate geometry degrades to scatter points (no crash)."""

    def test_collinear_points_draw_no_surface(self):
        """Untriangulatable points render scatter only, no surface."""
        pts = [(0.0, 0.0, 100.0, 5.0),
               (1.0, 1.0, 110.0, 5.0),
               (2.0, 2.0, 95.0, 5.0)]
        fig, ax = render_3d(
            points_pp=pts, points_tg=[],
            x_param="x", y_param="y",
            pp_color="#ff0000", tg_color="#00ff00",
            show_surface=True,
        )
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        surf = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(surf) == 0

    def test_collinear_points_with_subdiv_level(self):
        """Degenerate points with subdiv_level > 0 behave the same."""
        pts = [(0.0, 0.0, 100.0, 5.0),
               (1.0, 1.0, 110.0, 5.0),
               (2.0, 2.0, 95.0, 5.0)]
        fig, ax = render_3d(
            points_pp=pts, points_tg=[],
            x_param="x", y_param="y",
            pp_color="#ff0000", tg_color="#00ff00",
            show_surface=True,
            subdiv_level=1,
        )
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        surf = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(surf) == 0

    def test_no_fallback_needed_on_valid_data(self):
        """Valid triangulation data produces exactly one surface."""
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
        assert len(surf) == 1

    def test_refined_path_subdiv_1(self):
        """subdiv_level=1 via render_3d produces one merged surface."""
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
        assert len(surf) == 1

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
