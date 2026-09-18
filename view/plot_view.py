"""
view/plot_view.py

PlotView — the Matplotlib canvas + toolbar wrapper for llamagraph.

Contains:
  - PlotView: Tkinter Frame that hosts FigureCanvasTkAgg + CustomToolbar
  - render_2d(): stateless 2-D multi-file plot function
  - render_3d(): stateless 3-D surface plot function
  - CustomNavigationToolbar: Home-button override for camera persistence

All render_* functions return (Figure, Optional[Axes3D]) and are called
by the Presenter.  They have no side-effects beyond building the figure.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Optional

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.lines as mlines
import matplotlib.tri as mtri
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.figure import Figure

from utils.colors import COLORS, get_variant_color, normalize_series


# ── Custom Toolbar ────────────────────────────────────────────────────────────

class CustomNavigationToolbar(NavigationToolbar2Tk):
    """
    Overrides the Home button so that 3-D plots restore the exact
    camera state that was recorded at first-render time, not the
    generic Matplotlib default.
    """

    def __init__(self, canvas, parent, home_callback: Optional[Callable] = None):
        self._home_callback = home_callback
        super().__init__(canvas, parent, pack_toolbar=False)

    def home(self, *args):
        if self._home_callback and self._home_callback():
            return  # callback handled it
        super().home(*args)


# ── PlotView widget ───────────────────────────────────────────────────────────

class PlotView(tk.Frame):
    """
    Tkinter frame that owns the Matplotlib canvas.

    The Presenter calls:
      show_placeholder(text)
      render(fig, ax3d, on_pick_cb)   – display a freshly built Figure
      save_camera() / restore_camera() – delegates to the Presenter

    Callbacks set by the Presenter:
      set_home_callback(cb)   – called when the user clicks Home; return True
                                 if handled, False to let Matplotlib handle it
    """

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, bg=COLORS['bg'], **kwargs)

        self._canvas: Optional[FigureCanvasTkAgg] = None
        self._toolbar: Optional[CustomNavigationToolbar] = None
        self._home_cb: Optional[Callable] = None

        self._placeholder = tk.Label(
            self,
            text="📊 Select CSV file(s) with Ctrl+Click to display",
            bg=COLORS['bg'], fg=COLORS['fg'],
            font=('Segoe UI', 12),
        )
        self._placeholder.pack(expand=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_home_callback(self, cb: Callable) -> None:
        self._home_cb = cb

    def show_placeholder(self, text: str = "") -> None:
        """Clear the canvas and show placeholder text."""
        self._destroy_canvas()
        if text:
            self._placeholder.config(text=text)
        self._placeholder.pack(expand=True)

    def render(
        self,
        fig: Figure,
        ax3d=None,
        on_pick_cb: Optional[Callable] = None,
    ) -> None:
        """
        Display a Matplotlib Figure in the canvas area.

        Parameters
        ----------
        fig:
            The figure to display.
        ax3d:
            The 3-D Axes3D instance (if a 3-D plot), else None.
        on_pick_cb:
            Optional callback for 2-D pick events.
        """
        self._destroy_canvas()
        self._placeholder.pack_forget()

        self._canvas = FigureCanvasTkAgg(fig, master=self)
        self._canvas.draw()

        self._toolbar = CustomNavigationToolbar(
            self._canvas, self,
            home_callback=self._home_cb,
        )
        self._toolbar.update()
        self._toolbar.pack(side=tk.TOP, fill=tk.X)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if on_pick_cb and ax3d is None:
            self._canvas.mpl_connect('pick_event', on_pick_cb)

    def redraw_idle(self) -> None:
        if self._canvas:
            self._canvas.draw_idle()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _destroy_canvas(self) -> None:
        if self._toolbar:
            if self._toolbar.winfo_exists():
                self._toolbar.destroy()
            self._toolbar = None
        if self._canvas:
            widget = self._canvas.get_tk_widget()
            if widget.winfo_exists():
                widget.destroy()
            self._canvas = None


# ── 2-D Rendering ─────────────────────────────────────────────────────────────

def render_2d(
    datasets_raw: list[dict],
    series_data: dict,          # from Model.get_2d_series()
    x_param: str,
    pp_base: str,
    tg_base: str,
    show_pp_flags: list[bool],
    show_tg_flags: list[bool],
    do_unify: bool,
    dark_mode: bool = True,
    normalize: bool = False,
    z_label_mode: str = "%",
) -> Figure:
    """
    Build and return a 2-D multi-file comparison Figure.
    The function is stateless: it creates and returns a fresh Figure.
    """
    bg = COLORS['bg'] if dark_mode else 'white'
    fig = Figure(figsize=(10, 6), facecolor=bg)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg)
    ax.grid(True, linestyle='--', alpha=0.2, color=COLORS['fg'])

    norm_suffix = " (Normalized)" if normalize else ""
    title = ("🔗 Unified " if do_unify else "📊 Multi-File ") + \
            f"Comparison | X: {x_param.replace('_', ' ').title()}{norm_suffix}"
    ax.set_title(title, color=COLORS['fg'], fontsize=13, fontweight='bold')
    ax.set_xlabel(x_param.replace('_', ' ').title(), color=COLORS['fg'])

    scale_pct = (z_label_mode == "%")
    show_ts_label = "Tokens/s"
    y_label_pp = f"PP Performance (%)" if (normalize and scale_pct) else f"PP {show_ts_label}"
    ax.set_ylabel(y_label_pp, color=pp_base, fontweight='bold')

    has_tg_data = any(show_tg_flags) and series_data['tg']
    ax_tg = ax.twinx() if has_tg_data else None
    if ax_tg:
        ax_tg.set_facecolor(bg)
        ax_tg.grid(False)
        tg_lbl = "TG Performance (%)" if (normalize and scale_pct) else f"TG {show_ts_label}"
        ax_tg.set_ylabel(tg_lbl, color=tg_base, fontweight='bold')

    handles = []
    markers = ['o', 's', '^', 'D', 'v', '<', '>']

    if do_unify:
        # --- Unified mode: average across all enabled files ---
        pp_pts = [p for p in series_data['pp'] if show_pp_flags[p['file_idx']]]
        tg_pts = [p for p in series_data['tg'] if show_tg_flags[p['file_idx']]]

        _draw_unified(ax, pp_pts, pp_base, "Unified PP", '-', handles)
        if ax_tg:
            _draw_unified(ax_tg, tg_pts, tg_base, "Unified TG", '--', handles)
    else:
        # --- Per-file mode ---
        n_files = len(datasets_raw)
        for i in range(n_files):
            pp_c = get_variant_color(pp_base, i)
            tg_c = get_variant_color(tg_base, i)
            mk = markers[i % len(markers)]
            fname = Path(datasets_raw[i]['path']).stem

            pp_pts = sorted(
                [p for p in series_data['pp'] if p['file_idx'] == i],
                key=lambda p: p['x'],
            )
            tg_pts = sorted(
                [p for p in series_data['tg'] if p['file_idx'] == i],
                key=lambda p: p['x'],
            )

            if show_pp_flags[i] and pp_pts:
                ln = ax.errorbar(
                    [p['x'] for p in pp_pts], [p['y'] for p in pp_pts],
                    yerr=[p['err'] for p in pp_pts],
                    label=f"PP: {fname}", color=pp_c, marker=mk,
                    capsize=4, linestyle='-', linewidth=1.5,
                    markersize=6, picker=5,
                )
                handles.append(ln)

            if ax_tg and show_tg_flags[i] and tg_pts:
                ln = ax_tg.errorbar(
                    [p['x'] for p in tg_pts], [p['y'] for p in tg_pts],
                    yerr=[p['err'] for p in tg_pts],
                    label=f"TG: {fname}", color=tg_c, marker=mk,
                    capsize=4, linestyle='--', linewidth=1.5,
                    markersize=6, picker=5,
                )
                handles.append(ln)

    if handles:
        ax.legend(
            handles=handles, loc='upper left',
            facecolor=COLORS['bg'],
            edgecolor=COLORS['accent'],
            labelcolor=COLORS['fg'],
            fontsize=9,
        )

    ax_pp_color = pp_base if any(show_pp_flags) else '#888'
    ax.spines['left'].set_color(ax_pp_color)
    ax.tick_params(axis='y', colors=ax_pp_color)
    ax.tick_params(axis='x', colors=COLORS['fg'])
    ax.spines['bottom'].set_color(COLORS['separator'])
    ax.spines['top'].set_color(COLORS['separator'])
    ax.spines['right'].set_color(COLORS['separator'])

    if ax_tg:
        tg_spine_color = tg_base if any(show_tg_flags) else '#888'
        ax_tg.spines['right'].set_color(tg_spine_color)
        ax_tg.tick_params(axis='y', colors=tg_spine_color)

    fig.tight_layout()
    return fig


def _draw_unified(ax, pts, color, label, linestyle, handles):
    """Helper: collect all per-file points, average by x, draw one line."""
    if not pts:
        return
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for p in pts:
        buckets[p['x']].append((p['y'], p['err']))
    xs, ys, es = [], [], []
    for xv, vals in sorted(buckets.items()):
        means = [v[0] for v in vals]
        errs = [v[1] for v in vals]
        xs.append(xv)
        ys.append(sum(means) / len(means))
        es.append(math.sqrt(sum(e ** 2 for e in errs)) / len(errs))
    ln = ax.errorbar(
        xs, ys, yerr=es, label=label, color=color,
        marker='D', capsize=4, linestyle=linestyle,
        linewidth=2.5, picker=5,
    )
    handles.append(ln)


# ── 3-D Rendering ─────────────────────────────────────────────────────────────

def render_3d(
    points_pp: list[tuple],
    points_tg: list[tuple],
    x_param: str,
    y_param: str,
    pp_color: str,
    tg_color: str,
    dark_mode: bool = True,
    z_label_mode: str = "both-norm",
    show_surface: bool = True,
    show_wireframe: bool = False,
    show_projections: bool = False,
    show_errors_3d: bool = True,
    show_level: bool = False,
    level_val: int = 50,
    surface_style: str = "Solid",
    subdiv_level: int = 0,
    normalized: bool = False,
    pp_min: Optional[float] = None,
    pp_max: Optional[float] = None,
    tg_min: Optional[float] = None,
    tg_max: Optional[float] = None,
) -> tuple[Figure, Any]:
    """
    Build and return a (Figure, Axes3D) pair for the 3-D surface plot.

    Parameters
    ----------
    points_pp / points_tg:
        Lists of (x, y, z, err) tuples — already filtered and optionally
        normalized by the Model. With *normalized* set, both series are
        independently stretched to the full height and share the same
        volume, so they overlap by design; use z_label_mode "pp"/"tg" to
        read the Z axis in absolute units of one series (via the
        *pp_min*/*pp_max*/*tg_min*/*tg_max* statistics).
    normalized:
        Whether the Z values were normalized per series.
    pp_min / pp_max / tg_min / tg_max:
        Pre-normalization Z statistics used for absolute Z tick labels.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    bg = COLORS['bg'] if dark_mode else 'white'
    fig = Figure(figsize=(11, 7), facecolor=bg)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg)

    all_pts = points_pp + points_tg
    all_xs = [p[0] for p in all_pts]
    all_ys = [p[1] for p in all_pts]
    all_zs = [p[2] for p in all_pts]

    max_y_wall = max(all_ys) if all_ys else 0
    min_x_wall = min(all_xs) if all_xs else 0

    cmap_pp = LinearSegmentedColormap.from_list(
        "custom_pp", [COLORS['bg'], pp_color, "#ffffff"]
    )
    cmap_tg = LinearSegmentedColormap.from_list(
        "custom_tg", [COLORS['bg'], tg_color, "#ffffff"]
    )
    light = LightSource(azdeg=315, altdeg=45)

    def plot_series(pts, color, label, marker, cmap):
        if not pts:
            return
        xs, ys, zs, es = zip(*pts)

        # Enhanced 3-D error bars with cross-caps
        if show_errors_3d:
            cap_x = (max(all_xs) - min(all_xs)) * 0.015 if len(set(all_xs)) > 1 else 0.5
            cap_y = (max(all_ys) - min(all_ys)) * 0.015 if len(set(all_ys)) > 1 else 0.5
            for i in range(len(xs)):
                ax.plot(
                    [xs[i], xs[i]], [ys[i], ys[i]],
                    [zs[i] - es[i], zs[i] + es[i]],
                    color=color, alpha=1.0, linewidth=2.5,
                )
                for z_cap in (zs[i] - es[i], zs[i] + es[i]):
                    ax.plot(
                        [xs[i] - cap_x, xs[i] + cap_x],
                        [ys[i], ys[i]], [z_cap, z_cap],
                        color=color, alpha=1.0, linewidth=1.5,
                    )
                    ax.plot(
                        [xs[i], xs[i]],
                        [ys[i] - cap_y, ys[i] + cap_y],
                        [z_cap, z_cap],
                        color=color, alpha=1.0, linewidth=1.5,
                    )

        # Surface / trisurf
        if show_surface and len(pts) >= 3:
            edge_c = 'black' if show_wireframe else 'none'
            lw = 0.5 if show_wireframe else 0
            try:
                x_arr = np.array(xs, dtype=float)
                y_arr = np.array(ys, dtype=float)
                z_arr = np.array(zs, dtype=float)

                if subdiv_level > 0:
                    tri = mtri.Triangulation(x_arr, y_arr)
                    refiner = mtri.UniformTriRefiner(tri)
                    tri_r, z_r = refiner.refine_field(z_arr, subdiv=subdiv_level)
                    _draw_trisurf(ax, tri_r, z_r, None, None, color, cmap,
                                  surface_style, light, edge_c, lw)
                else:
                    _draw_trisurf(ax, None, None, x_arr, y_arr, color, cmap,
                                  surface_style, light, edge_c, lw,
                                  z_arr=z_arr)
            except Exception as exc:
                print(f"[PlotView] Surface fallback: {exc}")
                ax.plot_trisurf(
                    np.array(xs), np.array(ys), np.array(zs),
                    color=color, alpha=0.45, edgecolor=edge_c, linewidth=lw,
                )

        # Scatter points
        ax.scatter(
            xs, ys, zs, c=color, marker=marker, s=60, label=label,
            edgecolors='white', linewidth=0.8, alpha=1.0, depthshade=False,
        )

        # Wall projections
        if show_projections:
            ax.plot(xs, zs, zs=max_y_wall, zdir='y',
                    color=color, linestyle='--', marker=marker, markersize=4, alpha=0.5)
            ax.plot(ys, zs, zs=min_x_wall, zdir='x',
                    color=color, linestyle=':', marker=marker, markersize=4, alpha=0.5)

    plot_series(points_pp, pp_color, "PP", 'o', cmap_pp)
    plot_series(points_tg, tg_color, "TG", 's', cmap_tg)

    # Level plane
    if show_level and all_zs:
        z_min, z_max = min(all_zs), max(all_zs)
        z_plane = z_min + (level_val / 100.0) * (z_max - z_min)
        if len(set(all_xs)) > 1 and len(set(all_ys)) > 1:
            gx, gy = np.meshgrid(
                np.linspace(min(all_xs), max(all_xs), 12),
                np.linspace(min(all_ys), max(all_ys), 12),
            )
            ax.plot_surface(gx, gy, np.full_like(gx, z_plane),
                            color='#8888e8', alpha=0.25, zorder=1)

    # Axis labels
    ax.set_xlabel(x_param.replace('_', ' ').title(), color=COLORS['fg'])
    ax.set_ylabel(y_param.replace('_', ' ').title(), color=COLORS['fg'])
    ax.set_title(
        f"3D Parameter Space | X:{x_param.replace('_', ' ').title()} "
        f"Y:{y_param.replace('_', ' ').title()}",
        color=COLORS['fg'], fontsize=12,
    )

    _set_z_label(ax, z_label_mode, pp_color, tg_color)
    if normalized and z_label_mode in ("pp", "tg"):
        lo, hi = (pp_min, pp_max) if z_label_mode == "pp" else (tg_min, tg_max)
        _set_series_zticks(ax, lo, hi)

    # Legend
    handles = []
    if points_pp:
        handles.append(mlines.Line2D([], [], color=pp_color, marker='o',
                                     label='PP', linestyle='None'))
    if points_tg:
        handles.append(mlines.Line2D([], [], color=tg_color, marker='s',
                                     label='TG', linestyle='None'))
    if handles:
        ax.legend(handles=handles, loc='upper left',
                  facecolor=COLORS['bg'], edgecolor=COLORS['accent'],
                  labelcolor=COLORS['fg'], fontsize=9)

    # Styling
    ax.tick_params(colors=COLORS['fg'])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#444444')
    ax.grid(color='#444444', linestyle='--', alpha=0.3)

    fig.tight_layout()
    return fig, ax


def _draw_trisurf(ax, tri_r, z_r, x_arr, y_arr, color, cmap,
                  style, light, edge_c, lw, z_arr=None):
    """Dispatch helper for trisurf surface styles."""
    if tri_r is not None:
        # Refined triangulation path
        if style == "Shaded":
            ax.plot_trisurf(tri_r, z_r, color=color, alpha=0.8,
                            shade=True, lightsource=light,
                            edgecolor=edge_c, linewidth=lw, antialiased=True)
        elif style == "Colormap":
            ax.plot_trisurf(tri_r, z_r, cmap=cmap, alpha=0.85,
                            shade=True, lightsource=light,
                            edgecolor=edge_c, linewidth=lw, antialiased=True)
        else:  # Solid
            ax.plot_trisurf(tri_r, z_r, color=color, alpha=0.45,
                            edgecolor=edge_c, linewidth=lw, antialiased=True)
    else:
        # Direct array path
        if style == "Solid":
            ax.plot_trisurf(x_arr, y_arr, z_arr, color=color, alpha=0.45,
                            edgecolor=edge_c, linewidth=lw, antialiased=True)
        elif style == "Shaded":
            ax.plot_trisurf(x_arr, y_arr, z_arr, color=color, alpha=0.8,
                            shade=True, lightsource=light,
                            edgecolor=edge_c, linewidth=lw, antialiased=True)
        elif style == "Colormap":
            ax.plot_trisurf(x_arr, y_arr, z_arr, cmap=cmap, alpha=0.85,
                            shade=True, lightsource=light,
                            edgecolor=edge_c, linewidth=lw, antialiased=True)


def _set_series_zticks(ax, lo: Optional[float], hi: Optional[float]) -> None:
    """
    Relabel the Z ticks in absolute units of one normalized series.

    With normalized data every series spans the full [0, 1] height, so
    the raw tick positions are meaningless on their own. The ticks stay
    at their normalized positions but are labeled with the absolute
    values of the selected series (min + t * (max - min)). Does nothing
    when the statistics are missing.
    """
    if lo is None or hi is None:
        return
    if hi == lo:
        # Constant series: a single tick at the level, not six copies.
        ax.set_zticks([1.0])
        ax.set_zticklabels([f"{hi:.4g}"])
        return
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_zticks(ticks)
    ax.set_zticklabels([f"{lo + t * (hi - lo):.4g}" for t in ticks])


def _set_z_label(ax, mode: str, pp_color: str, tg_color: str) -> None:
    """Apply Z-axis label text and color based on the selected mode."""
    labels = {
        "pp":        ("PP Performance",           pp_color),
        "tg":        ("TG Performance",           tg_color),
        "both-norm": ("Normalized (PP & TG)",     '#cccccc'),
        "%":         ("Performance (%)",           '#cccccc'),
    }
    text, color = labels.get(mode, ("Performance", '#cccccc'))
    ax.set_zlabel(text, color=color, fontweight='bold')
    if mode == "%":
        ax.set_zlim(0, 100)