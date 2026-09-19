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
from types import SimpleNamespace
from typing import Any, Callable, Optional

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.lines as mlines
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import art3d
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LightSource, LinearSegmentedColormap, to_rgba
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
        self._pick_cb: Optional[Callable] = None
        self._press_xy: Optional[tuple] = None

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
            Optional callback for data-point clicks (2-D lines and
            3-D scatter). Clicks dispatch manually so every tagged
            artist is considered figure-wide (Matplotlib's built-in
            pick dispatch skips artists whose axes is not the topmost
            one — with twin axes the PP series would never fire) and
            the nearest hit wins instead of the last event. A click
            counts as press+release without dragging, so rotation
            drags never pick or dismiss; clicks with no hit notify
            with None (overlay dismissal).
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

        self._pick_cb = on_pick_cb
        if on_pick_cb:
            self._canvas.mpl_connect('button_press_event', self._on_press)
            self._canvas.mpl_connect('button_release_event', self._on_release)

    def _on_press(self, mouseevent) -> None:
        """Remember press position for click-vs-drag detection."""
        try:
            if getattr(mouseevent, 'button', 1) == 1:
                self._press_xy = (mouseevent.x, mouseevent.y)
            else:
                self._press_xy = None
        except Exception:
            self._press_xy = None

    def _on_release(self, mouseevent) -> None:
        """Dispatch only press+release pairs without dragging."""
        try:
            start = getattr(self, '_press_xy', None)
            self._press_xy = None
            if start is None or getattr(mouseevent, 'button', 1) != 1:
                return
            moved = math.hypot(mouseevent.x - start[0],
                               mouseevent.y - start[1])
            if moved > CLICK_DRAG_TOL_PX:
                return  # rotation/pan drag — no pick, no dismiss
        except Exception:
            return
        self._dispatch_pick(mouseevent)

    def _dispatch_pick(self, mouseevent) -> None:
        """
        Figure-wide nearest-hit click dispatch for data tooltips.

        Tests every line/collection carrying `_llama_records` with its
        own picker and calls the pick callback once for the closest
        hit — or with None when the click landed outside the data
        rectangles, so the presenter can dismiss transient overlays
        (connector line). Clicks on plot data never dismiss, keeping
        rotation drags harmless. Ignores non-left clicks and clicks
        while a toolbar tool (pan/zoom) is active. Never raises.
        """
        cb = self._pick_cb
        if cb is None:
            return
        try:
            if getattr(mouseevent, 'button', 1) != 1:
                return
            if getattr(self._toolbar, 'mode', ''):
                return
            fig = getattr(getattr(mouseevent, 'canvas', None), 'figure', None)
            axes = getattr(fig, 'axes', [])
        except Exception:
            return
        best = None  # (distance, artist, index)
        for ax in axes:
            artists = list(getattr(ax, 'lines', [])) + \
                list(getattr(ax, 'collections', []))
            for artist in artists:
                records = getattr(artist, '_llama_records', None)
                if not isinstance(records, list) or not records:
                    continue
                try:
                    picker = artist.get_picker()
                    if callable(picker):
                        hit, prop = picker(artist, mouseevent)
                    else:
                        hit, prop = artist.contains(mouseevent)
                    if not hit:
                        continue
                    inds = [i for i in list(prop.get('ind', []))
                            if 0 <= i < len(records)]
                    if not inds:
                        continue
                    dist, idx = _nearest_hit_index(artist, inds, mouseevent)
                except Exception:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, artist, idx)
        if best is None:
            # Click without a data hit (empty plot area, decorations,
            # legend, title, figure background): dismiss overlays.
            # Rotation drags never reach dispatch (click-vs-drag gate
            # in _on_release), so keeping the connector is safe.
            try:
                cb(None)
            except Exception:
                pass
            return
        _, artist, idx = best
        try:
            cb(SimpleNamespace(ind=[idx], artist=artist))
        except Exception:
            pass

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
    show_ts: bool = True,
    x_ticks: Optional[list[tuple]] = None,
) -> Figure:
    """
    Build and return a 2-D multi-file comparison Figure.
    The function is stateless: it creates and returns a fresh Figure.

    *x_ticks* is an optional list of (position, label) pairs showing
    the actually measured values instead of automatic decimal ticks.
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

    if x_ticks:
        ax.set_xticks([p for p, _ in x_ticks])
        ax.set_xticklabels([l for _, l in x_ticks])

    scale_pct = (z_label_mode == "%")
    show_ts_label = "Tokens/s" if show_ts else "Time (ns)"
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

        _draw_unified(ax, pp_pts, pp_base, "Unified PP", '-', handles, "pp")
        if ax_tg:
            _draw_unified(ax_tg, tg_pts, tg_base, "Unified TG", '--', handles, "tg")
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
                _attach_records(ln, pp_pts, f"PP: {fname}", "pp")

            if ax_tg and show_tg_flags[i] and tg_pts:
                ln = ax_tg.errorbar(
                    [p['x'] for p in tg_pts], [p['y'] for p in tg_pts],
                    yerr=[p['err'] for p in tg_pts],
                    label=f"TG: {fname}", color=tg_c, marker=mk,
                    capsize=4, linestyle='--', linewidth=1.5,
                    markersize=6, picker=5,
                )
                handles.append(ln)
                _attach_records(ln, tg_pts, f"TG: {fname}", "tg")

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


def _point_record(p: dict, label: Optional[str] = None,
                  series: Optional[str] = None) -> dict:
    """
    Tooltip payload for one 2-D series point (drawn order).

    The series label rides along because errorbar() keeps it on the
    container (for the legend) while the pickable data line itself
    stays at the default "_no_legend_". 'series' ('pp'/'tg') and
    'file_idx' allow the presenter to find the counterpart point of
    the other series for combined tooltips.
    """
    rec = {
        'x': p.get('x'),
        'ts': p.get('ts'),
        'ts_err': p.get('ts_err', 0.0),
        'ns': p.get('ns'),
        'ns_err': p.get('ns_err', 0.0),
    }
    if label is None:
        label = p.get('label')
    if label is not None:
        rec['label'] = label
    if series is None:
        series = p.get('series')
    if series is not None:
        rec['series'] = series
    if p.get('file_idx') is not None:
        rec['file_idx'] = p.get('file_idx')
    return rec


# Click-vs-drag tolerance: press+release pairs moving less than this
# (screen pixels) count as clicks; anything beyond is a rotation drag.
CLICK_DRAG_TOL_PX = 5.0


def _nearest_hit_index(artist, inds: list, mouseevent) -> tuple:
    """
    Closest hit index (and its screen distance) among candidate indices.

    Only Line2D artists support exact measurement via their transform;
    anything else keeps candidate order with distance 0 (correct
    whenever a single artist type is involved). Empty input yields
    (inf, -1) so it never wins a comparison.
    """
    inds = list(inds or [])
    if not inds:
        return float('inf'), -1
    try:
        if isinstance(artist, mlines.Line2D):
            xy = np.asarray(artist.get_xydata(), dtype=float)[inds]
            pts = artist.get_transform().transform(xy)
            dist = np.hypot(pts[:, 0] - mouseevent.x,
                            pts[:, 1] - mouseevent.y)
            best = int(np.argmin(dist))
            return float(dist[best]), int(inds[best])
    except Exception:
        pass
    return 0.0, int(inds[0])


def _marker_only_picker(artist, mouseevent, tol=8.0):
    """
    Pick callback accepting hits near actual markers only.

    The default Line2D picking also fires on connecting segments far
    from any measurement, which lets one series steal clicks meant for
    the other (with last-event-wins the steal is systematic). Returns
    (hit, {'ind': ...}) like contains().
    """
    try:
        data = artist.get_xydata()
        if data is None or len(data) == 0:
            return False, {}
        pts = artist.get_transform().transform(np.asarray(data, dtype=float))
        dist = np.hypot(pts[:, 0] - mouseevent.x, pts[:, 1] - mouseevent.y)
        hits = np.nonzero(dist <= tol)[0]
        if len(hits) == 0:
            return False, {}
        return True, {'ind': hits}
    except Exception:
        return False, {}


def _disable_pick_recursive(artist) -> None:
    """Turn picking off for an artist and any nested children."""
    if isinstance(artist, (list, tuple)):
        for child in artist:
            _disable_pick_recursive(child)
        return
    try:
        artist.set_picker(False)
    except (AttributeError, TypeError):
        pass


def _attach_records(container, pts: list[dict], label: Optional[str] = None,
                    series: Optional[str] = None) -> None:
    """
    Stash per-point tooltip records on the drawn data line, in drawn
    order, so the pick handler can show values without reverse lookup.
    Picking is restricted to markers so clicks near connecting segments
    cannot steal tooltips from the other series. Best effort — never
    raises.
    """
    try:
        data_line = container[0]
    except (IndexError, TypeError, AttributeError):
        return
    try:
        data_line._llama_records = [_point_record(p, label, series) for p in pts]
    except (AttributeError, TypeError):
        pass
    try:
        data_line.set_picker(_marker_only_picker)
    except (AttributeError, TypeError):
        pass
    try:
        for child in container.get_children():
            if child is not data_line:
                _disable_pick_recursive(child)
    except (AttributeError, TypeError):
        pass


def average_bucket(xv, members: list[dict]) -> tuple:
    """
    Average one x-bucket into drawn values plus a tooltip record.

    Returns (y, err, record) where err uses RMS combination (same as
    the drawn error bar) and the record carries averaged ts/ns with
    RMS-combined errors, so the tooltip explains the bar. Shared with
    the presenter, which averages unified counterparts the same way.
    """
    means = [v['y'] for v in members]
    errs = [v['err'] for v in members]
    rec: dict = {'x': xv}
    for key in ('ts', 'ts_err', 'ns', 'ns_err'):
        vals = [v[key] for v in members if v.get(key) is not None]
        if key.endswith('_err'):
            rec[key] = (math.sqrt(sum(e ** 2 for e in vals)) / len(vals)
                        if vals else 0.0)
        else:
            rec[key] = sum(vals) / len(vals) if vals else None
    y = sum(means) / len(means)
    err = math.sqrt(sum(e ** 2 for e in errs)) / len(errs)
    return y, err, rec


def _draw_unified(ax, pts, color, label, linestyle, handles, series=None):
    """
    Helper: collect all per-file points, average by x, draw one line.
    Returns tooltip records in drawn order (averaged ts/ns included).
    *series* ('pp'/'tg') is stored in the records; derived from the
    label when omitted (production callers pass it explicitly).
    """
    if not pts:
        return []
    if series is None:
        series = 'pp' if 'PP' in (label or '') else 'tg'
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for p in pts:
        buckets[p['x']].append(p)
    xs, ys, es, records = [], [], [], []
    for xv, members in sorted(buckets.items()):
        y, err, rec = average_bucket(xv, members)
        xs.append(xv)
        ys.append(y)
        es.append(err)
        rec['label'] = label
        rec['series'] = series
        records.append(rec)
    ln = ax.errorbar(
        xs, ys, yerr=es, label=label, color=color,
        marker='D', capsize=4, linestyle=linestyle,
        linewidth=2.5, picker=5,
    )
    handles.append(ln)
    _attach_records(ln, records, label)
    return records


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
    interp_method: str = "Cubic",
    clamp_surface: bool = False,
    mask_gaps: bool = False,
    normalized: bool = False,
    x_ticks: Optional[list[tuple]] = None,
    y_ticks: Optional[list[tuple]] = None,
    pp_min: Optional[float] = None,
    pp_max: Optional[float] = None,
    tg_min: Optional[float] = None,
    tg_max: Optional[float] = None,
    infos_pp: Optional[list] = None,
    infos_tg: Optional[list] = None,
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
    interp_method:
        Refinement interpolator for subdiv_level > 0: "Cubic" (smooth
        Clough-Tocher, may overshoot between sparse points) or "Linear"
        (piecewise linear, stays within the measured range). The
        "Cubic+Clamp" toolbar choice maps to Cubic plus *clamp_surface*.
    clamp_surface:
        Clamp the refined field to the measured z range (anti-overshoot).
    mask_gaps:
        Drop triangles spanning unmeasured parameter gaps: the sorted
        longest-edges are cut at the biggest relative jump (see
        _mask_gap_triangles), so unmeasured regions stay open instead
        of being bridged. Limitation: only the single largest
        discontinuity is cut, so with several gaps of different sizes
        the smaller bridges are retained.

    Both surfaces are drawn as a single merged Poly3DCollection so
    overlapping triangles are depth-sorted against each other.
    normalized:
        Whether the Z values were normalized per series.
    pp_min / pp_max / tg_min / tg_max:
        Pre-normalization Z statistics used for absolute Z tick labels.
    infos_pp / infos_tg:
        Optional per-point tooltip payloads aligned 1:1 with the point
        lists; attached to the scatter artists for click dispatch.
    x_ticks / y_ticks:
        Optional (position, label) pairs showing the actually measured
        values on X/Y instead of automatic decimal ticks.
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
    surfaces: list = []  # (verts, colors, edge, lw) per part, drawn merged below
    edge_c = 'black' if show_wireframe else 'none'
    lw = 0.5 if show_wireframe else 0

    def plot_series(pts, color, label, marker, cmap, infos=None):
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

        # Surface triangulation — collected here, drawn once below as a
        # single merged collection (see _draw_merged_surfaces).
        if show_surface and len(pts) >= 3:
            tri = _triangulate_series(
                xs, ys, zs, subdiv_level, interp_method,
                clamp_surface=clamp_surface, mask_gaps=mask_gaps,
            )
            if tri is not None:
                triangles, tx, ty, tz = tri
                verts, colors = _series_face_colors(
                    triangles, tx, ty, tz,
                    color, cmap, surface_style, light,
                )
                surfaces.append((verts, colors, edge_c, lw))

        # Scatter points (picker enabled for 3-D tooltips; records ride
        # along for figure-wide click dispatch)
        sc = ax.scatter(
            xs, ys, zs, c=color, marker=marker, s=60, label=label,
            edgecolors='white', linewidth=0.8, alpha=1.0, depthshade=False,
            picker=5,
        )
        if infos:
            sc._llama_records = list(infos)

        # Wall projections
        if show_projections:
            ax.plot(xs, zs, zs=max_y_wall, zdir='y',
                    color=color, linestyle='--', marker=marker, markersize=4, alpha=0.5)
            ax.plot(ys, zs, zs=min_x_wall, zdir='x',
                    color=color, linestyle=':', marker=marker, markersize=4, alpha=0.5)

    plot_series(points_pp, pp_color, "PP", 'o', cmap_pp, infos_pp)
    plot_series(points_tg, tg_color, "TG", 's', cmap_tg, infos_tg)

    # One joint surface collection so overlapping PP/TG triangles — and
    # the level plane below — are depth-sorted against each other
    # instead of one surface covering the other as a whole unit.
    if show_level and all_zs:
        z_min, z_max = min(all_zs), max(all_zs)
        z_plane = z_min + (level_val / 100.0) * (z_max - z_min)
        if len(set(all_xs)) > 1 and len(set(all_ys)) > 1:
            # Grid density follows the surface subdivision for
            # comparable triangle sizes at the crossing lines.
            divisions = min(LEVEL_BASE_DIVISIONS * (subdiv_level + 1),
                            LEVEL_MAX_DIVISIONS)
            verts, colors = _level_plane_surface(
                min(all_xs), max(all_xs), min(all_ys), max(all_ys),
                z_plane, divisions,
            )
            surfaces.append((verts, colors, 'none', 0.0))

    _draw_merged_surfaces(ax, surfaces)

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

    # Measured value ticks (the presenter passes categorical labels
    # instead for string-valued dimensions).
    if x_ticks:
        ax.set_xticks([p for p, _ in x_ticks])
        ax.set_xticklabels([l for _, l in x_ticks])
    if y_ticks:
        ax.set_yticks([p for p, _ in y_ticks])
        ax.set_yticklabels([l for _, l in y_ticks])

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


# Triangles past an edge-length jump of this ratio are treated as
# Delaunay bridges across unmeasured parameter gaps (see below).
GAP_EDGE_FACTOR = 2.0


def _triangulate_series(
    xs, ys, zs, subdiv_level, interp_method="Cubic",
    clamp_surface=False, mask_gaps=False,
):
    """
    Delaunay-triangulate one series for the merged 3-D surface.

    Returns (triangles, x, y, z) where triangles is an (n, 3) index
    array into the coordinate arrays (refined when subdiv_level > 0),
    or None when triangulation is impossible (fewer than 3 points,
    degenerate geometry, or everything masked as gap). Never raises —
    failures print a note and yield None, leaving that series as
    scatter points only.

    Refinement uses "Cubic" (smooth, may overshoot between sparse
    points) or "Linear" (piecewise linear, no overshoot interpolation);
    any other value falls back to Cubic. With *clamp_surface* the
    refined field is limited to the measured z range so no surface can
    leave it, whichever method is chosen. With *mask_gaps* long
    bridging triangles over unmeasured parameter gaps are dropped
    (see _mask_gap_triangles).
    """
    try:
        x_arr = np.asarray(list(xs), dtype=float)
        y_arr = np.asarray(list(ys), dtype=float)
        z_arr = np.asarray(list(zs), dtype=float)
    except (ValueError, TypeError) as exc:
        print(f"[PlotView] Triangulation skipped: {exc}")
        return None
    if len(x_arr) < 3:
        return None
    try:
        if subdiv_level > 0:
            base = mtri.Triangulation(x_arr, y_arr)
            refiner = mtri.UniformTriRefiner(base)
            if interp_method == "Linear":
                interp = mtri.LinearTriInterpolator(base, z_arr)
            else:
                interp = mtri.CubicTriInterpolator(base, z_arr)
            tri, z_ref = refiner.refine_field(
                z_arr, triinterpolator=interp, subdiv=subdiv_level)
            z_ref = np.asarray(z_ref, dtype=float)
            if clamp_surface:
                z_ref = _clamp_field(
                    z_ref, float(np.min(z_arr)), float(np.max(z_arr)))
            triangles, tx, ty = tri.triangles, tri.x, tri.y
        else:
            tri = mtri.Triangulation(x_arr, y_arr)
            triangles, tx, ty, z_ref = tri.triangles, x_arr, y_arr, z_arr
        if mask_gaps:
            triangles = _mask_gap_triangles(triangles, tx, ty)
            if len(triangles) == 0:
                return None
        return triangles, tx, ty, z_ref
    except Exception as exc:
        print(f"[PlotView] Triangulation skipped: {exc}")
        return None


def _mask_gap_triangles(triangles, x, y, factor=GAP_EDGE_FACTOR):
    """
    Drop triangles spanning unmeasured parameter gaps.

    Edge lengths are measured in per-axis-normalized x-y units (each
    axis scaled to [0, 1], so differently scaled parameters compare
    fairly). The sorted longest-edges are scanned for the biggest
    relative jump: bridging triangles are orders of magnitude longer
    than intra-cluster ones, while legitimate meshes grow smoothly.
    Only when that jump exceeds *factor* is everything above it
    dropped; otherwise the mesh is kept whole. Limitation: a single
    cut at the largest jump — with several gaps of different sizes,
    smaller bridges past further jumps are retained.
    """
    tris = np.asarray(triangles)
    if len(tris) == 0:
        return tris
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    x_span = float(np.max(xa) - np.min(xa)) or 1.0
    y_span = float(np.max(ya) - np.min(ya)) or 1.0
    xn = (xa - np.min(xa)) / x_span
    yn = (ya - np.min(ya)) / y_span
    px, py = xn[tris], yn[tris]
    edges = np.stack([
        np.hypot(px[:, 0] - px[:, 1], py[:, 0] - py[:, 1]),
        np.hypot(px[:, 1] - px[:, 2], py[:, 1] - py[:, 2]),
        np.hypot(px[:, 2] - px[:, 0], py[:, 2] - py[:, 0]),
    ], axis=1)
    longest = edges.max(axis=1)
    ordered = np.sort(longest)
    positive = ordered[ordered > 0]
    if len(positive) < 2:
        return tris
    ratios = positive[1:] / positive[:-1]
    jump = int(np.argmax(ratios))
    if ratios[jump] <= factor:
        return tris
    return tris[longest <= positive[jump]]


def _clamp_field(z_values, z_min: float, z_max: float):
    """Clamp refined z values to the measured range (anti-overshoot)."""
    return np.clip(np.asarray(z_values, dtype=float), z_min, z_max)


def interp_surface_z(xs, ys, zs, x: float, y: float) -> Optional[float]:
    """
    Linearly interpolated surface height at (x, y) from scattered points.

    Returns None when interpolation is impossible (fewer than 3 points,
    degenerate geometry, non-finite result, or (x, y) outside the
    triangulated hull — masked values). Never raises.
    """
    try:
        x_arr = np.asarray(list(xs), dtype=float)
        y_arr = np.asarray(list(ys), dtype=float)
        z_arr = np.asarray(list(zs), dtype=float)
        if len(x_arr) < 3:
            return None
        tri = mtri.Triangulation(x_arr, y_arr)
        interp = mtri.LinearTriInterpolator(tri, z_arr)
        z = interp(float(x), float(y))
    except Exception:
        return None
    try:
        if np.ma.is_masked(z):
            return None
        value = float(z)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def _series_face_colors(triangles, x, y, z, color, cmap, style, light):
    """
    Per-face RGBA colors for one triangulated series.

    Returns (verts, colors): verts is an (n, 3, 3) array of triangle
    corners, colors an (n, 4) array. Solid uses the base color, Shaded
    modulates it by face-normal lighting, Colormap maps the series' own
    z range through *cmap*. Alphas match the former per-series surfaces
    (0.45 / 0.8 / 0.85).
    """
    vx, vy, vz = x[triangles], y[triangles], z[triangles]
    verts = np.stack([vx, vy, vz], axis=-1)
    n = len(triangles)
    if style == "Colormap":
        zmin, zmax = float(np.min(z)), float(np.max(z))
        span = zmax - zmin
        t = np.full(n, 0.5) if span == 0 else (vz.mean(axis=1) - zmin) / span
        colors = np.asarray(cmap(t), dtype=float).reshape(n, 4)
        colors[:, 3] = 0.85
        return verts, colors
    base = np.array(to_rgba(color), dtype=float)[:3]
    if style == "Shaded":
        e1 = verts[:, 1] - verts[:, 0]
        e2 = verts[:, 2] - verts[:, 0]
        normals = np.cross(e1, e2)
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        intensity = light.shade_normals(normals / norm, fraction=1.0)
        shaded = np.clip(base[None, :] * intensity[:, None], 0.0, 1.0)
        return verts, np.column_stack([shaded, np.full(n, 0.8)])
    return verts, np.column_stack([np.tile(base, (n, 1)), np.full(n, 0.45)])


def _draw_merged_surfaces(ax, surfaces):
    """
    Draw every collected part as ONE
    Poly3DCollection so Matplotlib's painter algorithm depth-sorts
    every triangle jointly.

    Each entry in *surfaces* is a (verts, colors, edge, lw) tuple;
    per-face edge colors and widths let the level plane stay edge-free
    while data surfaces honor the Wire toggle. Separate collections
    are only sorted as whole units, which puts one surface entirely
    above the other and flips that order while rotating. Returns the
    collection, or None when there is nothing to draw.
    """
    parts = [(v, c, e, w) for v, c, e, w in surfaces
             if v is not None and len(v)]
    if not parts:
        return None
    verts = np.vstack([v for v, _, _, _ in parts])
    colors = np.vstack([c for _, c, _, _ in parts])
    edge_colors = np.vstack([
        np.tile(to_rgba(edge), (len(v), 1)) for v, _, edge, _ in parts
    ])
    line_widths = np.concatenate([
        np.full(len(v), w, dtype=float) for v, _, _, w in parts
    ])
    coll = art3d.Poly3DCollection(
        verts, facecolors=colors, edgecolors=edge_colors,
        linewidths=line_widths, zsort='average',
    )
    ax.add_collection3d(coll)
    return coll


# At most this many value ticks are placed on an axis; denser value
# sets are thinned evenly so labels stay readable.
MAX_VALUE_TICKS = 12


def thin_value_ticks(values, max_ticks=MAX_VALUE_TICKS):
    """
    Unique sorted axis values thinned to at most *max_ticks* entries.

    Returns a list of (position, label) pairs with compact labels
    ("%g" for numbers), thinned to at most *max_ticks* entries while
    always keeping the maximum measured value. Returns None when the
    values are unusable — empty, mutually unsortable mixed types, or
    strings (string dimensions keep Matplotlib's categorical ticks;
    numeric tick positions are required). The caller then keeps
    automatic ticks.
    """
    try:
        ordered = sorted(set(values))
    except TypeError:
        return None
    if not ordered:
        return None
    if any(isinstance(v, str) for v in ordered):
        return None
    unique = ordered
    if len(unique) > max_ticks:
        step = (len(unique) + max_ticks - 1) // max_ticks
        unique = unique[::step]
        if unique[-1] != ordered[-1]:
            unique.append(ordered[-1])
    return [(v, f"{v:g}") for v in unique]


# Level-plane appearance and grid density (divisions per side at
# subdiv_level 0; scaled up with subdivision, capped).
LEVEL_PLANE_COLOR = '#8888e8'
LEVEL_PLANE_ALPHA = 0.25
LEVEL_BASE_DIVISIONS = 12
LEVEL_MAX_DIVISIONS = 48


def _level_plane_surface(x0, x1, y0, y1, z_plane, divisions):
    """
    Level-plane triangles for the merged 3-D collection.

    Regular grid of divisions×divisions quads (two triangles each) at
    constant z, in translucent LEVEL_PLANE_COLOR. Returns
    (verts, colors) with verts shaped (n, 3, 3).
    """
    gx, gy = np.meshgrid(np.linspace(x0, x1, divisions + 1),
                         np.linspace(y0, y1, divisions + 1))
    gz = np.full_like(gx, z_plane, dtype=float)
    flat_x, flat_y, flat_z = gx.ravel(), gy.ravel(), gz.ravel()
    nx = divisions + 1
    tris = []
    for iy in range(divisions):
        for ix in range(divisions):
            a = iy * nx + ix
            b, c, d = a + 1, a + nx, a + nx + 1
            tris.append((a, b, d))
            tris.append((a, d, c))
    triangles = np.array(tris)
    verts = np.stack(
        [flat_x[triangles], flat_y[triangles], flat_z[triangles]], axis=-1)
    base = np.array(to_rgba(LEVEL_PLANE_COLOR), dtype=float)[:3]
    colors = np.column_stack(
        [np.tile(base, (len(triangles), 1)),
         np.full(len(triangles), LEVEL_PLANE_ALPHA)])
    return verts, colors


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