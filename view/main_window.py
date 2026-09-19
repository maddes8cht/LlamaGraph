"""
view/main_window.py

MainWindow — the top-level Tkinter window and upper toolbar for llamagraph.

Responsibilities:
  - Create the root Tk window
  - Build the PanedWindow that holds left_sidebar | plot | right_sidebar
  - Build the upper toolbar row (PP/TG global toggles, Unify, Norm, 3D switch,
    metric toggle)
  - Build the lower 3-D settings toolbar (always visible for pre-configuration)
  - Expose all Tkinter variables to the Presenter via properties
  - Wire all toolbar callbacks to the Presenter

The MainWindow does NOT contain data logic; it only wires UI to Presenter callbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk

from utils.colors import COLORS, DEFAULT_PP_COLOR, DEFAULT_TG_COLOR
from view.left_sidebar import LeftSidebar
from view.right_sidebar import RightSidebar
from view.plot_view import PlotView


class MainWindow:
    """
    Top-level application window.

    After construction, the Presenter receives references to:
      self.left_sidebar  – LeftSidebar
      self.right_sidebar – RightSidebar
      self.plot_view     – PlotView

    All Tkinter vars are exposed as properties so the Presenter can
    read them without knowing about Tkinter internals.
    """

    def __init__(
        self,
        root: tk.Tk,
        pp_color: str = DEFAULT_PP_COLOR,
        tg_color: str = DEFAULT_TG_COLOR,
    ) -> None:
        self._root = root
        self._pp_color = pp_color
        self._tg_color = tg_color

        # ── Tkinter state variables ──────────────────────────────────────────
        self._show_pp = tk.IntVar(value=1)
        self._show_tg = tk.IntVar(value=1)
        self._unify_var = tk.IntVar(value=0)
        self._normalize_var = tk.IntVar(value=0)
        self._mode_3d = tk.IntVar(value=0)

        # 3-D specific vars
        self._show_surface = tk.IntVar(value=1)
        self._surface_style_var = tk.StringVar(value="Solid")
        self._show_wireframe_var = tk.IntVar(value=0)
        self._show_projections_var = tk.IntVar(value=0)
        self._show_errors_3d = tk.IntVar(value=1)
        self._z_label_mode = tk.StringVar(value="both-norm")
        self._show_level_var = tk.IntVar(value=0)
        self._level_val_var = tk.IntVar(value=50)
        self._subdiv_var = tk.IntVar(value=0)
        self._interp_var = tk.StringVar(value="Cubic")
        self._mask_var = tk.IntVar(value=0)
        self._axis_x_var = tk.StringVar()
        self._axis_y_var = tk.StringVar()

        # ── Build UI ─────────────────────────────────────────────────────────
        self._configure_root()
        self._build_layout()

    # ── Root window setup ─────────────────────────────────────────────────────

    def _configure_root(self) -> None:
        self._root.title("llamagraph — llama-bench Visualizer")
        self._root.geometry("1680x900")
        self._root.configure(bg=COLORS['bg'])
        self._root.minsize(1200, 650)
        # Style Combobox dropdowns dark
        self._root.option_add('*TCombobox*Listbox.background', '#2d2d2d')
        self._root.option_add('*TCombobox*Listbox.foreground', COLORS['fg'])

    def _build_layout(self) -> None:
        # Main paned container: left sidebar | center (toolbars + plot) | right sidebar
        self._paned = ttk.PanedWindow(self._root, orient=tk.HORIZONTAL)
        self._paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left sidebar
        self.left_sidebar = LeftSidebar(self._paned, pp_color=self._pp_color,
                                        tg_color=self._tg_color, width=480)
        self._paned.add(self.left_sidebar, weight=0)

        # Center pane: toolbars above, plot canvas below
        center = tk.Frame(self._paned, bg=COLORS['bg'])
        self._paned.add(center, weight=1)
        self._build_upper_toolbar(center)
        self._build_3d_toolbar(center)

        self.plot_view = PlotView(center)
        self.plot_view.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        # Right sidebar
        self.right_sidebar = RightSidebar(self._paned, width=240)
        self._paned.add(self.right_sidebar, weight=0)

    # ── Upper toolbar ─────────────────────────────────────────────────────────

    def _build_upper_toolbar(self, parent: tk.Widget) -> None:
        """Build the row with PP/TG global toggles, Unify, Norm, 3D, metric."""
        bar = tk.Frame(parent, bg=COLORS['bg'])
        bar.pack(fill=tk.X, padx=10, pady=(10, 0))

        def chk(text, var, color, cmd, **kw):
            return tk.Checkbutton(
                bar, text=text, variable=var, command=cmd,
                bg=COLORS['bg'], fg=color,
                selectcolor=COLORS['checkbox_active'],
                font=('Segoe UI', 9, 'bold'), **kw
            )

        chk(" PP (All)", self._show_pp, self._pp_color, self._on_render).pack(
            side=tk.LEFT, padx=5)
        chk(" TG (All)", self._show_tg, self._tg_color, self._on_render).pack(
            side=tk.LEFT, padx=5)

        self._unify_chk = tk.Checkbutton(
            bar, text=" 🔗 Unify", variable=self._unify_var,
            command=self._on_render,
            bg=COLORS['bg'], fg='#9cdcfe',
            selectcolor=COLORS['checkbox_active'],
            font=('Segoe UI', 9),
        )
        self._unify_chk.pack(side=tk.LEFT, padx=5)

        self._toggle_btn = tk.Button(
            bar, text="t/s → time",
            bg=COLORS['accent'], fg='white', relief=tk.FLAT,
            cursor='hand2', font=('Segoe UI', 9),
        )
        self._toggle_btn.pack(side=tk.LEFT, padx=(20, 5))

        tk.Checkbutton(
            bar, text=" 📐 Norm", variable=self._normalize_var,
            command=self._on_render,
            bg=COLORS['bg'], fg='#b5cea8',
            selectcolor=COLORS['checkbox_active'],
            font=('Segoe UI', 9),
        ).pack(side=tk.LEFT, padx=5)

        tk.Checkbutton(
            bar, text=" 🧊 3D", variable=self._mode_3d,
            command=self._on_toggle_3d,
            bg=COLORS['bg'], fg='#ce9178',
            selectcolor=COLORS['checkbox_active'],
            font=('Segoe UI', 9),
        ).pack(side=tk.LEFT, padx=10)

        chk(" Lev", self._show_level_var, '#9cdcfe', self._on_render).pack(
            side=tk.LEFT, padx=5)
        self._btn_level_down = tk.Button(
            bar, text="◀", command=lambda: self._on_level_step(-1),
            bg='#3a3a3a', fg=COLORS['fg'],
            relief=tk.FLAT, cursor='hand2',
            font=('Segoe UI', 8), width=3,
            activebackground=COLORS['accent'], activeforeground='white',
        )
        self._btn_level_down.pack(side=tk.LEFT, padx=1)
        self._ent_level = tk.Entry(
            bar, width=4, justify='center',
            bg='#2d2d2d', fg=COLORS['fg'], insertbackground=COLORS['fg'],
            relief=tk.FLAT, font=('Consolas', 9),
        )
        self._ent_level.insert(0, str(self._level_val_var.get()))
        self._ent_level.bind('<Return>', lambda _e: self._commit_level_entry())
        self._ent_level.bind('<FocusOut>', lambda _e: self._commit_level_entry())
        self._ent_level.pack(side=tk.LEFT, padx=1)
        self._btn_level_up = tk.Button(
            bar, text="▶", command=lambda: self._on_level_step(1),
            bg='#3a3a3a', fg=COLORS['fg'],
            relief=tk.FLAT, cursor='hand2',
            font=('Segoe UI', 8), width=3,
            activebackground=COLORS['accent'], activeforeground='white',
        )
        self._btn_level_up.pack(side=tk.LEFT, padx=(1, 5))

    # ── 3-D settings toolbar ──────────────────────────────────────────────────

    def _build_3d_toolbar(self, parent: tk.Widget) -> None:
        """
        The lower toolbar row with 3-D-specific controls.
        Always packed (not hidden) so users can pre-configure before enabling 3-D.
        """
        bar = tk.Frame(parent, bg=COLORS['bg'])
        bar.pack(fill=tk.X, padx=10, pady=(5, 0))
        self._3d_toolbar_frame = bar

        def lbl(text):
            return tk.Label(bar, text=text, bg=COLORS['bg'], fg=COLORS['fg'],
                            font=('Segoe UI', 9))

        def combo(var, values, width, bind_render=True):
            cb = ttk.Combobox(bar, textvariable=var, state='readonly',
                              width=width, font=('Segoe UI', 8), values=values)
            if bind_render:
                cb.bind('<<ComboboxSelected>>', lambda _e: self._on_render())
            return cb

        def chk(text, var, color):
            return tk.Checkbutton(
                bar, text=text, variable=var, command=self._on_render,
                bg=COLORS['bg'], fg=color,
                selectcolor=COLORS['checkbox_active'],
                font=('Segoe UI', 9),
            )

        # X / Y axis selectors
        lbl("X:").pack(side=tk.LEFT, padx=2)
        self._cb_x = combo(self._axis_x_var, [], width=12)
        self._cb_x.pack(side=tk.LEFT, padx=2)

        lbl("Y:").pack(side=tk.LEFT, padx=2)
        self._cb_y = combo(self._axis_y_var, [], width=12)
        self._cb_y.pack(side=tk.LEFT, padx=2)

        # Z-label mode
        self._cb_z = combo(self._z_label_mode,
                           ["pp", "tg", "both-norm", "%"], width=9)
        self._cb_z.pack(side=tk.LEFT, padx=2)

        # Surface toggles
        chk(" Surf", self._show_surface, '#dcdcaa').pack(side=tk.LEFT, padx=2)
        combo(self._surface_style_var, ["Solid", "Shaded", "Colormap"], width=9).pack(
            side=tk.LEFT, padx=2)

        lbl(" SubDiv:").pack(side=tk.LEFT, padx=2)
        combo(self._subdiv_var, [0, 1, 2, 3, 4], width=3).pack(side=tk.LEFT, padx=2)

        lbl(" Interp:").pack(side=tk.LEFT, padx=2)
        # No "Linear+Clamp": linear interpolation cannot leave the
        # measured range, so clamping would be a no-op for it.
        combo(self._interp_var, ["Cubic", "Cubic+Clamp", "Linear"], width=12).pack(side=tk.LEFT, padx=2)

        chk(" Mask", self._mask_var, '#cccccc').pack(side=tk.LEFT, padx=2)

        chk(" Wire", self._show_wireframe_var, '#cccccc').pack(side=tk.LEFT, padx=2)
        chk(" Err", self._show_errors_3d, '#f44747').pack(side=tk.LEFT, padx=2)
        chk(" Proj", self._show_projections_var, '#569cd6').pack(side=tk.LEFT, padx=2)

        # Store controls that need state management
        self._3d_controls = [
            self._cb_x, self._cb_y, self._cb_z,
        ]

    # ── Callback stubs — replaced by Presenter ───────────────────────────────

    def _on_render(self) -> None:
        if self._render_cb:
            self._render_cb()

    def _on_toggle_3d(self) -> None:
        if self._toggle_3d_cb:
            self._toggle_3d_cb()

    def _on_level_step(self, delta: int) -> None:
        """Nudge the level value by *delta* (±1 via the arrow buttons)."""
        self._set_level_value(self._level_val_var.get() + delta)

    def _commit_level_entry(self) -> None:
        """Take over a manually typed level value (clamped to 0-100)."""
        try:
            value = int(self._ent_level.get().strip())
        except (ValueError, TypeError, AttributeError):
            value = self._level_val_var.get()
        self._set_level_value(value)

    def _set_level_value(self, value: int) -> None:
        """
        Store a clamped level value and sync the entry.

        Re-renders only when the value actually changed: stepping
        against a bound, focus passthrough, or garbage revert must not
        rebuild the (potentially subdivided) 3-D figure for nothing.
        TclError-safe for teardown while the entry has focus.
        """
        try:
            entry_text = self._ent_level.get()
        except tk.TclError:
            return  # widget gone (teardown) — nothing to sync
        clamped = max(0, min(100, int(value)))
        changed = (clamped != self._level_val_var.get())
        self._level_val_var.set(clamped)
        if entry_text.strip() != str(clamped):
            try:
                self._ent_level.delete(0, tk.END)
                self._ent_level.insert(0, str(clamped))
            except tk.TclError:
                return
        if changed:
            self._on_render()

    # Presenter injects these
    _render_cb: Optional[Callable] = None
    _toggle_3d_cb: Optional[Callable] = None

    def set_render_callback(self, cb: Callable) -> None:
        self._render_cb = cb

    def set_toggle_3d_callback(self, cb: Callable) -> None:
        self._toggle_3d_cb = cb

    def set_toggle_metric_callback(self, cb: Callable) -> None:
        self._toggle_btn.config(command=cb)

    def set_key_bindings(
        self,
        toggle_metric: Callable,
        refresh: Callable,
        quit_app: Callable,
    ) -> None:
        self._root.bind('<Control-t>', lambda _e: toggle_metric())
        self._root.bind('<Control-r>', lambda _e: refresh())
        self._root.bind('<Escape>', lambda _e: quit_app())

    # ── Public state: axis comboboxes ─────────────────────────────────────────

    def update_axis_choices(self, params: list[str]) -> None:
        """
        Update the X/Y axis combobox options.
        Preserves current selection if still valid; otherwise auto-selects.
        """
        state = 'readonly' if params else 'disabled'
        self._cb_x.config(values=params, state=state)
        self._cb_y.config(values=params, state=state)

        cur_x = self._axis_x_var.get()
        cur_y = self._axis_y_var.get()

        if not cur_x or cur_x not in params:
            self._axis_x_var.set(params[0] if params else '')
        if not cur_y or cur_y not in params or cur_y == self._axis_x_var.get():
            other = next((p for p in params if p != self._axis_x_var.get()), '')
            self._axis_y_var.set(other)

    def set_unify_state(self, enabled: bool) -> None:
        self._unify_chk.config(state=tk.NORMAL if enabled else tk.DISABLED)

    def set_metric_button_text(self, text: str) -> None:
        self._toggle_btn.config(text=text)

    # ── Property accessors (read by Presenter) ────────────────────────────────

    @property
    def show_pp(self) -> bool:
        return bool(self._show_pp.get())

    @property
    def show_tg(self) -> bool:
        return bool(self._show_tg.get())

    @property
    def unify(self) -> bool:
        return bool(self._unify_var.get())

    @property
    def normalize(self) -> bool:
        return bool(self._normalize_var.get())

    @property
    def mode_3d(self) -> bool:
        return bool(self._mode_3d.get())

    @property
    def show_surface(self) -> bool:
        return bool(self._show_surface.get())

    @property
    def show_wireframe(self) -> bool:
        return bool(self._show_wireframe_var.get())

    @property
    def show_projections(self) -> bool:
        return bool(self._show_projections_var.get())

    @property
    def show_errors_3d(self) -> bool:
        return bool(self._show_errors_3d.get())

    @property
    def z_label_mode(self) -> str:
        return self._z_label_mode.get()

    @property
    def show_level(self) -> bool:
        return bool(self._show_level_var.get())

    @property
    def level_val(self) -> int:
        return self._level_val_var.get()

    @property
    def subdiv_level(self) -> int:
        return int(self._subdiv_var.get())

    @property
    def interp_method(self) -> str:
        return self._interp_var.get()

    @property
    def mask_gaps(self) -> bool:
        return bool(self._mask_var.get())

    @property
    def surface_style(self) -> str:
        return self._surface_style_var.get()

    @property
    def axis_x(self) -> str:
        return self._axis_x_var.get()

    @property
    def axis_y(self) -> str:
        return self._axis_y_var.get()

    @property
    def root(self) -> tk.Tk:
        return self._root
