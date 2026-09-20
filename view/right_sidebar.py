"""
view/right_sidebar.py

Right sidebar View component for llamagraph.

Displays filter controls for all varying dimensions, grouped with the
currently plotted axes first ("Plot Axes") followed by a visible
separator and the remaining dimensions ("Other Dimensions").  For each
dimension the sidebar shows:
  - A labelled section header with the dimension name (axis sections
    carry an "X:" / "Y:" badge showing which plot axis they drive)
  - A Listbox (extended selection: click selects a single value,
    Ctrl+Click toggles individual values, Shift+Click selects a range)
    showing all known values for that dimension
  - Buttons: Select All / Clear for that section

When the user changes a selection the Presenter's filter-change callback
is called with the full current filter state.

This class is purely UI — it fires callbacks and never calls the Model.
"""

from __future__ import annotations

from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk

from utils.colors import COLORS


class RightSidebar(tk.Frame):
    """
    Right sidebar with per-dimension value filter controls.

    Public methods called by the Presenter:
      update_filter_sections(dim_values, active_axes, current_filters)

    Callbacks set by the Presenter:
      set_filter_change_callback(cb)   – cb(filter_dict: dict[str, set])
    """

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, bg=COLORS['bg'], **kwargs)
        self._filter_change_cb: Optional[Callable[[dict[str, set]], None]] = None

        # Internal: {dim_name: {'listbox': Listbox, 'values': list}}
        self._sections: dict[str, dict] = {}

        self._build_static_ui()

    # ── Static UI skeleton ────────────────────────────────────────────────────

    def _build_static_ui(self) -> None:
        tk.Label(
            self, text="🔍 Dimension Filters",
            bg=COLORS['bg'], fg=COLORS['fg'],
            font=('Segoe UI', 10, 'bold'),
        ).pack(pady=(10, 5))

        # Scrollable container for the filter sections
        outer = tk.Frame(self, bg=COLORS['bg'])
        outer.pack(fill=tk.BOTH, expand=True, padx=4)

        canvas = tk.Canvas(outer, bg=COLORS['bg'], highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._scroll_canvas = canvas
        self._inner_frame = tk.Frame(canvas, bg=COLORS['bg'])
        self._canvas_window = canvas.create_window(
            (0, 0), window=self._inner_frame, anchor='nw'
        )

        self._inner_frame.bind('<Configure>', self._on_inner_configure)
        canvas.bind('<Configure>', self._on_canvas_resize)

        # Placeholder shown when no dims need filtering
        self._placeholder = tk.Label(
            self._inner_frame,
            text="No variable dimensions\nto filter.",
            bg=COLORS['bg'], fg='#666666',
            font=('Segoe UI', 9),
            justify='center',
        )
        self._placeholder.pack(expand=True, pady=20)

    def _on_inner_configure(self, _event=None) -> None:
        self._scroll_canvas.configure(
            scrollregion=self._scroll_canvas.bbox('all')
        )

    def _on_canvas_resize(self, event) -> None:
        self._scroll_canvas.itemconfig(
            self._canvas_window, width=event.width
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update_filter_sections(
        self,
        dim_values: dict[str, list],
        active_axes: set[str] | list[str] | tuple[str, ...],
        current_filters: dict[str, set],
    ) -> None:
        """
        Rebuild filter sections for all varying dimensions.

        Dimensions currently used as plot axes are shown first under a
        "Plot Axes" group header (in X, Y order), followed by the
        remaining dimensions under an "Other Dimensions" group header
        that acts as a visible separator.  Axis sections stay fully
        interactive filters: restricting their values narrows the
        plotted range.  Dimensions with a single value carry no
        filtering information and are hidden in both groups.

        Parameters
        ----------
        dim_values:
            Full {dim_name: sorted_values} from the Model.
        active_axes:
            Dimension names currently used as plot axes, in X, Y order
            (a set is also accepted and then ordered by *dim_values*).
        current_filters:
            The current filter state from the Model.
        """
        # Remove all old widgets
        for widget in self._inner_frame.winfo_children():
            widget.destroy()
        self._sections.clear()

        if isinstance(active_axes, (list, tuple)):
            active_order = [d for d in active_axes if d]
        else:
            active_order = [d for d in dim_values if d in active_axes]
        active_set = set(active_order)

        # Axis dims first (X, Y order), then all other varying dims.
        axis_dims = [
            d for d in active_order
            if d in dim_values and len(dim_values[d]) > 1
        ]
        other_dims = [
            d for d, vals in dim_values.items()
            if d not in active_set and len(vals) > 1
        ]

        if not axis_dims and not other_dims:
            tk.Label(
                self._inner_frame,
                text="No variable dimensions\nto filter.",
                bg=COLORS['bg'], fg='#666666',
                font=('Segoe UI', 9), justify='center',
            ).pack(expand=True, pady=20)
            return

        if axis_dims:
            self._build_group_header("Plot Axes")
            for dim in axis_dims:
                values = dim_values[dim]
                self._build_section(
                    dim, values, current_filters.get(dim, set(values)),
                    axis_label=self._axis_badge(dim, active_order),
                    # The last axis section drops its own trailing line
                    # when the group separator below takes over.
                    show_separator=not (
                        other_dims and dim == axis_dims[-1]),
                )

        if other_dims:
            if axis_dims:
                self._build_group_separator("Other Dimensions")
            else:
                self._build_group_header("Dimensions")
            for dim in other_dims:
                values = dim_values[dim]
                self._build_section(
                    dim, values, current_filters.get(dim, set(values))
                )

    def get_current_filters(self) -> dict[str, set]:
        """Return the current filter state from all section listboxes."""
        result: dict[str, set] = {}
        for dim, sec in self._sections.items():
            lb = sec['listbox']
            all_vals = sec['values']
            selected_indices = lb.curselection()
            result[dim] = {all_vals[i] for i in selected_indices}
        return result

    # ── Callback registration ─────────────────────────────────────────────────

    def set_filter_change_callback(
        self, cb: Callable[[dict[str, set]], None]
    ) -> None:
        self._filter_change_cb = cb

    # ── Section builder ───────────────────────────────────────────────────────

    @staticmethod
    def _axis_badge(dim: str, active_order: list[str]) -> Optional[str]:
        """Return "X" / "Y" for the plot axis *dim* drives, else None."""
        if not active_order:
            return None
        if dim == active_order[0]:
            return "X"
        if len(active_order) > 1 and dim == active_order[1]:
            return "Y"
        return None

    def _build_group_header(self, title: str) -> None:
        """Build a group heading (e.g. "Plot Axes")."""
        tk.Label(
            self._inner_frame, text=title,
            bg=COLORS['bg'], fg='#888888',
            font=('Segoe UI', 9, 'bold'), anchor='w',
        ).pack(fill=tk.X, padx=8, pady=(8, 0))

    def _build_group_separator(self, title: str) -> None:
        """Build a visible separator with a group heading."""
        tk.Frame(self._inner_frame, bg=COLORS['separator'], height=1).pack(
            fill='x', padx=4, pady=(10, 2)
        )
        self._build_group_header(title)

    def _build_section(
        self, dim: str, values: list, selected_values: set,
        axis_label: Optional[str] = None,
        show_separator: bool = True,
    ) -> None:
        """
        Build one filter section for *dim*.

        *axis_label* is "X" / "Y" when the dimension drives a plot
        axis and None otherwise; it is shown as a badge prefix in the
        section header (e.g. "X: N Batch").  *show_separator* controls
        the trailing separator line (suppressed where a group
        separator follows instead, so lines never double up).
        """
        section = tk.Frame(self._inner_frame, bg=COLORS['bg'])
        section.pack(fill=tk.X, padx=4, pady=(6, 2))

        # Header row
        hdr = tk.Frame(section, bg=COLORS['highlight'])
        hdr.pack(fill=tk.X)
        title = dim.replace('_', ' ').title()
        header_text = f"  {axis_label}: {title}" if axis_label else f"  {title}"
        tk.Label(
            hdr, text=header_text,
            bg=COLORS['highlight'], fg=COLORS['fg'],
            font=('Segoe UI', 9, 'bold'), anchor='w',
        ).pack(side=tk.LEFT, padx=4, pady=3)

        # All / Clear quick buttons
        btn_frame = tk.Frame(hdr, bg=COLORS['highlight'])
        btn_frame.pack(side=tk.RIGHT, padx=4)

        tk.Button(
            btn_frame, text="All",
            width=3, pady=0,
            bg='#444', fg=COLORS['fg'],
            relief=tk.FLAT, cursor='hand2',
            font=('Segoe UI', 7),
            command=lambda d=dim: self._select_all_in(d),
        ).pack(side=tk.LEFT, padx=1)

        tk.Button(
            btn_frame, text="✕",
            width=2, pady=0,
            bg='#444', fg=COLORS['fg'],
            relief=tk.FLAT, cursor='hand2',
            font=('Segoe UI', 7),
            command=lambda d=dim: self._clear_in(d),
        ).pack(side=tk.LEFT, padx=1)

        # Listbox — height capped at 8 entries
        lb_frame = tk.Frame(section, bg=COLORS['bg'])
        lb_frame.pack(fill=tk.X)

        height = min(len(values), 8)
        lb = tk.Listbox(
            lb_frame,
            bg='#2d2d2d', fg=COLORS['fg'],
            selectbackground=COLORS['accent'], selectforeground='white',
            activestyle='none', font=('Consolas', 9),
            height=height,
            selectmode=tk.EXTENDED, exportselection=0,
            relief=tk.FLAT,
        )
        lb.pack(fill=tk.X, padx=2, pady=2)

        for val in values:
            lb.insert(tk.END, self._fmt_value(val))

        # Apply pre-selected items
        for i, val in enumerate(values):
            if val in selected_values:
                lb.selection_set(i)

        lb.bind('<<ListboxSelect>>', self._on_any_filter_change)

        self._sections[dim] = {'listbox': lb, 'values': values}

        # Separator (skipped where a group separator follows instead)
        if show_separator:
            tk.Frame(self._inner_frame, bg=COLORS['separator'], height=1).pack(
                fill='x', padx=4, pady=2
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_value(val) -> str:
        """Format a dimension value for display in the listbox."""
        if isinstance(val, float) and val == int(val):
            return str(int(val))
        return str(val)

    def _select_all_in(self, dim: str) -> None:
        sec = self._sections.get(dim)
        if sec:
            sec['listbox'].selection_set(0, tk.END)
            self._on_any_filter_change(None)

    def _clear_in(self, dim: str) -> None:
        sec = self._sections.get(dim)
        if sec:
            sec['listbox'].selection_clear(0, tk.END)
            self._on_any_filter_change(None)

    def _on_any_filter_change(self, _event) -> None:
        if self._filter_change_cb:
            self._filter_change_cb(self.get_current_filters())