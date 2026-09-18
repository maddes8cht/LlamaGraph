"""
view/left_sidebar.py

Left sidebar View component for llamagraph.

Displays:
  - Current working directory label
  - File list (Listbox with multi-select)
  - Sort / Refresh buttons
  - Per-series PP/TG checkboxes (rebuilt whenever selection changes)

This class is purely UI — it only fires callbacks to the Presenter.
It never calls the Model directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk, filedialog

from utils.colors import COLORS, get_variant_color, DEFAULT_PP_COLOR, DEFAULT_TG_COLOR


class LeftSidebar(tk.Frame):
    """
    Left sidebar containing the CSV file list and per-series toggles.

    Public methods called by the Presenter:
      populate_file_list(names, sort_label)
      update_series_toggles(dataset_paths)
      set_directory_label(path)
      get_pp_flag(i) / get_tg_flag(i)

    Callbacks set by the Presenter:
      set_file_select_callback(cb)        – cb(selected_indices: list[int])
      set_sort_callback(cb)               – cb()
      set_refresh_callback(cb)            – cb()
      set_series_toggle_callback(cb)      – cb()  (any toggle changed)
      set_select_all_callback(cb)         – cb()
      set_deselect_all_callback(cb)       – cb()
      set_md_toggle_callback(cb)          – cb()  (.md visibility toggled)
      set_choose_directory_callback(cb)   – cb(chosen_path: Path)
    """

    def __init__(self, parent: tk.Widget, pp_color: str = DEFAULT_PP_COLOR,
                 tg_color: str = DEFAULT_TG_COLOR, **kwargs) -> None:
        super().__init__(parent, bg=COLORS['bg'], **kwargs)
        self._pp_color = pp_color
        self._tg_color = tg_color

        # Per-series toggle variables {idx: tk.IntVar}
        self._pp_vars: dict[int, tk.IntVar] = {}
        self._tg_vars: dict[int, tk.IntVar] = {}

        # Callbacks (injected by Presenter)
        self._file_select_cb: Optional[Callable] = None
        self._sort_cb: Optional[Callable] = None
        self._refresh_cb: Optional[Callable] = None
        self._series_toggle_cb: Optional[Callable] = None
        self._select_all_cb: Optional[Callable] = None
        self._deselect_all_cb: Optional[Callable] = None
        self._md_toggle_cb: Optional[Callable] = None
        self._choose_directory_cb: Optional[Callable[[Path], None]] = None

        # Whether the .md button is currently packed (packed in _build_ui)
        self._md_visible: bool = True

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Header row: title + Browse button on the right
        header_row = tk.Frame(self, bg=COLORS['bg'])
        header_row.pack(fill=tk.X, padx=5, pady=(10, 2))

        tk.Label(header_row, text="📁 CSV Files", bg=COLORS['bg'], fg=COLORS['fg'],
                 font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)

        tk.Button(
            header_row, text="📂 Browse…",
            command=self._on_choose_directory,
            bg='#3a3a3a', fg=COLORS['fg'],
            relief=tk.FLAT, cursor='hand2',
            font=('Segoe UI', 8),
            activebackground=COLORS['accent'], activeforeground='white',
        ).pack(side=tk.RIGHT)

        # Current directory label (truncated, right-to-left ellipsis)
        self._dir_label = tk.Label(
            self, text="", bg=COLORS['bg'], fg='#888888',
            font=('Consolas', 7), anchor='w', justify='left',
        )
        self._dir_label.pack(fill=tk.X, padx=6, pady=(0, 4))

        # Select All / Deselect All / .md visibility toggle
        btn_row = tk.Frame(self, bg=COLORS['bg'])
        btn_row.pack(fill=tk.X, padx=5)
        tk.Button(btn_row, text="Select All",
                  command=self._on_select_all,
                  bg=COLORS['accent'], fg='white', relief=tk.FLAT,
                  cursor='hand2', font=('Segoe UI', 8)
                  ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(btn_row, text="Deselect All",
                  command=self._on_deselect_all,
                  bg='#444', fg='white', relief=tk.FLAT,
                  cursor='hand2', font=('Segoe UI', 8)
                  ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self._md_btn = tk.Button(btn_row, text=".md",
                  command=self._on_md_toggle,
                  bg=COLORS['accent'], fg='white', relief=tk.FLAT,
                  cursor='hand2', font=('Segoe UI', 8)
                  )
        self._md_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # File Listbox with scrollbar
        list_frame = tk.Frame(self, bg=COLORS['bg'])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._file_list = tk.Listbox(
            list_frame, bg='#2d2d2d', fg=COLORS['fg'],
            selectbackground=COLORS['accent'], selectforeground='white',
            activestyle='none', font=('Consolas', 9),
            yscrollcommand=scrollbar.set,
            selectmode=tk.MULTIPLE, exportselection=0,
        )
        self._file_list.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._file_list.yview)
        self._file_list.bind('<<ListboxSelect>>', self._on_listbox_select)

        # Series-toggle area (populated dynamically)
        self._series_header = tk.Label(
            self, text="🎛 Series Toggles",
            bg=COLORS['bg'], fg=COLORS['fg'],
            font=('Segoe UI', 9, 'bold'),
        )
        self._series_header.pack(anchor='w', padx=7, pady=(4, 0))

        self._series_frame = tk.Frame(self, bg=COLORS['bg'])
        self._series_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Sort / Refresh buttons
        btn_bottom = tk.Frame(self, bg=COLORS['bg'])
        btn_bottom.pack(fill=tk.X, padx=5, pady=(0, 8))

        self._sort_btn = tk.Button(
            btn_bottom, text="Sort: Time ↓",
            command=self._on_sort,
            bg=COLORS['accent'], fg='white', relief=tk.FLAT,
            cursor='hand2', font=('Segoe UI', 9),
        )
        self._sort_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        tk.Button(
            btn_bottom, text="Refresh",
            command=self._on_refresh,
            bg=COLORS['accent'], fg='white', relief=tk.FLAT,
            cursor='hand2', font=('Segoe UI', 9),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

    # ── Public API (called by Presenter) ─────────────────────────────────────

    def populate_file_list(self, names: list[str], sort_label: str) -> None:
        """Rebuild the file listbox with *names* and update the sort button."""
        self._file_list.delete(0, tk.END)
        for name in names:
            self._file_list.insert(tk.END, name)
        self._sort_btn.config(text=sort_label)

    def set_directory_label(self, path) -> None:
        """
        Update the directory path label below the header.
        Truncates long paths from the left so the folder name tail is visible.
        """
        text = str(path)
        if len(text) > 42:
            text = "\u2026" + text[-41:]
        self._dir_label.config(text=text)

    def get_selected_indices(self) -> list[int]:
        return list(self._file_list.curselection())

    def select_index(self, idx: int) -> None:
        """Programmatically select a specific index in the listbox."""
        self._file_list.selection_set(idx)
        self._file_list.see(idx)

    def select_all(self) -> None:
        self._file_list.selection_set(0, tk.END)
        self._on_listbox_select(None)

    def deselect_all(self) -> None:
        self._file_list.selection_clear(0, tk.END)
        self._on_listbox_select(None)

    def set_md_toggle_visible(self, visible: bool) -> None:
        """Show or hide the `.md` toggle button (hidden with `--no-md`)."""
        if visible and not self._md_visible:
            self._md_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
            self._md_visible = True
        elif not visible and self._md_visible:
            self._md_btn.pack_forget()
            self._md_visible = False

    def set_md_toggle_state(self, active: bool) -> None:
        """Reflect whether `.md` files are currently listed (accent = on)."""
        self._md_btn.config(bg=COLORS['accent'] if active else '#444')

    def update_series_toggles(self, dataset_paths: list[Path]) -> None:
        """
        Rebuild the per-series PP/TG checkboxes.
        Called by the Presenter after the dataset selection changes.
        """
        # Tear down old widgets
        for widget in self._series_frame.winfo_children():
            widget.destroy()
        self._pp_vars.clear()
        self._tg_vars.clear()

        for i, path in enumerate(dataset_paths):
            stem = Path(path).stem[:16]
            pp_var = tk.IntVar(value=1)
            tg_var = tk.IntVar(value=1)
            self._pp_vars[i] = pp_var
            self._tg_vars[i] = tg_var

            pp_color = get_variant_color(self._pp_color, i)
            tg_color = get_variant_color(self._tg_color, i)

            tk.Checkbutton(
                self._series_frame, text=f"PP  {stem}",
                variable=pp_var, command=self._on_series_toggle,
                bg=COLORS['bg'], fg=pp_color,
                selectcolor=COLORS['checkbox_active'], font=('Segoe UI', 8),
            ).pack(anchor='w', padx=5)

            tk.Checkbutton(
                self._series_frame, text=f"TG  {stem}",
                variable=tg_var, command=self._on_series_toggle,
                bg=COLORS['bg'], fg=tg_color,
                selectcolor=COLORS['checkbox_active'], font=('Segoe UI', 8),
            ).pack(anchor='w', padx=5)

            if i < len(dataset_paths) - 1:
                tk.Frame(self._series_frame, bg='#444', height=1).pack(
                    fill='x', pady=3
                )

    def get_pp_flag(self, i: int) -> bool:
        var = self._pp_vars.get(i)
        return bool(var.get()) if var else True

    def get_tg_flag(self, i: int) -> bool:
        var = self._tg_vars.get(i)
        return bool(var.get()) if var else True
    
    # ── Callback registration (called by Presenter) ───────────────────────────

    def set_file_select_callback(self, cb: Callable[[list[int]], None]) -> None:
        self._file_select_cb = cb

    def set_sort_callback(self, cb: Callable[[], None]) -> None:
        self._sort_cb = cb

    def set_refresh_callback(self, cb: Callable[[], None]) -> None:
        self._refresh_cb = cb

    def set_series_toggle_callback(self, cb: Callable[[], None]) -> None:
        self._series_toggle_cb = cb

    def set_select_all_callback(self, cb: Callable[[], None]) -> None:
        self._select_all_cb = cb

    def set_deselect_all_callback(self, cb: Callable[[], None]) -> None:
        self._deselect_all_cb = cb

    def set_md_toggle_callback(self, cb: Callable[[], None]) -> None:
        self._md_toggle_cb = cb

    def set_choose_directory_callback(self, cb) -> None:
        """Register callback called with the chosen Path when user picks a dir."""
        self._choose_directory_cb = cb

    # ── Internal event handlers ───────────────────────────────────────────────

    def _on_listbox_select(self, _event) -> None:
        if self._file_select_cb:
            self._file_select_cb(list(self._file_list.curselection()))

    def _on_sort(self) -> None:
        if self._sort_cb:
            self._sort_cb()

    def _on_refresh(self) -> None:
        if self._refresh_cb:
            self._refresh_cb()

    def _on_series_toggle(self) -> None:
        if self._series_toggle_cb:
            self._series_toggle_cb()

    def _on_select_all(self) -> None:
        if self._select_all_cb:
            self._select_all_cb()
        else:
            self.select_all()

    def _on_deselect_all(self) -> None:
        if self._deselect_all_cb:
            self._deselect_all_cb()
        else:
            self.deselect_all()

    def _on_md_toggle(self) -> None:
        if self._md_toggle_cb:
            self._md_toggle_cb()

    def _on_choose_directory(self) -> None:
        """Open a native directory chooser; fire callback with the chosen Path."""
        chosen = filedialog.askdirectory(title="Select benchmark directory")
        if chosen and self._choose_directory_cb:
            self._choose_directory_cb(Path(chosen))