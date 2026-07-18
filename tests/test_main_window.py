"""Tests for view/main_window.py - Tkinter window and toolbar logic."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def tk_root():
    """Create a single hidden tk root for all MainWindow tests."""
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


class TestUpdateAxisChoices:
    """Tests for MainWindow.update_axis_choices() - axis combobox logic."""

    def test_populates_comboboxes(self, tk_root):
        """Normal params list → comboboxes populated and state=readonly."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        mw.update_axis_choices(["a", "b", "c"])
        assert mw._cb_x.cget("values") == ("a", "b", "c")
        assert mw._cb_y.cget("values") == ("a", "b", "c")
        assert str(mw._cb_x.cget("state")) == "readonly"
        assert str(mw._cb_y.cget("state")) == "readonly"

    def test_empty_params_disables(self, tk_root):
        """Empty params → combobox state=disabled and selections cleared."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        mw.update_axis_choices([])
        assert str(mw._cb_x.cget("state")) == "disabled"
        assert str(mw._cb_y.cget("state")) == "disabled"
        assert mw.axis_x == ""
        assert mw.axis_y == ""

    def test_auto_selects_first(self, tk_root):
        """No prior selection → auto-selects first param for X, second for Y."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        mw.update_axis_choices(["alpha", "beta", "gamma"])
        assert mw.axis_x == "alpha"
        assert mw.axis_y == "beta"

    def test_preserves_valid_x_selection(self, tk_root):
        """Current X still in params → kept."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        mw.update_axis_choices(["a", "b", "c"])
        mw._axis_x_var.set("b")
        mw.update_axis_choices(["a", "b", "c", "d"])
        assert mw.axis_x == "b"

    def test_resets_invalid_x_selection(self, tk_root):
        """Current X no longer in params → reset to first."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        mw.update_axis_choices(["a", "b", "c"])
        mw._axis_x_var.set("c")
        mw.update_axis_choices(["a", "b"])
        assert mw.axis_x == "a"

    def test_preserves_valid_y_selection(self, tk_root):
        """Current Y still in params and != X → kept."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        mw.update_axis_choices(["a", "b", "c"])
        mw._axis_x_var.set("a")
        mw._axis_y_var.set("c")
        mw.update_axis_choices(["a", "c", "d"])
        assert mw.axis_y == "c"

    def test_avoids_duplicate_xy(self, tk_root):
        """If X and Y would be the same, Y is moved to the next available."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        mw.update_axis_choices(["a", "b", "c"])
        mw._axis_x_var.set("b")
        mw._axis_y_var.set("b")
        mw.update_axis_choices(["a", "b", "c"])
        assert mw.axis_x == "b"
        assert mw.axis_y != "b"
        assert mw.axis_y in ("a", "c")

    def test_single_param(self, tk_root):
        """Only one param → X gets it, Y gets empty string."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        mw.update_axis_choices(["only"])
        assert mw.axis_x == "only"
        assert mw.axis_y == ""
