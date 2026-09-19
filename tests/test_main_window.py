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


# ── Callback stubs ──────────────────────────────────────────────────────


class TestCallbackStubs:
    """Tests for MainWindow callback stubs (_on_render, _on_toggle_3d)."""

    def test_on_render_calls_callback(self, tk_root):
        """_on_render invokes _render_cb when set."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        cb = MagicMock()
        mw._render_cb = cb
        mw._on_render()
        cb.assert_called_once()

    def test_on_render_no_callback(self, tk_root):
        """_on_render does nothing when _render_cb is None."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        mw._render_cb = None
        mw._on_render()  # should not raise

    def test_on_toggle_3d_calls_callback(self, tk_root):
        """_on_toggle_3d invokes _toggle_3d_cb when set."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        cb = MagicMock()
        mw._toggle_3d_cb = cb
        mw._on_toggle_3d()
        cb.assert_called_once()

    def test_on_toggle_3d_no_callback(self, tk_root):
        """_on_toggle_3d does nothing when _toggle_3d_cb is None."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        mw._toggle_3d_cb = None
        mw._on_toggle_3d()  # should not raise


# ── Callback setters ────────────────────────────────────────────────────


class TestCallbackSetters:
    """Tests for MainWindow callback setter methods."""

    def test_set_render_callback(self, tk_root):
        """set_render_callback stores the callback."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        cb = MagicMock()
        mw.set_render_callback(cb)
        assert mw._render_cb == cb

    def test_set_toggle_3d_callback(self, tk_root):
        """set_toggle_3d_callback stores the callback."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        cb = MagicMock()
        mw.set_toggle_3d_callback(cb)
        assert mw._toggle_3d_cb == cb

    def test_set_toggle_metric_callback(self, tk_root):
        """set_toggle_metric_callback configures the toggle button."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        cb = MagicMock()
        mw.set_toggle_metric_callback(cb)
        mw._toggle_btn.invoke()
        cb.assert_called_once()


# ── UI state methods ────────────────────────────────────────────────────


class TestUISetters:
    """Tests for set_unify_state and set_metric_button_text."""

    def test_set_unify_state_enabled(self, tk_root):
        """set_unify_state(True) → unify checkbox state=NORMAL."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        mw.set_unify_state(True)
        assert str(mw._unify_chk.cget("state")) == "normal"

    def test_set_unify_state_disabled(self, tk_root):
        """set_unify_state(False) → unify checkbox state=DISABLED."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        mw.set_unify_state(False)
        assert str(mw._unify_chk.cget("state")) == "disabled"

    def test_set_metric_button_text(self, tk_root):
        """set_metric_button_text updates toggle button text."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        mw.set_metric_button_text("t/s → time")
        assert mw._toggle_btn.cget("text") == "t/s → time"


# ── Key bindings ────────────────────────────────────────────────────────


class TestKeyBindings:
    """Tests for MainWindow.set_key_bindings()."""

    def test_registers_three_bindings(self, tk_root):
        """set_key_bindings registers exactly 3 key bindings on root."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        toggle = MagicMock()
        refresh = MagicMock()
        quit_app = MagicMock()
        mw.set_key_bindings(toggle, refresh, quit_app)
        bindings = mw._root.bind()
        assert '<Control-Key-t>' in bindings
        assert '<Control-Key-r>' in bindings
        assert '<Key-Escape>' in bindings

    def test_control_t_triggers_toggle(self, tk_root):
        """Ctrl+T triggers toggle_metric via callback."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        toggle = MagicMock()
        refresh = MagicMock()
        quit_app = MagicMock()
        mw.set_key_bindings(toggle, refresh, quit_app)
        mw._root.update()
        mw._root.event_generate('<Control-t>', when='tail')
        mw._root.update()
        toggle.assert_called_once()

    def test_control_r_triggers_refresh(self, tk_root):
        """Ctrl+R triggers refresh via callback."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        toggle = MagicMock()
        refresh = MagicMock()
        quit_app = MagicMock()
        mw.set_key_bindings(toggle, refresh, quit_app)
        mw._root.update()
        mw._root.event_generate('<Control-r>', when='tail')
        mw._root.update()
        refresh.assert_called_once()

    def test_escape_triggers_quit(self, tk_root):
        """Escape triggers quit_app via callback."""
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        toggle = MagicMock()
        refresh = MagicMock()
        quit_app = MagicMock()
        mw.set_key_bindings(toggle, refresh, quit_app)
        mw._root.update()
        mw._root.event_generate('<Escape>', when='tail')
        mw._root.update()
        quit_app.assert_called_once()


# ── Level controls ────────────────────────────────────────────────────


class TestLevelControls:
    """Tests for the Lev stepper buttons and the editable value entry."""

    def _make_window(self, tk_root):
        from view.main_window import MainWindow
        mw = MainWindow(tk_root)
        render_cb = MagicMock()
        mw.set_render_callback(render_cb)
        return mw, render_cb

    def test_initial_value_shown(self, tk_root):
        """Entry starts with the default level value (50)."""
        mw, _ = self._make_window(tk_root)
        assert mw._ent_level.get() == "50"
        assert mw.level_val == 50

    def test_step_buttons_nudge_and_render(self, tk_root):
        """Arrow buttons change the value by ±1 and re-render."""
        mw, render_cb = self._make_window(tk_root)
        mw._btn_level_up.invoke()
        assert mw.level_val == 51
        assert mw._ent_level.get() == "51"
        mw._btn_level_down.invoke()
        mw._btn_level_down.invoke()
        assert mw.level_val == 49
        assert render_cb.call_count == 3

    def test_step_clamps_at_bounds(self, tk_root):
        """Stepping past 0/100 clamps instead of wrapping."""
        mw, _ = self._make_window(tk_root)
        mw._set_level_value(0)
        mw._btn_level_down.invoke()
        assert mw.level_val == 0
        mw._set_level_value(100)
        mw._btn_level_up.invoke()
        assert mw.level_val == 100

    def test_entry_commit_takes_valid_value(self, tk_root):
        """Typing a number and committing applies it (clamped)."""
        mw, render_cb = self._make_window(tk_root)
        mw._ent_level.delete(0, "end")
        mw._ent_level.insert(0, "200")
        mw._commit_level_entry()
        assert mw.level_val == 100
        assert mw._ent_level.get() == "100"
        render_cb.assert_called()

    def test_entry_commit_reverts_garbage(self, tk_root):
        """Non-numeric entry restores the current value without rendering."""
        mw, render_cb = self._make_window(tk_root)
        render_cb.reset_mock()
        mw._ent_level.delete(0, "end")
        mw._ent_level.insert(0, "abc")
        mw._commit_level_entry()
        assert mw.level_val == 50
        assert mw._ent_level.get() == "50"
        render_cb.assert_not_called()

    def test_noop_commit_skips_render(self, tk_root):
        """Unchanged value (e.g. focus passthrough) triggers no render."""
        mw, render_cb = self._make_window(tk_root)
        render_cb.reset_mock()
        mw._commit_level_entry()  # entry already shows the current value
        render_cb.assert_not_called()
