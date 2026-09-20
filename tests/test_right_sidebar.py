"""Tests for view/right_sidebar.py - Dimension filter sidebar."""
from __future__ import annotations

import tkinter as tk
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from view.right_sidebar import RightSidebar


@contextmanager
def _patched_rs():
    """RightSidebar with all tk widget constructors mocked out."""
    with (
        patch('view.right_sidebar.tk.Frame'),
        patch('view.right_sidebar.tk.Label'),
        patch('view.right_sidebar.tk.Listbox') as _lb,
        patch('view.right_sidebar.tk.Canvas'),
        patch('view.right_sidebar.ttk.Scrollbar'),
        patch('view.right_sidebar.tk.Button'),
    ):
        rs = RightSidebar(MagicMock())
        rs._inner_frame = MagicMock()
        rs._placeholder = MagicMock()
        yield rs, _lb


# ── _fmt_value (static method) ─────────────────────────────────────────


class TestFmtValue:
    """Tests for RightSidebar._fmt_value() - value formatting."""

    def test_float_int_value(self):
        """Float with no fractional part → formatted as int string."""
        assert RightSidebar._fmt_value(42.0) == "42"
        assert RightSidebar._fmt_value(0.0) == "0"
        assert RightSidebar._fmt_value(-5.0) == "-5"

    def test_float_with_fraction(self):
        """Float with fractional part → default str()."""
        assert RightSidebar._fmt_value(3.14) == "3.14"
        assert RightSidebar._fmt_value(0.5) == "0.5"

    def test_string_value(self):
        """String value → returned as-is."""
        assert RightSidebar._fmt_value("hello") == "hello"
        assert RightSidebar._fmt_value("70B") == "70B"
        assert RightSidebar._fmt_value("") == ""

    def test_int_value(self):
        """int value (not float) → default str()."""
        assert RightSidebar._fmt_value(42) == "42"
        assert RightSidebar._fmt_value(0) == "0"

    def test_none_value(self):
        """None value → 'None' (Python default str)."""
        assert RightSidebar._fmt_value(None) == "None"


# ── get_current_filters ────────────────────────────────────────────────


class TestGetCurrentFilters:
    """Tests for RightSidebar.get_current_filters() - filter state readback."""

    def test_empty_sections(self):
        """No sections → empty dict."""
        with _patched_rs() as (rs, _):
            rs._sections = {}
            assert rs.get_current_filters() == {}

    def test_single_dimension(self):
        """One dimension with selections."""
        with _patched_rs() as (rs, _):
            lb = MagicMock()
            lb.curselection.return_value = (0, 2)
            rs._sections = {
                'gpu_name': {'listbox': lb, 'values': ['GPU_A', 'GPU_B', 'GPU_C']},
            }
            result = rs.get_current_filters()
            assert result == {'gpu_name': {'GPU_A', 'GPU_C'}}

    def test_multiple_dimensions(self):
        """Multiple dimensions each return their selected values."""
        with _patched_rs() as (rs, _):
            lb1 = MagicMock()
            lb1.curselection.return_value = (1,)
            lb2 = MagicMock()
            lb2.curselection.return_value = (0, 1, 2)
            rs._sections = {
                'dim1': {'listbox': lb1, 'values': ['a', 'b', 'c']},
                'dim2': {'listbox': lb2, 'values': [10, 20, 30]},
            }
            result = rs.get_current_filters()
            assert result == {'dim1': {'b'}, 'dim2': {10, 20, 30}}

    def test_no_selection(self):
        """No selected indices → empty set for that dim."""
        with _patched_rs() as (rs, _):
            lb = MagicMock()
            lb.curselection.return_value = ()
            rs._sections = {
                'params': {'listbox': lb, 'values': ['70B', '7B', '13B']},
            }
            result = rs.get_current_filters()
            assert result == {'params': set()}


# ── update_filter_sections ─────────────────────────────────────────────


class TestUpdateFilterSections:
    """Tests for RightSidebar.update_filter_sections() - filter section rebuild."""

    def test_empty_dim_values(self):
        """Empty dim_values → sections cleared, no _build_section call."""
        with _patched_rs() as (rs, _):
            rs._inner_frame.winfo_children.return_value = []
            rs._sections = {'old': 'garbage'}

            with patch.object(rs, '_build_section') as mock_build:
                rs.update_filter_sections({}, set(), {})
                mock_build.assert_not_called()
                assert rs._sections == {}

    def test_all_dims_are_active_axes(self):
        """Active axes are shown first under the Plot Axes group."""
        with _patched_rs() as (rs, _):
            rs._inner_frame.winfo_children.return_value = []

            with patch.object(rs, '_build_section') as mock_build:
                rs.update_filter_sections(
                    {"dim1": ["a", "b"], "dim2": ["x", "y"]},
                    ["dim1", "dim2"],
                    {},
                )
                assert mock_build.call_count == 2
                called_dims = [call[0][0] for call in mock_build.call_args_list]
                assert called_dims == ["dim1", "dim2"]
                assert mock_build.call_args_list[0].kwargs.get("axis_label") == "X"
                assert mock_build.call_args_list[1].kwargs.get("axis_label") == "Y"

    def test_excludes_dims_with_single_value(self):
        """A dim with only 1 value is excluded (needs >1 for filtering)."""
        with _patched_rs() as (rs, _):
            rs._inner_frame.winfo_children.return_value = []

            with patch.object(rs, '_build_section') as mock_build:
                rs.update_filter_sections(
                    {"dim1": ["only"], "dim2": ["a", "b"]},
                    set(),
                    {},
                )
                mock_build.assert_called_once()
                args = mock_build.call_args[0]
                assert args[0] == "dim2"

    def test_builds_sections_for_inactive_dims(self):
        """Axis dims come first (X, Y order), then the other dimensions."""
        with _patched_rs() as (rs, _):
            rs._inner_frame.winfo_children.return_value = []

            with patch.object(rs, '_build_section') as mock_build:
                rs.update_filter_sections(
                    {"dim_a": [1, 2], "dim_b": [10, 20], "dim_c": [1, 2]},
                    ["dim_b"],
                    {},
                )
                assert mock_build.call_count == 3
                called_dims = [call[0][0] for call in mock_build.call_args_list]
                assert called_dims == ["dim_b", "dim_a", "dim_c"]
                assert mock_build.call_args_list[0].kwargs.get("axis_label") == "X"
                assert "axis_label" not in mock_build.call_args_list[1].kwargs
                assert "axis_label" not in mock_build.call_args_list[2].kwargs

    def test_axis_dim_with_single_value_hidden(self):
        """An axis dim with one value is hidden; others still show."""
        with _patched_rs() as (rs, _):
            rs._inner_frame.winfo_children.return_value = []

            with patch.object(rs, '_build_section') as mock_build:
                rs.update_filter_sections(
                    {"dim1": ["only"], "dim2": ["a", "b"]},
                    ["dim1"],
                    {},
                )
                mock_build.assert_called_once()
                args = mock_build.call_args[0]
                assert args[0] == "dim2"

    def test_group_headers_and_separator(self):
        """Plot Axes header first, then a separator before Other Dimensions."""
        with _patched_rs() as (rs, _):
            rs._inner_frame.winfo_children.return_value = []

            with (
                patch.object(rs, '_build_section'),
                patch.object(rs, '_build_group_header') as mock_header,
                patch.object(rs, '_build_group_separator') as mock_sep,
            ):
                rs.update_filter_sections(
                    {"dim_a": [1, 2], "dim_b": [10, 20]},
                    ["dim_b"],
                    {},
                )
                mock_header.assert_any_call("Plot Axes")
                mock_sep.assert_called_once_with("Other Dimensions")

    def test_no_second_group_without_other_dims(self):
        """Only axes present → no Other Dimensions separator."""
        with _patched_rs() as (rs, _):
            rs._inner_frame.winfo_children.return_value = []

            with (
                patch.object(rs, '_build_section'),
                patch.object(rs, '_build_group_header') as mock_header,
                patch.object(rs, '_build_group_separator') as mock_sep,
            ):
                rs.update_filter_sections(
                    {"dim1": ["a", "b"]},
                    ["dim1"],
                    {},
                )
                mock_header.assert_called_once_with("Plot Axes")
                mock_sep.assert_not_called()

    def test_neutral_header_without_axis_dims(self):
        """No axis dims → plain "Dimensions" header, no separator."""
        with _patched_rs() as (rs, _):
            rs._inner_frame.winfo_children.return_value = []

            with (
                patch.object(rs, '_build_section') as mock_build,
                patch.object(rs, '_build_group_header') as mock_header,
                patch.object(rs, '_build_group_separator') as mock_sep,
            ):
                rs.update_filter_sections(
                    {"dim1": ["a", "b"], "dim2": ["x", "y"]},
                    [],
                    {},
                )
                assert mock_build.call_count == 2
                mock_header.assert_called_once_with("Dimensions")
                mock_sep.assert_not_called()

    def test_no_double_separator_before_other_group(self):
        """Last axis section skips its line where the group separator follows."""
        with _patched_rs() as (rs, _):
            rs._inner_frame.winfo_children.return_value = []

            with patch.object(rs, '_build_section') as mock_build:
                rs.update_filter_sections(
                    {"dim_a": [1, 2], "dim_b": [10, 20]},
                    ["dim_a"],
                    {},
                )
                assert mock_build.call_count == 2
                axis_kwargs = mock_build.call_args_list[0].kwargs
                other_kwargs = mock_build.call_args_list[1].kwargs
                assert axis_kwargs.get("show_separator") is False
                assert other_kwargs.get("show_separator", True) is True

    def test_axis_separator_kept_without_other_group(self):
        """Only axes present → axis sections keep their own separator."""
        with _patched_rs() as (rs, _):
            rs._inner_frame.winfo_children.return_value = []

            with patch.object(rs, '_build_section') as mock_build:
                rs.update_filter_sections(
                    {"dim1": ["a", "b"], "dim2": ["x", "y"]},
                    ["dim1", "dim2"],
                    {},
                )
                for call in mock_build.call_args_list:
                    assert call.kwargs.get("show_separator", True) is True

    def test_current_filters_passed_to_build(self):
        """Current filter set is forwarded as the third argument."""
        with _patched_rs() as (rs, _):
            rs._inner_frame.winfo_children.return_value = []

            with patch.object(rs, '_build_section') as mock_build:
                rs.update_filter_sections(
                    {"dim1": ["a", "b", "c"]},
                    set(),
                    {"dim1": {"a"}},
                )
                mock_build.assert_called_once()
                args = mock_build.call_args[0]
                assert args[0] == "dim1"
                assert args[1] == ["a", "b", "c"]
                assert args[2] == {"a"}

    def test_falls_back_to_all_values_when_no_filter(self):
        """Dim without entry in current_filters → passes set(values) instead."""
        with _patched_rs() as (rs, _):
            rs._inner_frame.winfo_children.return_value = []

            with patch.object(rs, '_build_section') as mock_build:
                rs.update_filter_sections(
                    {"dim1": ["a", "b"]},
                    set(),
                    {},  # no filter for dim1
                )
                mock_build.assert_called_once()
                args = mock_build.call_args[0]
                assert args[2] == {"a", "b"}


# ── _build_section (direct, not mocked) ─────────────────────────────────


class TestBuildSection:
    """Tests for RightSidebar._build_section() - UI section construction."""

    def test_build_section_populates_sections_dict(self):
        """_build_section stores dim, listbox, and values in _sections."""
        with _patched_rs() as (rs, _):
            rs._sections = {}

            rs._build_section("gpu", ["A", "B", "C"], {"A", "C"})

            assert "gpu" in rs._sections
            sec = rs._sections["gpu"]
            assert sec["values"] == ["A", "B", "C"]
            assert sec["listbox"] is not None

    def test_build_section_selects_matching_values(self):
        """Pre-selected values cause selection_set calls on the listbox."""
        with _patched_rs() as (rs, _):
            rs._sections = {}

            rs._build_section("dim", ["a", "b", "c"], {"b"})

            lb = rs._sections["dim"]["listbox"]
            lb.selection_set.assert_any_call(1)

    def test_build_section_caps_height_at_8(self):
        """Listbox height is capped at 8 even with many values."""
        with _patched_rs() as (rs, lb_mock):
            rs._sections = {}

            many_vals = [str(i) for i in range(20)]
            rs._build_section("dim", many_vals, set())

            call_kwargs = lb_mock.call_args.kwargs
            assert call_kwargs["height"] == 8

    def test_build_section_binds_filter_change(self):
        """Listbox gets <<ListboxSelect>> bound."""
        with _patched_rs() as (rs, _):
            rs._sections = {}

            rs._build_section("dim", ["a", "b"], set())

            lb = rs._sections["dim"]["listbox"]
            lb.bind.assert_any_call('<<ListboxSelect>>',
                                    rs._on_any_filter_change)

    def test_build_section_with_no_selected_values(self):
        """No selected_values → no selection_set calls."""
        with _patched_rs() as (rs, _):
            rs._sections = {}

            rs._build_section("dim", ["a", "b", "c"], set())

            lb = rs._sections["dim"]["listbox"]
            lb.selection_set.assert_not_called()


# ── _select_all_in / _clear_in ──────────────────────────────────────────


class TestSelectClearIn:
    """Tests for _select_all_in and _clear_in helper methods."""

    def test_select_all_in_selects_and_fires_callback(self):
        """_select_all_in selects all items and dispatches filter change."""
        with _patched_rs() as (rs, _):
            lb = MagicMock()
            rs._sections = {"dim": {"listbox": lb, "values": ["a", "b"]}}
            rs._filter_change_cb = MagicMock()
            rs.get_current_filters = MagicMock(return_value={"dim": {"a", "b"}})

            rs._select_all_in("dim")

            lb.selection_set.assert_called_once_with(0, tk.END)
            rs._filter_change_cb.assert_called_once()
            assert rs._filter_change_cb.call_args[0][0] == {"dim": {"a", "b"}}

    def test_clear_in_clears_and_fires_callback(self):
        """_clear_in clears all items and dispatches filter change."""
        with _patched_rs() as (rs, _):
            lb = MagicMock()
            rs._sections = {"dim": {"listbox": lb, "values": ["a", "b"]}}
            rs._filter_change_cb = MagicMock()
            rs.get_current_filters = MagicMock(return_value={"dim": set()})

            rs._clear_in("dim")

            lb.selection_clear.assert_called_once_with(0, tk.END)
            rs._filter_change_cb.assert_called_once()
            assert rs._filter_change_cb.call_args[0][0] == {"dim": set()}

    def test_select_all_in_missing_dim_no_error(self):
        """_select_all_in with non-existent dim silently does nothing."""
        with _patched_rs() as (rs, _):
            rs._sections = {}
            rs._filter_change_cb = MagicMock()

            rs._select_all_in("nonexistent")  # should not raise
            rs._filter_change_cb.assert_not_called()

    def test_clear_in_missing_dim_no_error(self):
        """_clear_in with non-existent dim silently does nothing."""
        with _patched_rs() as (rs, _):
            rs._sections = {}
            rs._filter_change_cb = MagicMock()

            rs._clear_in("nonexistent")  # should not raise
            rs._filter_change_cb.assert_not_called()


# ── _on_any_filter_change ───────────────────────────────────────────────


class TestOnAnyFilterChange:
    """Tests for _on_any_filter_change callback dispatch."""

    def test_fires_callback(self):
        """_on_any_filter_change fires the registered callback."""
        with _patched_rs() as (rs, _):
            rs._filter_change_cb = MagicMock()
            rs.get_current_filters = MagicMock(return_value={"dim": {"a"}})

            rs._on_any_filter_change(None)

            rs._filter_change_cb.assert_called_once_with({"dim": {"a"}})

    def test_no_callback_no_error(self):
        """_on_any_filter_change does nothing when no callback set."""
        with _patched_rs() as (rs, _):
            rs._filter_change_cb = None

            rs._on_any_filter_change(None)  # should not raise


# ── _axis_badge / selection mode ──────────────────────────────────────


class TestAxisBadge:
    """Tests for RightSidebar._axis_badge() - X/Y badge mapping."""

    def test_first_axis_is_x(self):
        """First entry of the ordered axes is the X axis."""
        assert RightSidebar._axis_badge("n_batch", ["n_batch", "n_gpu"]) == "X"

    def test_second_axis_is_y(self):
        """Second entry of the ordered axes is the Y axis."""
        assert RightSidebar._axis_badge("n_gpu", ["n_batch", "n_gpu"]) == "Y"

    def test_non_axis_returns_none(self):
        """A dimension outside the ordered axes has no badge."""
        assert RightSidebar._axis_badge("other", ["n_batch", "n_gpu"]) is None

    def test_empty_axes_returns_none(self):
        """No active axes → no badge."""
        assert RightSidebar._axis_badge("dim", []) is None

    def test_single_axis_only_x(self):
        """2-D mode with one axis: only X exists, nothing maps to Y."""
        assert RightSidebar._axis_badge("x_dim", ["x_dim"]) == "X"
        assert RightSidebar._axis_badge("other", ["x_dim"]) is None


class TestExtendedSelectionMode:
    """Both listboxes use EXTENDED so plain click selects one item."""

    def test_filter_listbox_uses_extended_mode(self):
        """Filter value listboxes are created with selectmode=EXTENDED."""
        with _patched_rs() as (rs, lb_mock):
            rs._sections = {}
            rs._build_section("dim", ["a", "b"], set())
            call_kwargs = lb_mock.call_args.kwargs
            assert call_kwargs["selectmode"] == tk.EXTENDED

    def test_build_section_axis_badge_in_header(self):
        """An axis badge is prefixed to the section header text."""
        with _patched_rs() as (rs, _):
            rs._sections = {}
            with patch('view.right_sidebar.tk.Label') as mock_label:
                rs._build_section("n_batch", [512, 1024], {512},
                                  axis_label="X")
                header_text = mock_label.call_args.kwargs.get("text", "")
                assert header_text.startswith("  X:")
