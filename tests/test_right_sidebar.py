"""Tests for view/right_sidebar.py - Dimension filter sidebar."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from view.right_sidebar import RightSidebar


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

    @patch('view.right_sidebar.tk.Frame')
    @patch('view.right_sidebar.tk.Label')
    @patch('view.right_sidebar.tk.Listbox')
    @patch('view.right_sidebar.tk.Canvas')
    @patch('view.right_sidebar.ttk.Scrollbar')
    @patch('view.right_sidebar.tk.Button')
    def test_empty_sections(self, *mocks):
        """No sections → empty dict."""
        rs = RightSidebar(MagicMock())
        rs._sections = {}
        assert rs.get_current_filters() == {}

    @patch('view.right_sidebar.tk.Frame')
    @patch('view.right_sidebar.tk.Label')
    @patch('view.right_sidebar.tk.Listbox')
    @patch('view.right_sidebar.tk.Canvas')
    @patch('view.right_sidebar.ttk.Scrollbar')
    @patch('view.right_sidebar.tk.Button')
    def test_single_dimension(self, *mocks):
        """One dimension with selections."""
        rs = RightSidebar(MagicMock())
        lb = MagicMock()
        lb.curselection.return_value = (0, 2)
        rs._sections = {
            'gpu_name': {'listbox': lb, 'values': ['GPU_A', 'GPU_B', 'GPU_C']},
        }
        result = rs.get_current_filters()
        assert result == {'gpu_name': {'GPU_A', 'GPU_C'}}

    @patch('view.right_sidebar.tk.Frame')
    @patch('view.right_sidebar.tk.Label')
    @patch('view.right_sidebar.tk.Listbox')
    @patch('view.right_sidebar.tk.Canvas')
    @patch('view.right_sidebar.ttk.Scrollbar')
    @patch('view.right_sidebar.tk.Button')
    def test_multiple_dimensions(self, *mocks):
        """Multiple dimensions each return their selected values."""
        rs = RightSidebar(MagicMock())
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

    @patch('view.right_sidebar.tk.Frame')
    @patch('view.right_sidebar.tk.Label')
    @patch('view.right_sidebar.tk.Listbox')
    @patch('view.right_sidebar.tk.Canvas')
    @patch('view.right_sidebar.ttk.Scrollbar')
    @patch('view.right_sidebar.tk.Button')
    def test_no_selection(self, *mocks):
        """No selected indices → empty set for that dim."""
        rs = RightSidebar(MagicMock())
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

    @patch('view.right_sidebar.tk.Frame')
    @patch('view.right_sidebar.tk.Label')
    @patch('view.right_sidebar.tk.Listbox')
    @patch('view.right_sidebar.tk.Canvas')
    @patch('view.right_sidebar.ttk.Scrollbar')
    @patch('view.right_sidebar.tk.Button')
    def test_empty_dim_values(self, *mocks):
        """Empty dim_values → sections cleared, no _build_section call."""
        rs = RightSidebar(MagicMock())
        rs._inner_frame.winfo_children.return_value = []
        rs._sections = {'old': 'garbage'}

        with patch.object(rs, '_build_section') as mock_build:
            rs.update_filter_sections({}, set(), {})
            mock_build.assert_not_called()
            assert rs._sections == {}

    @patch('view.right_sidebar.tk.Frame')
    @patch('view.right_sidebar.tk.Label')
    @patch('view.right_sidebar.tk.Listbox')
    @patch('view.right_sidebar.tk.Canvas')
    @patch('view.right_sidebar.ttk.Scrollbar')
    @patch('view.right_sidebar.tk.Button')
    def test_all_dims_are_active_axes(self, *mocks):
        """Every dim is in active_axes → nothing to show."""
        rs = RightSidebar(MagicMock())
        rs._inner_frame.winfo_children.return_value = []

        with patch.object(rs, '_build_section') as mock_build:
            rs.update_filter_sections(
                {"dim1": ["a", "b"], "dim2": ["x", "y"]},
                {"dim1", "dim2"},
                {},
            )
            mock_build.assert_not_called()

    @patch('view.right_sidebar.tk.Frame')
    @patch('view.right_sidebar.tk.Label')
    @patch('view.right_sidebar.tk.Listbox')
    @patch('view.right_sidebar.tk.Canvas')
    @patch('view.right_sidebar.ttk.Scrollbar')
    @patch('view.right_sidebar.tk.Button')
    def test_excludes_dims_with_single_value(self, *mocks):
        """A dim with only 1 value is excluded (needs >1 for filtering)."""
        rs = RightSidebar(MagicMock())
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

    @patch('view.right_sidebar.tk.Frame')
    @patch('view.right_sidebar.tk.Label')
    @patch('view.right_sidebar.tk.Listbox')
    @patch('view.right_sidebar.tk.Canvas')
    @patch('view.right_sidebar.ttk.Scrollbar')
    @patch('view.right_sidebar.tk.Button')
    def test_builds_sections_for_inactive_dims(self, *mocks):
        """Non-active dims with >1 value → _build_section called for each."""
        rs = RightSidebar(MagicMock())
        rs._inner_frame.winfo_children.return_value = []

        with patch.object(rs, '_build_section') as mock_build:
            rs.update_filter_sections(
                {"dim_a": [1, 2], "dim_b": [10, 20], "dim_c": [1, 2]},
                {"dim_b"},
                {},
            )
            assert mock_build.call_count == 2
            called_dims = [call[0][0] for call in mock_build.call_args_list]
            assert "dim_a" in called_dims
            assert "dim_c" in called_dims
            assert "dim_b" not in called_dims

    @patch('view.right_sidebar.tk.Frame')
    @patch('view.right_sidebar.tk.Label')
    @patch('view.right_sidebar.tk.Listbox')
    @patch('view.right_sidebar.tk.Canvas')
    @patch('view.right_sidebar.ttk.Scrollbar')
    @patch('view.right_sidebar.tk.Button')
    def test_current_filters_passed_to_build(self, *mocks):
        """Current filter set is forwarded as the third argument."""
        rs = RightSidebar(MagicMock())
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

    @patch('view.right_sidebar.tk.Frame')
    @patch('view.right_sidebar.tk.Label')
    @patch('view.right_sidebar.tk.Listbox')
    @patch('view.right_sidebar.tk.Canvas')
    @patch('view.right_sidebar.ttk.Scrollbar')
    @patch('view.right_sidebar.tk.Button')
    def test_falls_back_to_all_values_when_no_filter(self, *mocks):
        """Dim without entry in current_filters → passes set(values) instead."""
        rs = RightSidebar(MagicMock())
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
