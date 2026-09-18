"""Tests for llamagraph.py - Entry point with CLI argument parsing and main()."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── parse_args tests (pure function, only depends on sys.argv) ────────────────


def test_parse_args_default():
    """No arguments → default path='.' and ts=True (--ns=False)."""
    import llamagraph
    with patch.object(sys, 'argv', ['llamagraph']):
        args = llamagraph.parse_args()
    assert args.path == Path('.')
    assert args.ns is False


def test_parse_args_custom_path():
    """Path argument is forwarded."""
    import llamagraph
    with patch.object(sys, 'argv', ['llamagraph', '/some/dir']):
        args = llamagraph.parse_args()
    assert args.path == Path('/some/dir')


def test_parse_args_ns_flag():
    """--ns flag sets ns=True."""
    import llamagraph
    with patch.object(sys, 'argv', ['llamagraph', '--ns']):
        args = llamagraph.parse_args()
    assert args.ns is True


def test_parse_args_path_with_ns():
    """Path and --ns together."""
    import llamagraph
    with patch.object(sys, 'argv', ['llamagraph', '/path', '--ns']):
        args = llamagraph.parse_args()
    assert args.path == Path('/path')
    assert args.ns is True


def test_parse_args_no_md_flag():
    """--no-md defaults to False and is settable."""
    import llamagraph
    with patch.object(sys, 'argv', ['llamagraph']):
        assert llamagraph.parse_args().no_md is False
    with patch.object(sys, 'argv', ['llamagraph', '--no-md']):
        assert llamagraph.parse_args().no_md is True


# ── main() path logic tests (GUI creation mocked) ────────────────────────────


def test_main_directory_path(tmp_path):
    """A directory path sets start_dir and no initial_file."""
    import llamagraph
    mw = MagicMock()
    pp = MagicMock()
    with patch.object(sys, 'argv', ['llamagraph', str(tmp_path)]), \
         patch('tkinter.Tk') as mock_tk, \
         patch('llamagraph.MainWindow', return_value=mw), \
         patch('llamagraph.PlotterPresenter', return_value=pp) as mock_pp:
        llamagraph.main()
        mock_pp.assert_called_once()
        _, kwargs = mock_pp.call_args
        assert kwargs['start_dir'] == tmp_path
        assert kwargs['initial_selection_file'] is None
        assert kwargs['default_ts'] is True


def test_main_csv_file_path(tmp_path):
    """A valid llama-bench CSV sets initial_file."""
    import llamagraph
    csv_file = tmp_path / "bench.csv"
    csv_file.write_text("n_prompt,n_gen,avg_ts,avg_ns\n1024,0,100.5,50000\n")
    mw = MagicMock()
    pp = MagicMock()
    with patch.object(sys, 'argv', ['llamagraph', str(csv_file)]), \
         patch('tkinter.Tk') as mock_tk, \
         patch('llamagraph.MainWindow', return_value=mw), \
         patch('llamagraph.PlotterPresenter', return_value=pp) as mock_pp:
        llamagraph.main()
        _, kwargs = mock_pp.call_args
        assert kwargs['start_dir'] == tmp_path
        assert kwargs['initial_selection_file'] == csv_file


def test_main_non_csv_file_path(tmp_path):
    """A file without llama-bench headers does NOT set initial_file."""
    import llamagraph
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1,col2\n1,2\n")
    mw = MagicMock()
    pp = MagicMock()
    with patch.object(sys, 'argv', ['llamagraph', str(csv_file)]), \
         patch('tkinter.Tk') as mock_tk, \
         patch('llamagraph.MainWindow', return_value=mw), \
         patch('llamagraph.PlotterPresenter', return_value=pp) as mock_pp:
        llamagraph.main()
        _, kwargs = mock_pp.call_args
        assert kwargs['start_dir'] == tmp_path
        assert kwargs['initial_selection_file'] is None


def test_main_nonexistent_path():
    """Non-existent path → sys.exit(1)."""
    import llamagraph
    with patch.object(sys, 'argv', ['llamagraph', '/nonexistent/path/12345']):
        with pytest.raises(SystemExit) as exc_info:
            llamagraph.main()
        assert exc_info.value.code == 1


def test_main_ns_flag(tmp_path):
    """--ns flag sets default_ts=False."""
    import llamagraph
    mw = MagicMock()
    pp = MagicMock()
    with patch.object(sys, 'argv', ['llamagraph', str(tmp_path), '--ns']), \
         patch('tkinter.Tk') as mock_tk, \
         patch('llamagraph.MainWindow', return_value=mw), \
         patch('llamagraph.PlotterPresenter', return_value=pp) as mock_pp:
        llamagraph.main()
        _, kwargs = mock_pp.call_args
        assert kwargs['default_ts'] is False


def test_main_no_md_ignores_md_file_with_warning(tmp_path, capsys):
    """--no-md + .md file path: no initial_file, warning on stderr."""
    import llamagraph
    md_file = tmp_path / "bench.md"
    md_file.write_text(
        "| model | test | t/s |\n| ----- | ----: | ---: |\n"
        "| m1 | pp512 | 50.0 ± 1.0 |\n"
    )
    mw = MagicMock()
    pp = MagicMock()
    with patch.object(sys, 'argv', ['llamagraph', str(md_file), '--no-md']), \
         patch('tkinter.Tk'), \
         patch('llamagraph.MainWindow', return_value=mw), \
         patch('llamagraph.PlotterPresenter', return_value=pp) as mock_pp:
        llamagraph.main()
        _, kwargs = mock_pp.call_args
        assert kwargs['initial_selection_file'] is None
        assert kwargs['show_md'] is False
    assert "--no-md" in capsys.readouterr().err


def test_main_invalid_file_warns(tmp_path, capsys):
    """Unrecognized file: no initial_file, warning on stderr."""
    import llamagraph
    bad = tmp_path / "notes.txt"
    bad.write_text("just text\n")
    mw = MagicMock()
    pp = MagicMock()
    with patch.object(sys, 'argv', ['llamagraph', str(bad)]), \
         patch('tkinter.Tk'), \
         patch('llamagraph.MainWindow', return_value=mw), \
         patch('llamagraph.PlotterPresenter', return_value=pp) as mock_pp:
        llamagraph.main()
        _, kwargs = mock_pp.call_args
        assert kwargs['initial_selection_file'] is None
    assert "not a recognized" in capsys.readouterr().err


def test_main_show_md_default_and_no_md(tmp_path):
    """show_md defaults to True; --no-md sets it to False."""
    import llamagraph
    for argv, expected in ((['llamagraph', str(tmp_path)], True),
                           (['llamagraph', str(tmp_path), '--no-md'], False)):
        mw = MagicMock()
        pp = MagicMock()
        with patch.object(sys, 'argv', argv), \
             patch('tkinter.Tk'), \
             patch('llamagraph.MainWindow', return_value=mw), \
             patch('llamagraph.PlotterPresenter', return_value=pp) as mock_pp:
            llamagraph.main()
            _, kwargs = mock_pp.call_args
            assert kwargs['show_md'] is expected
