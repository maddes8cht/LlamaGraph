"""Tests for utils/startup_config.py and CLI/config merging in llamagraph.py."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Simple YAML parser ────────────────────────────────────────────────────


def test_parse_flat_pairs():
    from utils.startup_config import parse_simple_yaml
    raw = parse_simple_yaml(
        "mode_3d: true\nlevel_value: 75 # comment\n"
        'surface_style: "Shaded"\n'
    )
    assert raw == {"mode_3d": True, "level_value": 75,
                   "surface_style": "Shaded"}


def test_parse_rejects_nested_and_lists(tmp_path):
    from utils.startup_config import parse_simple_yaml
    with pytest.raises(ValueError):
        parse_simple_yaml("outer:\n  inner: true\n")
    with pytest.raises(ValueError):
        parse_simple_yaml("- item\n")


def test_parse_rejects_duplicate_key():
    from utils.startup_config import parse_simple_yaml
    with pytest.raises(ValueError):
        parse_simple_yaml("mode_3d: true\nmode_3d: false\n")


def test_coerce_ignores_unknown_and_bad_values(capsys):
    from utils.startup_config import coerce_config
    typed = coerce_config({"nope": 1, "level_value": 999,
                           "mode_3d": True}, source="test")
    assert typed == {"mode_3d": True}
    err = capsys.readouterr().err
    assert "unknown key" in err
    assert "level_value" in err


def test_load_missing_file_warns_and_returns_empty(tmp_path, capsys):
    from utils.startup_config import load_config_file
    result = load_config_file(tmp_path / "absent.yml")
    assert result == {}
    assert "cannot read config" in capsys.readouterr().err


# ── Auto-find ─────────────────────────────────────────────────────────────


def test_find_auto_config_only_in_script_dir(tmp_path):
    from utils import startup_config
    found = tmp_path / "llamagraph.config.yml"
    found.write_text("mode_3d: true\n")
    assert startup_config.find_auto_config(tmp_path) == found
    assert startup_config.find_auto_config(tmp_path / "sub") is None


def test_resolve_config_path_priority(tmp_path):
    from utils.startup_config import resolve_config_path
    auto = tmp_path / "llamagraph.config.yml"
    auto.write_text("mode_3d: true\n")
    other = tmp_path / "other.yml"
    other.write_text("mode_3d: false\n")
    assert resolve_config_path(None, True, tmp_path) is None
    assert resolve_config_path(other, False, tmp_path) == other
    assert resolve_config_path(None, False, tmp_path) == auto


# ── CLI precedence ────────────────────────────────────────────────────────


def test_cli_defaults_keep_old_behavior():
    import llamagraph
    with patch.object(sys, 'argv', ['llamagraph']):
        args = llamagraph.parse_args()
    assert args.path == Path('.')
    assert args.ns is False
    assert args.no_md is False
    assert args.config is None
    assert args.no_config is False
    assert args.normalize is None
    assert args.mode_3d is None


def test_auto_load_path_without_explicit_config(tmp_path):
    """No --config flag: llamagraph.config.yml in the script dir applies."""
    import llamagraph
    auto = tmp_path / "llamagraph.config.yml"
    auto.write_text("mode_3d: true\nlevel_value: 66\n")
    with patch.object(sys, 'argv', ['llamagraph']), \
         patch("utils.startup_config.script_dir", return_value=tmp_path):
        args = llamagraph.parse_args()
        settings, used = llamagraph.resolve_startup(args)
    assert used == auto
    assert settings["mode_3d"] is True
    assert settings["level_value"] == 66


def test_config_values_apply_without_cli(tmp_path):
    import llamagraph
    cfg = tmp_path / "c.yml"
    cfg.write_text("latency_ns: true\nuse_md: false\nmode_3d: true\n"
                   "level_value: 80\nsurface_style: Shaded\n")
    with patch.object(sys, 'argv', ['llamagraph', '--config', str(cfg)]):
        args = llamagraph.parse_args()
        settings, used = llamagraph.resolve_startup(args)
    assert used == cfg
    assert settings["default_ts"] is False
    assert settings["show_md"] is False
    assert settings["mode_3d"] is True
    assert settings["level_value"] == 80
    assert settings["surface_style"] == "Shaded"


def test_cli_wins_over_config(tmp_path):
    import llamagraph
    cfg = tmp_path / "c.yml"
    cfg.write_text("latency_ns: true\nnormalize: true\nmode_3d: true\n")
    argv = ['llamagraph', '--config', str(cfg), '--ts', '--no-normalize',
            '--no-mode-3d']
    with patch.object(sys, 'argv', argv):
        args = llamagraph.parse_args()
        settings, _ = llamagraph.resolve_startup(args)
    assert settings["default_ts"] is True
    assert settings["normalize"] is False
    assert settings["mode_3d"] is False


def test_no_config_ignores_file(tmp_path):
    import llamagraph
    cfg = tmp_path / "c.yml"
    cfg.write_text("mode_3d: true\n")
    argv = ['llamagraph', '--config', str(cfg), '--no-config']
    with patch.object(sys, 'argv', argv):
        args = llamagraph.parse_args()
        settings, used = llamagraph.resolve_startup(args)
    assert used is None
    assert settings["mode_3d"] is False


def test_explicit_missing_config_exits(tmp_path):
    import llamagraph
    missing = tmp_path / "absent.yml"
    with patch.object(sys, 'argv',
                      ['llamagraph', '--config', str(missing)]):
        args = llamagraph.parse_args()
        with pytest.raises(SystemExit):
            llamagraph.resolve_startup(args)


def test_main_applies_example_config_2d_3d_state(tmp_path):
    """Proof path: --config example.llamagraph.config.yml drives startup."""
    import llamagraph
    example = Path(__file__).absolute().parent.parent \
        / "example.llamagraph.config.yml"
    assert example.is_file()
    mw = MagicMock()
    pp = MagicMock()
    argv = ['llamagraph', str(tmp_path), '--config', str(example)]
    with patch.object(sys, 'argv', argv), \
         patch('tkinter.Tk'), \
         patch('llamagraph.MainWindow', return_value=mw) as mock_win, \
         patch('llamagraph.PlotterPresenter', return_value=pp) as mock_pp:
        llamagraph.main()
        _, wkwargs = mock_win.call_args
        _, pkwargs = mock_pp.call_args
        # Example file holds built-in defaults: 2D tokens/s start.
        assert wkwargs['mode_3d'] is False
        assert wkwargs['normalize'] is False
        assert wkwargs['surface_style'] == "Solid"
        assert wkwargs['interp_method'] == "Cubic"
        assert pkwargs['default_ts'] is True
        assert pkwargs['show_md'] is True


def test_main_config_3d_start_state(tmp_path):
    """A 3D config flips the initial 2D/3D startup state."""
    import llamagraph
    cfg = tmp_path / "start3d.yml"
    cfg.write_text("mode_3d: true\nlatency_ns: true\nnormalize: true\n"
                   "show_level: true\nlevel_value: 75\n"
                   "surface_style: Colormap\nsubdiv_level: 2\n"
                   "interp_method: Linear\nmask_gaps: true\n"
                   "show_wireframe: true\nshow_errors: false\n"
                   "show_projections: true\ndolly: false\n"
                   "z_label_mode: '%'\nunify: true\n"
                   "show_pp: false\nsurface_visible: false\n")
    mw = MagicMock()
    pp = MagicMock()
    argv = ['llamagraph', str(tmp_path), '--config', str(cfg)]
    with patch.object(sys, 'argv', argv), \
         patch('tkinter.Tk'), \
         patch('llamagraph.MainWindow', return_value=mw) as mock_win, \
         patch('llamagraph.PlotterPresenter', return_value=pp) as mock_pp:
        llamagraph.main()
        _, wkwargs = mock_win.call_args
        _, pkwargs = mock_pp.call_args
        assert wkwargs['mode_3d'] is True
        assert wkwargs['normalize'] is True
        assert wkwargs['show_level'] is True
        assert wkwargs['level_value'] == 75
        assert wkwargs['surface_style'] == "Colormap"
        assert wkwargs['subdiv_level'] == 2
        assert wkwargs['interp_method'] == "Linear"
        assert wkwargs['mask_gaps'] is True
        assert wkwargs['show_wireframe'] is True
        assert wkwargs['show_errors'] is False
        assert wkwargs['show_projections'] is True
        assert wkwargs['dolly'] is False
        assert wkwargs['z_label_mode'] == "%"
        assert wkwargs['unify'] is True
        assert wkwargs['show_pp'] is False
        assert wkwargs['show_surface'] is False
        assert pkwargs['default_ts'] is False
