#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llamagraph.py — llamagraph entry point

Parses CLI arguments, applies the optional YAML startup configuration,
creates the Tkinter root, builds the MainWindow, instantiates the
Presenter (which wires everything together), and starts the event loop.

Startup configuration (all English):
  - ``llamagraph.config.yml`` next to this script is used automatically
    when present.
  - ``--config <file>`` uses any other config file instead.
  - ``--no-config`` ignores every config file.
  - Explicit CLI options always win over config values; the negated
    forms (``--ts`` vs ``--ns``, ``--md`` vs ``--no-md``,
    ``--no-normalize`` vs ``--normalize``, ...) opt out of a config
    default per setting.

This file stays minimal: orchestration logic lives in the Presenter,
flat-file parsing lives in utils/startup_config.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import tkinter as tk

from view.main_window import MainWindow
from presenter.plotter_presenter import PlotterPresenter
from utils.colors import DEFAULT_PP_COLOR, DEFAULT_TG_COLOR
from utils.csv_parser import is_llama_bench_csv, is_llama_bench_md
from utils.startup_config import (
    INTERP_METHODS,
    PROJECTION_MODES,
    SURFACE_STYLES,
    Z_LABEL_MODES,
    effective_value,
    load_config_file,
    resolve_config_path,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="llamagraph — llama-bench benchmark visualizer"
    )
    parser.add_argument(
        'path',
        nargs='?',
        type=Path,
        default=Path('.'),
        help="Directory or specific CSV/MD file to load (default: current dir)",
    )
    parser.add_argument(
        '--ns',
        action='store_true',
        help="Start in latency (ns) view instead of tokens/s",
    )
    parser.add_argument(
        '--ts',
        action='store_true',
        help="Start in throughput (tokens/s) view (overrides config latency_ns)",
    )
    parser.add_argument(
        '--no-md',
        action='store_true',
        help="Hide .md files entirely (only CSV files are listed, no .md toggle button)",
    )
    parser.add_argument(
        '--md',
        action='store_true',
        help="List .md files (overrides config use_md)",
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help="Use the given YAML startup config file instead of the "
             "auto-detected llamagraph.config.yml",
    )
    parser.add_argument(
        '--no-config',
        action='store_true',
        help="Ignore every config file and use built-in startup defaults",
    )
    # Startup state (each flag overrides the same-named config key;
    # the negated form opts out of a config default).
    parser.add_argument('--normalize', dest='normalize',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Start with normalized values (or not)")
    parser.add_argument('--mode-3d', dest='mode_3d',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Start in 3D mode (or 2D)")
    parser.add_argument('--show-level', dest='show_level',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Show the 3D level plane at startup (or not)")
    parser.add_argument('--level-value', dest='level_value', type=int,
                        default=None, metavar='0-100',
                        help="Level plane value at startup (0-100)")
    parser.add_argument('--surface', dest='surface_visible',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Show the 3D surface at startup (or not)")
    parser.add_argument('--surface-style', dest='surface_style',
                        choices=list(SURFACE_STYLES), default=None,
                        help="3D surface style at startup")
    parser.add_argument('--subdiv-level', dest='subdiv_level', type=int,
                        default=None, metavar='0-4',
                        help="3D subdivision level at startup (0-4)")
    parser.add_argument('--interp-method', dest='interp_method',
                        choices=list(INTERP_METHODS), default=None,
                        help="3D interpolation mode at startup")
    parser.add_argument('--mask-gaps', dest='mask_gaps',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Mask unmeasured gaps at startup (or not)")
    parser.add_argument('--wireframe', dest='show_wireframe',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Show the 3D wireframe at startup (or not)")
    parser.add_argument('--errors', dest='show_errors',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Show 3D error bars at startup (or not)")
    parser.add_argument('--projections', dest='show_projections',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Deprecated: use --projection-mode (true maps "
                             "to 'back', false to 'none')")
    parser.add_argument('--projection-mode', dest='projection_mode',
                        choices=list(PROJECTION_MODES), default=None,
                        help="3D wall projections at startup")
    parser.add_argument('--dolly', dest='dolly',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Use dolly (roll-locked) rotation at startup (or not)")
    parser.add_argument('--z-label-mode', dest='z_label_mode',
                        choices=list(Z_LABEL_MODES), default=None,
                        help="Z label mode at startup")
    parser.add_argument('--unify', dest='unify',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Start with unified series (or not)")
    parser.add_argument('--show-pp', dest='show_pp',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Show PP series at startup (or not)")
    parser.add_argument('--show-tg', dest='show_tg',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Show TG series at startup (or not)")
    return parser.parse_args(argv)


def _check_int_range(value: int, low: int, high: int, name: str) -> int:
    """Validate a CLI integer is within range (English error otherwise)."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        print(f"Error: option '{name}' expects an integer, "
              f"got {value!r}.", file=sys.stderr)
        sys.exit(2)
    if not (low <= number <= high):
        print(f"Error: option '{name}' expects {low}-{high}, "
              f"got {number}.", file=sys.stderr)
        sys.exit(2)
    return number


def resolve_startup(args: argparse.Namespace) -> tuple[dict, Optional[Path]]:
    """
    Merge CLI options over the YAML startup config.

    Returns (settings, config_path_used). settings holds plain startup
    values with the config keys from utils/startup_config.DEFAULTS
    plus 'default_ts'/'show_md' derived for the Presenter.
    """
    config_path = resolve_config_path(args.config, args.no_config)
    config: dict = {}
    if config_path is not None:
        if args.config is not None and not config_path.is_file():
            print(f"Error: config file '{config_path}' does not exist.",
                  file=sys.stderr)
            sys.exit(1)
        if config_path.is_file():
            config = load_config_file(config_path)

    # Metric: --ns forces latency, --ts forces throughput, else config.
    if args.ns and args.ts:
        print("Error: options '--ns' and '--ts' are mutually exclusive.",
              file=sys.stderr)
        sys.exit(2)
    if args.ns:
        latency_ns = True
    elif args.ts:
        latency_ns = False
    else:
        latency_ns = bool(config.get("latency_ns", False))

    # Markdown files: --no-md hides, --md shows, else config.
    if args.no_md and args.md:
        print("Error: options '--no-md' and '--md' are mutually exclusive.",
              file=sys.stderr)
        sys.exit(2)
    if args.no_md:
        use_md = False
    elif args.md:
        use_md = True
    else:
        use_md = bool(config.get("use_md", True))

    # Data path: an explicit positional path wins; the config default
    # applies only when the positional argument was left at '.'. Use
    # --no-config to force '.' despite a configured data_path.
    if args.path != Path('.'):
        data_path = args.path
    elif "data_path" in config:
        data_path = Path(str(config["data_path"])).expanduser()
    else:
        data_path = Path('.')

    settings: dict = {
        "data_path": data_path,
        "latency_ns": latency_ns,
        "use_md": use_md,
        "default_ts": not latency_ns,
        "show_md": use_md,
    }
    tri_bool_keys = ("normalize", "mode_3d", "show_level",
                     "surface_visible", "mask_gaps", "show_wireframe",
                     "show_errors", "dolly",
                     "unify", "show_pp", "show_tg")
    for key in tri_bool_keys:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            settings[key] = bool(cli_value)
        else:
            settings[key] = effective_value(None, config, key)

    # Projection mode: explicit --projection-mode wins, then legacy
    # --projections/--no-projections, then config (incl. legacy
    # show_projections boolean), then the built-in default.
    if args.projection_mode is not None:
        settings["projection_mode"] = args.projection_mode
    elif getattr(args, "show_projections", None) is not None:
        settings["projection_mode"] = \
            "back" if args.show_projections else "none"
    else:
        settings["projection_mode"] = effective_value(
            None, config, "projection_mode")

    if args.level_value is not None:
        settings["level_value"] = _check_int_range(
            args.level_value, 0, 100, "--level-value")
    else:
        settings["level_value"] = effective_value(None, config, "level_value")
    if args.subdiv_level is not None:
        settings["subdiv_level"] = _check_int_range(
            args.subdiv_level, 0, 4, "--subdiv-level")
    else:
        settings["subdiv_level"] = effective_value(
            None, config, "subdiv_level")
    for key in ("surface_style", "interp_method", "z_label_mode"):
        cli_value = getattr(args, key, None)
        settings[key] = effective_value(cli_value, config, key)

    used_path = config_path if config_path is not None \
        and config_path.is_file() else None
    # An explicitly requested but missing file already exited above;
    # report the actually applied file (None = built-in defaults).
    if args.no_config:
        used_path = None
    return settings, used_path


def main() -> None:
    args = parse_args()
    settings, _config_used = resolve_startup(args)
    input_path = Path(settings["data_path"]).absolute()

    start_dir: Path
    initial_file: Optional[Path] = None

    # Logic: If a file is provided, get its parent and check if it's a valid file
    if input_path.is_file():
        start_dir = input_path.parent
        if is_llama_bench_csv(input_path):
            initial_file = input_path
        elif is_llama_bench_md(input_path):
            if not settings["show_md"]:
                print(f"Warning: ignoring '{input_path.name}': "
                      f".md files are disabled with --no-md.", file=sys.stderr)
            else:
                initial_file = input_path
        else:
            print(f"Warning: ignoring '{input_path.name}': "
                  f"not a recognized llama-bench CSV/MD file.", file=sys.stderr)
    else:
        start_dir = input_path

    if not start_dir.exists():
        print(f"Error: path '{start_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()

    # Build the window (View layer) with the resolved startup state
    window = MainWindow(
        root,
        pp_color=DEFAULT_PP_COLOR,
        tg_color=DEFAULT_TG_COLOR,
        show_pp=settings["show_pp"],
        show_tg=settings["show_tg"],
        unify=settings["unify"],
        normalize=settings["normalize"],
        mode_3d=settings["mode_3d"],
        show_surface=settings["surface_visible"],
        surface_style=settings["surface_style"],
        show_wireframe=settings["show_wireframe"],
        projection_mode=settings["projection_mode"],
        show_errors=settings["show_errors"],
        dolly=settings["dolly"],
        z_label_mode=settings["z_label_mode"],
        show_level=settings["show_level"],
        level_value=settings["level_value"],
        subdiv_level=settings["subdiv_level"],
        interp_method=settings["interp_method"],
        mask_gaps=settings["mask_gaps"],
    )

    # Instantiate the Presenter
    _presenter = PlotterPresenter(
        window=window,
        start_dir=start_dir,
        pp_color=DEFAULT_PP_COLOR,
        tg_color=DEFAULT_TG_COLOR,
        default_ts=settings["default_ts"],
        initial_selection_file=initial_file,
        show_md=settings["show_md"],
    )

    root.mainloop()


if __name__ == "__main__":
    main()
