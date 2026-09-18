#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — llamagraph entry point

Parses CLI arguments, creates the Tkinter root, builds the MainWindow,
instantiates the Presenter (which wires everything together), and starts
the event loop.

This file should stay minimal: orchestration logic lives in the Presenter.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import tkinter as tk

from view.main_window import MainWindow
from presenter.plotter_presenter import PlotterPresenter
from utils.colors import DEFAULT_PP_COLOR, DEFAULT_TG_COLOR
from utils.csv_parser import is_llama_bench_csv, is_llama_bench_md


def parse_args() -> argparse.Namespace:
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
        '--no-md',
        action='store_true',
        help="Hide .md files entirely (only CSV files are listed, no .md toggle button)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.path.absolute()

    start_dir: Path
    initial_file: Optional[Path] = None

    # Logic: If a file is provided, get its parent and check if it's a valid file
    if input_path.is_file():
        start_dir = input_path.parent
        if is_llama_bench_csv(input_path):
            initial_file = input_path
        elif is_llama_bench_md(input_path):
            if args.no_md:
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

    # Build the window (View layer)
    window = MainWindow(
        root,
        pp_color=DEFAULT_PP_COLOR,
        tg_color=DEFAULT_TG_COLOR,
    )

    # Instantiate the Presenter
    _presenter = PlotterPresenter(
        window=window,
        start_dir=start_dir,
        pp_color=DEFAULT_PP_COLOR,
        tg_color=DEFAULT_TG_COLOR,
        default_ts=not args.ns,
        initial_selection_file=initial_file,
        show_md=not args.no_md,
    )

    root.mainloop()


if __name__ == "__main__":
    main()