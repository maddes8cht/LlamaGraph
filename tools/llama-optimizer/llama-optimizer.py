#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llama-optimizer.py (v3.0 + Grid Search Mode + Weighted Ranking)
Sequential & Grid parameter wrapper for llama-bench.
- Supports sequential phase-by-phase optimization (default, TG-only)
- Supports full matrix/grid search via --grid CLI flag or ::grid directive
- Grid mode ranks all combinations with weighted PP/TG scores
  (see model/ranking.py; --top / --table-weights)
- Extracts system/model metadata from the first CSV output
- Logs all phases, warnings, and final recommendations to a unified .md file
- Maintains consistent timestamping for run grouping
"""
import os
import sys
import subprocess
import csv
import time
import re
from pathlib import Path
import argparse
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict, OrderedDict
from math import prod
from functools import reduce
import operator

# Shared weighted PP/TG ranking (single source of truth in model/ranking.py).
# The optimizer stays runnable standalone from its own directory: fall back
# to adding the repo root (two levels up) to sys.path when needed.
try:
    from model import ranking as ranking_mod
except ModuleNotFoundError:
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from model import ranking as ranking_mod

# ==================== Color Constants ====================
CLR_CYAN = "\033[96m"
CLR_GREEN = "\033[92m"
CLR_YELLOW = "\033[93m"
CLR_RED = "\033[91m"
CLR_RESET = "\033[0m"
BORDER_CHAR = "═"
BOX_WIDTH = 70

def print_cmd_box(cmd_str: str) -> None:
    """Prints a formatted box around the benchmark command for better CLI visibility."""
    line = BORDER_CHAR * BOX_WIDTH
    print(f"\n{CLR_CYAN}{line}{CLR_RESET}")
    print(f"{CLR_CYAN}║ 🚀 llama-bench run:{' ' * (BOX_WIDTH - 4)} ║{CLR_RESET}")
    print(f"{CLR_CYAN}║ {cmd_str}{' ' * max(1, BOX_WIDTH - 5 - len(cmd_str))} ║{CLR_RESET}")
    print(f"{CLR_CYAN}{line}{CLR_RESET}\n")

def select_file(initial_dir: Path, title: str, file_ext: str) -> Path | None:
    """Opens a native file dialog to select a file."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        initialdir=str(initial_dir), title=title, filetypes=[(title, file_ext)]
    )
    return Path(path) if path else None

def load_params_recursive(file_path: Path) -> tuple[dict, list, OrderedDict]:
    """
    Recursively loads configuration, base arguments, and optimization targets from a params file.
    Supports ::optimize, ::grid, ::optimize-order, and ::params-file directives.
    """
    configs = {}
    llama_args = []
    optimize_targets = OrderedDict()
    
    if not file_path.exists():
        return configs, llama_args, optimize_targets
        
    print(f"{CLR_CYAN}Loading params file:{CLR_RESET} {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '#' in line:
                line = line.split('#', 1)[0].strip()
                
            if line.lower().startswith('::optimize '):
                opt_content = line[11:].strip()
                match = re.match(r'(-{1,2}\S+)\s+(.*)', opt_content)
                if match:
                    flag = match.group(1)
                    raw_part = match.group(2)
                    # Values may be comma-separated (64,128,256) or
                    # space-separated (64 128 256); tokens with '='
                    # are metadata and ignored in both cases.
                    values = []
                    if ',' in raw_part:
                        for segment in raw_part.split(','):
                            tokens = segment.strip().split()
                            if not tokens: continue
                            val = tokens[0]
                            if '=' not in val:
                                values.append(val)
                    else:
                        for token in raw_part.split():
                            if '=' not in token:
                                values.append(token)
                    if values and flag not in optimize_targets:
                        optimize_targets[flag] = values
                        
            elif line.lower() == '::grid':
                configs['grid'] = True
                
            elif line.startswith('::'):
                parts = line[2:].strip().split(None, 1)
                key = parts[0].lower()
                val = parts[1].strip().strip('"\'') if len(parts) > 1 else True
                configs[key] = val
                
            elif line.startswith('-'):
                llama_args.extend(line.split())
                
    # Handle recursive includes: the including (current) file always wins over
    # the included one — directives are overridden, base args are appended
    # after the included ones (later CLI occurrences win in llama-bench),
    # and optimization targets keep the current file's values per flag.
    if 'params-file' in configs:
        next_file = Path(configs['params-file'])
        sub_configs, sub_args, sub_targets = load_params_recursive(next_file)
        configs = {**sub_configs, **configs}
        llama_args = sub_args + llama_args
        for f, v in sub_targets.items():
            if f not in optimize_targets:
                optimize_targets[f] = v
                
    return configs, llama_args, optimize_targets

# ==================== CSV Column Mapping ====================
CLI_TO_CSV_COL = {
    "--batch-size": "n_batch", "-b": "n_batch",
    "--ubatch-size": "n_ubatch", "-ub": "n_ubatch",
    "--n-gpu-layers": "n_gpu_layers", "-ngl": "n_gpu_layers",
    "--threads": "n_threads", "-t": "n_threads",
    "--n-prompt": "n_prompt", "-p": "n_prompt",
    "--n-gen": "n_gen", "-n": "n_gen",
    "--flash-attn": "flash_attn", "-fa": "flash_attn",
    "--cache-type-k": "type_k", "-ctk": "type_k",
    "--cache-type-v": "type_v", "-ctv": "type_v",
    "--no-kv-offload": "no_kv_offload", "-nkvo": "no_kv_offload",
    "--embeddings": "embeddings", "-embd": "embeddings",
    "--main-gpu": "main_gpu", "-mg": "main_gpu",
    "--split-mode": "split_mode", "-sm": "split_mode",
    "--device": "devices", "-dev": "devices",
    "--fit-target": "fit_target",
    "--fit-ctx": "fit_ctx",
}

def get_csv_column(flag: str) -> str:
    """Maps a CLI flag to its corresponding llama-bench CSV column name."""
    return CLI_TO_CSV_COL.get(flag.lower(), flag.lstrip('-').replace('-', '_').lower())

# ==================== Dual Output (CSV + Markdown) ====================
# llama-bench writes a single format to stdout (-o) and optionally a second
# format to stderr (-oe). We use stdout=CSV (machine-readable, for
# LlamaGraph) and stderr=Markdown (human-readable), each in its own file.
OUTPUT_FORMATS = ('both', 'csv', 'md')
_OUTPUT_TAKES_VALUE = frozenset({'-o', '--output', '-oe', '--output-err'})


def resolve_output_format(cli_val, configs: dict) -> str:
    """CLI --output-format wins, then ::output-format from params file, else 'both'."""
    if cli_val is not None:
        return cli_val
    cfg_val = configs.get('output-format', 'both')
    cfg_val = str(cfg_val).strip().lower() if cfg_val is not True else 'both'
    return cfg_val if cfg_val in OUTPUT_FORMATS else 'both'


def strip_output_format_args(args: list) -> list:
    """Remove user-supplied -o/--output/-oe/--output-err (plus values) so the
    resolved --output-format is the single source of truth."""
    cleaned = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg in _OUTPUT_TAKES_VALUE:
            skip = True
            continue
        if arg.startswith('-o=') or arg.startswith('--output=') or arg.startswith('--output-err='):
            continue
        cleaned.append(arg)
    return cleaned


def output_args_for(fmt: str) -> list:
    """llama-bench flags for the requested format."""
    if fmt == 'csv':
        return ['-o', 'csv']
    if fmt == 'md':
        return ['-o', 'md']
    return ['-o', 'csv', '-oe', 'md']  # 'both': csv->stdout, md->stderr


# ==================== Weighted Ranking Options (grid mode only) ====================
# Table columns run from pure TG to increasing PP share, stored TG-first:
#   TG | 70/30 | 50/50 | 30/70 | PP  ==  1/0 | 0.7/0.3 | 0.5/0.5 | 0.3/0.7 | 0/1
# Scoring itself lives in model/ranking.py; here only CLI/config resolution.

def resolve_top(cli_val, configs: dict) -> int:
    """CLI --top wins, then ::top from params file, else 1 (rows per weighting).

    A present-but-invalid value warns and falls back to 1; a missing
    value stays silent (it is just the default).
    """
    raw = cli_val if cli_val is not None else configs.get('top', None)
    if raw is None or raw is True:
        return 1
    try:
        top = int(str(raw).strip())
    except (ValueError, TypeError, AttributeError):
        print(f"{CLR_YELLOW}⚠ Invalid ::top value ({raw!r}); using 1.{CLR_RESET}")
        return 1
    if top < 1:
        print(f"{CLR_YELLOW}⚠ Invalid ::top value ({raw!r}); using 1.{CLR_RESET}")
        return 1
    return top


def resolve_table_weights(cli_val, configs: dict, strict: bool = False) -> list[tuple[float, float]]:
    """CLI --table-weights wins, then ::table-weights, else the TG..PP default.

    Returns [(w_tg, w_pp), ...]. With strict=False (default) a parse
    error warns and falls back to the default; with strict=True the
    ValueError propagates so callers (e.g. main()) can abort instead.
    """
    raw = cli_val if cli_val is not None else configs.get(
        'table-weights', ranking_mod.DEFAULT_WEIGHTS_SPEC)
    if raw is True:
        raw = ranking_mod.DEFAULT_WEIGHTS_SPEC
    try:
        return ranking_mod.parse_weights(str(raw))
    except ValueError as exc:
        if strict:
            raise
        print(f"{CLR_YELLOW}⚠ Invalid weighting spec ({exc}); using default.{CLR_RESET}")
        return ranking_mod.default_weights()


def build_grid_param_map(target_flags: list[str]) -> dict[str, str]:
    """Map optimizer CLI flags to their llama-bench CSV columns."""
    return {flag: get_csv_column(flag) for flag in target_flags}


def run_bench(cmd: list, csv_file: Path | None, md_file: Path | None) -> None:
    """Run llama-bench, routing stdout to the CSV file and stderr to the MD file.

    Either path may be None (format not requested). Raises
    subprocess.CalledProcessError on failure, like the previous inline code.
    """
    if csv_file is not None and md_file is not None:
        with open(csv_file, 'w', encoding='utf-8') as out_f, \
             open(md_file, 'w', encoding='utf-8') as err_f:
            subprocess.run(cmd, stdout=out_f, stderr=err_f, text=True, check=True)
    elif csv_file is not None:
        with open(csv_file, 'w', encoding='utf-8') as out_f:
            subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True, check=True)
    else:
        with open(md_file, 'w', encoding='utf-8') as out_f:  # type: ignore[arg-type]
            subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True, check=True)

def parse_bench_output(csv_path: Path, target_flag: str) -> dict:
    """
    Parses a sequential-phase CSV file.
    Returns a dictionary mapping each tested value of the target flag to its best TG t/s.
    """
    results = defaultdict(float)
    target_col = get_csv_column(target_flag)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return results
            
        matched_col = next((col for col in reader.fieldnames if col.lower() == target_col.lower()), None)
        if not matched_col:
            print(f"{CLR_YELLOW}⚠ Warning: Column '{target_col}' not found in CSV.{CLR_RESET}")
            return results
            
        perf_col = next((col for col in ['avg_ts', 't/s'] if col in reader.fieldnames), None)
        if not perf_col:
            print(f"{CLR_YELLOW}⚠ Warning: Performance column (avg_ts/t/s) not found in CSV.{CLR_RESET}")
            return results
            
        for row in reader:
            is_tg = 'tg' in row.get('test', '').lower()
            if not is_tg:
                try:
                    if int(row.get('n_prompt', -1)) == 0 and int(row.get('n_gen', -1)) > 0:
                        is_tg = True
                except (ValueError, TypeError):
                    pass
            if not is_tg:
                continue
                
            try:
                tg_val = float(row.get(perf_col, 0))
                param_val = str(row.get(matched_col, '')).strip()
                if param_val and tg_val > results[param_val]:
                    results[param_val] = tg_val
            except (ValueError, TypeError):
                continue
    return dict(results)

def parse_grid_output(csv_path: Path, target_flags: list[str]) -> tuple[dict, float]:
    """
    Legacy TG-only grid parser: finds the single configuration with the
    highest TG performance. Returns ({flag: best_value}, max TG t/s).

    Kept for backwards compatibility (and its tests); grid ranking in
    main() now uses model/ranking.py (weighted PP+TG) instead.
    """
    best_row = None
    max_tg = -1.0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}, 0.0
            
        perf_col = next((col for col in ['avg_ts', 't/s'] if col in reader.fieldnames), None)
        if not perf_col:
            return {}, 0.0
            
        for row in reader:
            is_tg = 'tg' in row.get('test', '').lower()
            if not is_tg:
                try:
                    if int(row.get('n_prompt', -1)) == 0 and int(row.get('n_gen', -1)) > 0:
                        is_tg = True
                except (ValueError, TypeError):
                    pass
            if not is_tg:
                continue
                
            try:
                tg_val = float(row.get(perf_col, 0))
                if tg_val > max_tg:
                    max_tg = tg_val
                    best_row = row
            except (ValueError, TypeError):
                continue
                
    if not best_row:
        return {}, 0.0
        
    best_config = {}
    for flag in target_flags:
        col = get_csv_column(flag)
        val = best_row.get(col)
        if val is None:
            # Fallback for slight naming variations in llama-bench outputs
            alt_col = col.replace('n_', '').replace('-', '_')
            val = best_row.get(alt_col)
        best_config[flag] = val if val is not None else 'unknown'
        
    return best_config, max_tg

# ==================== Metadata Extraction ====================
def extract_metadata(csv_path: Path) -> dict:
    """Extracts system and model metadata from the header/first row of a benchmark CSV."""
    meta = {}
    if not csv_path.exists():
        return meta
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            row = next(reader, None)
            if row:
                for key in ['build_number', 'cpu_info', 'backends', 'model_filename', 'model_type', 'model_size']:
                    meta[key] = row.get(key, 'N/A').strip()
    return meta

def extract_repetitions(base_args: list) -> int:
    """Extracts the -r value from base arguments. Defaults to 1 if not specified."""
    if "-r" in base_args:
        idx = base_args.index("-r")
        try:
            return int(base_args[idx + 1])
        except (IndexError, ValueError):
            pass
    return 1

def main() -> None:
    parser = argparse.ArgumentParser(description="llama-optimizer.py - Benchmark parameter wrapper for llama-bench")
    parser.add_argument('--params-file', type=Path, help="Path to the optimization parameters file")
    parser.add_argument('--llama-dir', type=Path, default=Path("./bin"), help="Directory containing llama-bench")
    parser.add_argument('--model-dir', type=Path, default=Path("../models"), help="Directory containing GGUF models")
    parser.add_argument('-m', '--model', type=Path, help="Direct path to the model file (overrides params file)")
    parser.add_argument('--grid', action='store_true', help="Enable grid search mode (tests all ::optimize combinations simultaneously)")
    parser.add_argument('--output-dir', type=Path, default=None, help="Directory for result CSV/TXT files (default: current directory, can also be set via ::output-dir)")
    parser.add_argument('--output-format', choices=list(OUTPUT_FORMATS), default=None, help="Benchmark output format: 'both' writes bench_*.csv (for LlamaGraph) + bench_*.md (human-readable), 'csv' or 'md' writes only one (default: both, can also be set via ::output-format)")
    parser.add_argument('--top', type=int, default=None, help="Grid mode: result rows per TG/PP weighting (default: 1, can also be set via ::top)")
    parser.add_argument('--table-weights', default=None, help="Grid mode: comma-separated TG/PP weightings, e.g. '1/0,0.7/0.3,0.5/0.5,0.3/0.7,0/1' (default: TG|70/30|50/50|30/70|PP, can also be set via ::table-weights)")
    args, unknown = parser.parse_known_args()

    # ==================== Load Configuration ====================
    if args.params_file and args.params_file.exists():
        configs, base_args, optimize_targets = load_params_recursive(args.params_file)
    else:
        selected = select_file(Path("."), "Params File for Optimization", "*.txt")
        if not selected: sys.exit(0)
        configs, base_args, optimize_targets = load_params_recursive(selected)

    # ==================== Resolve Model Path ====================
    model_path = args.model
    if not model_path:
        for i, arg in enumerate(base_args):
            if arg in ['-m', '--model'] and i + 1 < len(base_args):
                model_path = Path(base_args[i + 1])
                break
    if not model_path:
        model_path = select_file(args.model_dir, "Select Model for Benchmark", "*.gguf")
    if not model_path:
        sys.exit(0)
        
    print(f"{CLR_GREEN}=== Optimization started for: {model_path.name} ==={CLR_RESET}")

    # ==================== Determine Execution Mode ====================
    is_grid = args.grid or configs.get('grid', False)

    # Resolve output directory: CLI --output-dir wins, then ::output-dir, else CWD
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        cfg_val = configs.get('output-dir', '.')
        output_dir = Path(str(cfg_val)) if cfg_val is not True else Path('.')
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"{CLR_RED}❌ Cannot create output directory '{output_dir}': {e}{CLR_RESET}")
        sys.exit(1)
    print(f"{CLR_CYAN}Output directory:{CLR_RESET} {output_dir.resolve()}")

    # Resolve benchmark output format: CLI --output-format wins, then ::output-format
    out_fmt = resolve_output_format(args.output_format, configs)
    print(f"{CLR_CYAN}Output format:{CLR_RESET} {out_fmt} "
          f"({'bench_*.csv + bench_*.md' if out_fmt == 'both' else 'bench_*.' + out_fmt})")

    # Echo parsed targets so value-list mistakes are visible before the run
    if optimize_targets:
        targets_str = ", ".join(
            f"{flag}=[{', '.join(values)}]"
            for flag, values in optimize_targets.items()
        )
        print(f"{CLR_CYAN}Optimization targets:{CLR_RESET} {targets_str}")
    
    # Resolve optimization order
    order_str = configs.get('optimize-order', '').strip()
    optimize_order = [x.strip() for x in order_str.split(',') if x.strip()] if order_str else list(optimize_targets.keys())
    
    if not optimize_order:
        print(f"{CLR_RED}❌ Error: No ::optimize directives or ::optimize-order defined.{CLR_RESET}")
        sys.exit(1)
        
    # Binary & Path resolution
    bench_binary = "llama-bench.exe" if sys.platform == "win32" else "llama-bench"
    llama_bench_path = args.llama_dir / bench_binary
    if not llama_bench_path.exists():
        print(f"{CLR_RED}❌ llama-bench not found at: {llama_bench_path}{CLR_RESET}")
        sys.exit(1)

    # ==================== Grid Search Mode ====================
    if is_grid:
        print(f"{CLR_CYAN}Grid Search Mode activated. Testing all parameter combinations.{CLR_RESET}")
        
        total_combinations = prod(len(v) for v in optimize_targets.values())
        repetitions = extract_repetitions(base_args)
        total_tests = total_combinations * repetitions
        
        print(f"\n{CLR_RED}{BORDER_CHAR * BOX_WIDTH}")
        print(f"⚠️  WARNING: GRID SEARCH MODE ACTIVE ⚠️")
        print(f"{BORDER_CHAR * BOX_WIDTH}{CLR_RESET}")
        print(f"{CLR_YELLOW}You are about to test EVERY combination of your ::optimize targets.")
        print(f"Total unique combinations: {total_combinations}")
        print(f"Total benchmark tests (including -r {repetitions}): {total_tests}{CLR_RESET}\n")
        
        if total_tests > 50:
            confirm = input(f"{CLR_RED}🚨 Over 50 tests detected. Proceed? (y/N): {CLR_RESET}").lower().strip()
            if confirm != 'y':
                print("Operation aborted."); sys.exit(0)
                
        print(f"{CLR_CYAN}llama-bench is starting with these settings in:{CLR_RESET}")
        for i in range(3, 0, -1):
            sys.stdout.write(f"\r  {i} second{'s' if i > 1 else ''} remaining... ")
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\r  Starting now!               \n")
        sys.stdout.flush()

        # Build single command with all grid targets
        grid_args = []
        for flag, values in optimize_targets.items():
            grid_args.extend([flag, ",".join(values)])
            
        # Clean base args: remove -m and any existing target flags to prevent duplicates
        clean_args = []
        skip = False
        for arg in base_args:
            if skip:
                skip = False
                continue
            if arg in ['-m', '--model']:
                skip = True
                continue
            if arg in optimize_targets:
                skip = True
                continue
            clean_args.append(arg)
            
        clean_args = strip_output_format_args(clean_args)
        cmd = [str(llama_bench_path), "-m", str(model_path)] + clean_args + grid_args + output_args_for(out_fmt)
        run_timestamp = int(time.time())
        csv_file = output_dir / f"bench_{model_path.stem}_grid_{run_timestamp}.csv" if out_fmt in ('both', 'csv') else None
        md_file = output_dir / f"bench_{model_path.stem}_grid_{run_timestamp}.md" if out_fmt in ('both', 'md') else None
        result_md_path = output_dir / f"opt_results_{model_path.stem}_{run_timestamp}.md"

        print_cmd_box(" ".join(cmd))

        try:
            run_bench(cmd, csv_file, md_file)
        except subprocess.CalledProcessError as e:
            print(f"\n{CLR_RED}❌ llama-bench failed with exit code {e.returncode}{CLR_RESET}")
            if e.stderr and e.stderr.strip(): print(f"{CLR_RED}{e.stderr.strip()}{CLR_RESET}")
            sys.exit(1)
            
        # Parse & Log Grid Results (auto-evaluation needs the CSV variant).
        # Weighted PP/TG ranking over the whole grid dataset: entries are
        # grouped by parameter combination (best PP + best TG each),
        # min-max normalized over the dataset, then scored per TG/PP
        # weighting (see model/ranking.py). Incomplete combinations
        # (missing PP or TG rows) are skipped.
        if args.top is not None and args.top < 1:
            print(f"{CLR_RED}❌ --top must be >= 1.{CLR_RESET}")
            sys.exit(1)
        top_n = resolve_top(args.top, configs)
        try:
            weights = resolve_table_weights(
                args.table_weights, configs, strict=True)
        except ValueError as exc:
            print(f"{CLR_RED}❌ Invalid --table-weights: {exc}{CLR_RESET}")
            sys.exit(1)
        print(f"{CLR_CYAN}Ranking:{CLR_RESET} top={top_n}, "
              f"weights=[{', '.join(ranking_mod.header_for_weight(w) for w in weights)}] "
              f"(TG/PP shares, normalized over the grid dataset)")

        result_set = None
        if csv_file is not None:
            meta = extract_metadata(csv_file)
            param_map = build_grid_param_map(list(optimize_targets.keys()))
            entries = ranking_mod.parse_grid_csv(csv_file, param_map)
            if not entries:
                print(f"{CLR_YELLOW}⚠ No complete (pp+tg) configurations found in CSV.{CLR_RESET}")
            else:
                result_set = ranking_mod.rank(entries, weights, top_n)
        else:
            print(f"{CLR_YELLOW}⚠ --output-format md: skipping auto-evaluation (needs CSV).{CLR_RESET}")
            meta, result_set = {}, None

        with open(result_md_path, 'w', encoding='utf-8') as res_f:
            res_f.write(f"=== Grid Search Run: {model_path.name} ===\n")
            res_f.write(f"Timestamp: {run_timestamp}\n")
            res_f.write(f"Mode: Grid Search (Combinations: {total_combinations}, Repetitions: {repetitions})\n")
            res_f.write(f"Ranking: top={top_n}, "
                        f"weights=[{', '.join(ranking_mod.header_for_weight(w) for w in weights)}] "
                        f"(TG/PP shares, min-max normalized over this dataset)\n")
            if meta:
                res_f.write("\n🔍 System & Model Metadata:\n")
                meta_labels = [
                    ("llama build", "build_number"), ("cpu info", "cpu_info"), ("backends", "backends"),
                    ("model file", "model_filename"), ("model type", "model_type"), ("model size", "model_size")
                ]
                max_label_len = max(len(lbl) for lbl, _ in meta_labels)
                for label, key in meta_labels:
                    res_f.write(f"{label.ljust(max_label_len)} : {meta.get(key, 'N/A')}\n")

            if result_set is None:
                res_f.write("\n⚠ No ranking available (no CSV output or no complete configs).\n")
            else:
                table_md = ranking_mod.render_markdown(result_set)
                res_f.write("\n✅ WEIGHTED RANKING (rows: rank 1..N, "
                            "sub-rows: par = copy-paste params, pp/tg = t/s, score = weighted norm):\n\n")
                res_f.write(table_md)
                res_f.write("\n📝 Recommended lines per weighting (rank 1 each):\n")
                for header, col in zip(result_set["headers"], result_set["cells"]):
                    if not col:
                        continue
                    line = ranking_mod.format_params(col[0]["params"])
                    print(f"  {CLR_GREEN}[{header}] {line}{CLR_RESET} "
                          f"(pp {col[0]['pp']:.2f} / tg {col[0]['tg']:.2f} t/s, "
                          f"score {col[0]['score']:.4f})")
                    res_f.write(f"[{header}] {line} "
                                f"(pp {col[0]['pp']:.2f} / tg {col[0]['tg']:.2f} t/s, "
                                f"score {col[0]['score']:.4f})\n")
                
        print(f"\n{CLR_GREEN}Finished! Benchmark files and results log saved to: {output_dir.resolve()}{CLR_RESET}")
        if csv_file is not None:
            print(f"{CLR_CYAN}CSV file: {csv_file}{CLR_RESET}")
        if md_file is not None:
            print(f"{CLR_CYAN}Markdown file: {md_file}{CLR_RESET}")
        print(f"{CLR_CYAN}Result file: {result_md_path}{CLR_RESET}")
        return

    # ==================== Sequential Optimization Mode (Default) ====================
    print(f"{CLR_CYAN}Optimization sequence:{CLR_RESET} {optimize_order}")
    
    run_timestamp = int(time.time())
    result_md_path = output_dir / f"opt_results_{model_path.stem}_{run_timestamp}.md"
    first_csv_path = None
    metadata_written = False
    best_config = {}
    current_base_args = base_args.copy()
    
    with open(result_md_path, 'w', encoding='utf-8') as res_f:
        res_f.write(f"=== Sequential Optimization Run: {model_path.name} ===\n")
        res_f.write(f"Timestamp: {run_timestamp}\n")
        res_f.write(f"Order: {optimize_order}\n")
        res_f.write("-" * 50 + "\n")
        res_f.flush()
        
        for phase, flag in enumerate(optimize_order, 1):
            print(f"\n{CLR_YELLOW}📊 Phase {phase}: Optimizing '{flag}'{CLR_RESET}")
            if flag not in optimize_targets:
                print(f"{CLR_YELLOW}⏭ Skipped: No values defined for ::optimize {flag}.{CLR_RESET}")
                continue
                
            values = optimize_targets[flag]
            safe_flag = flag.lstrip('-').replace('-', '_')
            csv_file = output_dir / f"bench_{model_path.stem}_{safe_flag}_phase{phase}_{run_timestamp}.csv" if out_fmt in ('both', 'csv') else None
            md_file = output_dir / f"bench_{model_path.stem}_{safe_flag}_phase{phase}_{run_timestamp}.md" if out_fmt in ('both', 'md') else None

            # Filter out -m and current target flag from base args to avoid duplication
            clean_args = []
            skip = False
            for arg in current_base_args:
                if skip:
                    skip = False
                    continue
                if arg in ['-m', '--model'] or arg == flag:
                    skip = True
                    continue
                clean_args.append(arg)

            clean_args = strip_output_format_args(clean_args)
            clean_args.extend([flag, ",".join(values)])
            cmd = [str(llama_bench_path), "-m", str(model_path)] + clean_args + output_args_for(out_fmt)

            print_cmd_box(" ".join(cmd))
            print(f"{CLR_CYAN}Starting run {phase}/{len(optimize_order)} ({len(values)} values)...{CLR_RESET}")

            try:
                run_bench(cmd, csv_file, md_file)
            except subprocess.CalledProcessError as e:
                print(f"\n{CLR_RED}❌ llama-bench failed (Exit {e.returncode}){CLR_RESET}")
                if e.stderr and e.stderr.strip(): print(f"{CLR_RED}{e.stderr.strip()}{CLR_RESET}")
                print(f"{CLR_YELLOW}💡 Check options in your params file. The script passes them directly to llama-bench.{CLR_RESET}")
                sys.exit(1)

            # Extract metadata once from the first CSV
            if first_csv_path is None and csv_file is not None:
                first_csv_path = csv_file
                meta = extract_metadata(first_csv_path)
                if meta and not metadata_written:
                    res_f.write("\n🔍 System & Model Metadata:\n")
                    meta_labels = [
                        ("llama build", "build_number"), ("cpu info", "cpu_info"), ("backends", "backends"),
                        ("model file", "model_filename"), ("model type", "model_type"), ("model size", "model_size")
                    ]
                    max_label_len = max(len(lbl) for lbl, _ in meta_labels)
                    for label, key in meta_labels:
                        res_f.write(f"{label.ljust(max_label_len)} : {meta.get(key, 'N/A')}\n")
                    res_f.write("\n")
                    res_f.flush()
                    metadata_written = True
                    
            res_f.write("📊 Phase Results:\n")
            if csv_file is None:
                msg = (f"Phase {phase}: {flag} → no auto-evaluation "
                       f"(--output-format md has no CSV to parse)")
                print(f"{CLR_YELLOW}⚠ {msg}{CLR_RESET}")
                res_f.write(f"{msg}\n")
                res_f.flush()
                continue
            results = parse_bench_output(csv_file, flag)
            if not results:
                print(f"{CLR_RED}⚠ No valid TG results found for '{flag}'.{CLR_RESET}")
                continue
                
            best_value, best_tg = max(results.items(), key=lambda x: x[1])
            best_config[flag] = best_value
            phase_msg = f"Phase {phase}: {flag}={best_value} → {best_tg:.2f} t/s"
            print(f"{CLR_GREEN}✅ Best value: {phase_msg}{CLR_RESET}")
            res_f.write(f"{phase_msg}\n")
            res_f.flush()
            
            # Append best value to base args for the next phase
            final_args = []
            skip = False
            for arg in current_base_args:
                if skip:
                    skip = False
                    continue
                if arg == flag:
                    skip = True
                    continue
                final_args.append(arg)
            final_args.extend([flag, str(best_value)])
            current_base_args = final_args
            
        # ==================== Final Sequential Output ====================
        res_f.write("\n" + "=" * 50 + "\n")
        res_f.write("✅ BEST CONFIG VALUES:\n")
        for param, value in best_config.items():
            line = f"{param} = {value}"
            print(f"  {line}")
            res_f.write(f"{line}\n")
        res_f.write("\n📝 Recommended lines for your config:\n")
        for param, value in best_config.items():
            line = f"{param} {value}"
            print(line)
            res_f.write(f"{line}\n")
            
    print(f"\n{CLR_GREEN}Finished! Benchmark files and result log saved to: {output_dir.resolve()}{CLR_RESET}")
    print(f"{CLR_CYAN}Result file: {result_md_path}{CLR_RESET}")

if __name__ == "__main__":
    main()