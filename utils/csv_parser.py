"""
utils/csv_parser.py

Low-level CSV parsing for llama-bench output files.
Extracts raw rows, detects varying/constant parameters,
and identifies measurement columns (avg_ts, avg_ns, etc.).

No Tkinter, no Matplotlib dependencies.
"""

import csv
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Columns that are metadata / build info and never count as "parameters"
_IGNORE_COLS = frozenset({
    'build_commit', 'build_number', 'cpu_info', 'gpu_info', 'backends',
    'model_filename', 'model_type', 'model_size', 'model_n_params',
    'test_time', 'avg_ns', 'stddev_ns', 'avg_ts', 'stddev_ts',
    'n_prompt', 'n_gen', 'n_depth', 'fit_target', 'fit_min_ctx',
})


def parse_bench_csv(csv_path: Path) -> Optional[dict]:
    """
    Parse a single llama-bench CSV file.

    Returns a dict with:
      - 'source': str path
      - 'varying_params': list[str]  – params that change across rows
      - 'constant_params': dict[str, str]  – params fixed for the whole file
      - 'raw_rows': list[dict]  – one entry per measurement row, type='pp'|'tg'

    Returns None if the file is empty or cannot be parsed.
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except Exception as exc:
        print(f"[csv_parser] Cannot read {csv_path.name}: {exc}")
        return None

    if not rows:
        return None

    # ---- Detect varying vs. constant parameters -------------------------
    all_cols = list(rows[0].keys())
    param_cols = [c for c in all_cols if c not in _IGNORE_COLS]

    varying_params: list[str] = []
    constant_params: dict[str, str] = {}

    for col in param_cols:
        vals = [r[col].strip() for r in rows if r[col].strip()]
        unique = set(vals)
        if len(unique) > 1:
            varying_params.append(col)
        elif len(unique) == 1:
            constant_params[col] = unique.pop()

    # ---- Parse measurement rows -----------------------------------------
    raw_rows: list[dict] = []

    for row in rows:
        try:
            n_prompt = int(row.get('n_prompt', -1))
            n_gen = int(row.get('n_gen', -1))
        except (ValueError, TypeError):
            print(f"[csv_parser] Skipping row: invalid n_prompt/n_gen "
                  f"(n_prompt={row.get('n_prompt')!r}, "
                  f"n_gen={row.get('n_gen')!r})")
            continue

        is_pp = (n_prompt > 0 and n_gen == 0)
        is_tg = (n_prompt == 0 and n_gen > 0)
        if not (is_pp or is_tg):
            print(f"[csv_parser] Skipping row: neither PP nor TG "
                  f"(n_prompt={n_prompt}, n_gen={n_gen})")
            continue

        entry: dict = {'type': 'pp' if is_pp else 'tg'}

        # Copy all non-ignored columns with numeric coercion where possible
        for col in all_cols:
            if col in _IGNORE_COLS:
                continue
            raw = row.get(col, '').strip()
            if raw == '':
                entry[col] = None
            else:
                try:
                    entry[col] = float(raw)
                except ValueError:
                    entry[col] = raw  # keep as string (e.g. 'layer', 'auto')

        # Measurement values
        try:
            entry['ts_val'] = float(row['avg_ts']) if row.get('avg_ts', '').strip() else None
            entry['ts_err'] = float(row.get('stddev_ts', 0) or 0)
        except (ValueError, KeyError):
            print(f"[csv_parser] Invalid avg_ts/stddev_ts in row "
                  f"(avg_ts={row.get('avg_ts')!r}, "
                  f"stddev_ts={row.get('stddev_ts')!r})")
            entry['ts_val'] = None
            entry['ts_err'] = 0.0

        try:
            entry['ns_val'] = float(row['avg_ns']) if row.get('avg_ns', '').strip() else None
            entry['ns_err'] = float(row.get('stddev_ns', 0) or 0)
        except (ValueError, KeyError):
            print(f"[csv_parser] Invalid avg_ns/stddev_ns in row "
                  f"(avg_ns={row.get('avg_ns')!r}, "
                  f"stddev_ns={row.get('stddev_ns')!r})")
            entry['ns_val'] = None
            entry['ns_err'] = 0.0

        raw_rows.append(entry)

    if not raw_rows:
        return None

    return {
        'source': str(csv_path),
        'varying_params': varying_params,
        'constant_params': constant_params,
        'raw_rows': raw_rows,
    }


def is_llama_bench_csv(csv_path: Path) -> bool:
    """
    Quick check: does this file look like a llama-bench output?
    Reads only the header line.
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as fh:
            header = fh.readline()
        return ('avg_ts' in header or 'avg_ns' in header) \
            and 'n_prompt' in header \
            and 'n_gen' in header
    except Exception:
        return False