"""
utils/csv_parser.py

Low-level CSV parsing for llama-bench output files.
Extracts raw rows, detects varying/constant parameters,
and identifies measurement columns (avg_ts, avg_ns, etc.).

No Tkinter, no Matplotlib dependencies.
"""

import csv
import re
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
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as fh:
            header = fh.readline()
        return ('avg_ts' in header or 'avg_ns' in header) \
            and 'n_prompt' in header \
            and 'n_gen' in header
    except Exception:
        return False


# ── Markdown (llama-bench `-o md`) support ────────────────────────────────

# Short md column names mapped to their CSV counterparts so that mixed
# CSV+MD datasets share filter/axis dimensions for the key tuning params.
_MD_COL_ALIASES = {
    'ngl': 'n_gpu_layers',
    'fa': 'flash_attn',
    'backend': 'backends',
}

_TEST_RE = re.compile(r'^\s*(pp|tg)\s*(\d+)\s*$', re.IGNORECASE)
_BUILD_RE = re.compile(r'build\s*:\s*(\S+?)\s*\(\s*(\d+)\s*\)', re.IGNORECASE)
_SEP_CELL_RE = re.compile(r'^:?-{1,}:?$')


def _split_md_row(line: str) -> list[str]:
    """Split a `| a | b |` markdown table line into stripped cells."""
    text = line.strip()
    if text.startswith('|'):
        text = text[1:]
    if text.endswith('|'):
        text = text[:-1]
    return [cell.strip() for cell in text.split('|')]


def _is_md_separator(cells: list[str]) -> bool:
    return bool(cells) and all(_SEP_CELL_RE.match(c) for c in cells)


def _parse_ts_cell(raw: str) -> Optional[tuple[float, float]]:
    """Parse a `t/s` cell like `101.48 ± 2.22` (or a plain number)."""
    raw = raw.strip()
    if not raw:
        return None
    for sep in ('±', '+/-'):
        if sep in raw:
            val_s, _, err_s = raw.partition(sep)
            try:
                return float(val_s.strip()), float(err_s.strip().split()[0])
            except (ValueError, IndexError):
                return None
    try:
        return float(raw.split()[0]), 0.0
    except ValueError:
        return None


def _parse_test_cell(raw: str) -> Optional[tuple[str, int, int]]:
    """Map `pp2048` → ('pp', 2048, 0) and `tg512` → ('tg', 0, 512)."""
    m = _TEST_RE.match(raw or '')
    if not m:
        return None
    kind = m.group(1).lower()
    n = int(m.group(2))
    return ('pp', n, 0) if kind == 'pp' else ('tg', 0, n)


def _coerce_md_value(raw: str):
    if raw == '':
        return None
    try:
        return float(raw)
    except ValueError:
        return raw


def is_llama_bench_md(md_path: Path) -> bool:
    """
    Quick check: does this file look like a llama-bench markdown table?
    Scans the first 100 lines for a `| ... | test | ... | t/s |` header.
    Works regardless of file extension (also catches misnamed `.csv` files).
    """
    try:
        with open(md_path, 'r', encoding='utf-8', errors='replace') as fh:
            for i, line in enumerate(fh):
                if i >= 100:
                    break
                if not line.strip().startswith('|'):
                    continue
                cells = [c.lower() for c in _split_md_row(line)]
                if 'test' in cells and 't/s' in cells:
                    return True
        return False
    except Exception:
        return False


def parse_bench_md(md_path: Path) -> Optional[dict]:
    """
    Parse a llama-bench markdown (`-o md`) table into the same dict shape as
    :func:`parse_bench_csv` (``source``/``varying_params``/``constant_params``/
    ``raw_rows`` with ``type``/``ts_val``/``ts_err``/``ns_val``/``ns_err``).

    Notes on the lossy MD format:
      - `test` encodes the row type: ``pp<N>`` / ``tg<N>``.
      - `t/s` holds ``value ± err``; ``avg_ns`` is derived via
        ``ns = 1e9 * N / ts`` (same relation as the CSV output).
      - Short columns are aliased (``ngl``→``n_gpu_layers``,
        ``fa``→``flash_attn``, ``backend``→``backends``); all other headers
        are lowercased so mixed CSV+MD loads share dimensions.
      - ``n_prompt``/``n_gen`` (decoded from ``test``) are kept as real
        values and dimensions — unlike the CSV path, where both are ignored
        columns. MD loads therefore offer ``n_prompt`` as a plot axis.
      - A trailing ``build: <commit> (<number>)`` line becomes constants.

    Returns None if no valid table/rows are found.
    """
    try:
        with open(md_path, 'r', encoding='utf-8', errors='replace') as fh:
            lines = fh.read().splitlines()
    except Exception as exc:
        print(f"[csv_parser] Cannot read {md_path.name}: {exc}")
        return None

    headers: Optional[list[str]] = None
    build_commit: Optional[str] = None
    build_number: Optional[str] = None
    data_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        m = _BUILD_RE.search(stripped)
        if m:
            build_commit, build_number = m.group(1), m.group(2)
            continue
        if not stripped.startswith('|'):
            continue
        cells = _split_md_row(line)
        if headers is None:
            lowered = [c.lower() for c in cells]
            if 'test' in lowered and 't/s' in lowered:
                headers = cells
            continue
        if _is_md_separator(cells):
            continue
        data_lines.append(line)

    if headers is None or not data_lines:
        return None

    lowered_headers = [h.lower() for h in headers]
    try:
        test_idx = lowered_headers.index('test')
        ts_idx = lowered_headers.index('t/s')
    except ValueError:
        return None

    # Lowercase everything so MD dims unify with CSV dims in mixed loads
    # (alias lookup is case-insensitive, fallback is the lowered header).
    param_headers = [
        (i, _MD_COL_ALIASES.get(h.lower(), h.lower()))
        for i, h in enumerate(headers)
        if i not in (test_idx, ts_idx)
    ]

    raw_rows: list[dict] = []
    col_values: dict[str, set] = defaultdict(set)

    for line in data_lines:
        cells = _split_md_row(line)
        if len(cells) != len(headers):
            print(f"[csv_parser] Skipping malformed md row in {md_path.name}: {line.strip()!r}")
            continue

        parsed_test = _parse_test_cell(cells[test_idx])
        if parsed_test is None:
            print(f"[csv_parser] Skipping md row: unparseable test cell "
                  f"(test={cells[test_idx]!r})")
            continue
        row_type, n_prompt, n_gen = parsed_test

        parsed_ts = _parse_ts_cell(cells[ts_idx])
        if parsed_ts is None:
            print(f"[csv_parser] Skipping md row: unparseable t/s cell "
                  f"(t/s={cells[ts_idx]!r})")
            continue
        ts_val, ts_err = parsed_ts

        n = n_prompt if row_type == 'pp' else n_gen
        if ts_val > 0:
            ns_val: Optional[float] = 1e9 * n / ts_val
            ns_err: float = ns_val * (ts_err / ts_val)
        else:
            ns_val, ns_err = None, 0.0

        entry: dict = {
            'type': row_type,
            'ts_val': ts_val,
            'ts_err': ts_err,
            'ns_val': ns_val,
            'ns_err': ns_err,
            # The `test` cell encodes both type and size (pp<N>/tg<N>).
            # Keep the sizes as real values so same-type different-size rows
            # (e.g. pp512 + pp1024) stay distinguishable and selectable.
            'n_prompt': float(n_prompt),
            'n_gen': float(n_gen),
        }
        col_values['n_prompt'].add(float(n_prompt))
        col_values['n_gen'].add(float(n_gen))
        for i, name in param_headers:
            val = _coerce_md_value(cells[i])
            entry[name] = val
            if val is not None:
                col_values[name].add(val)
        raw_rows.append(entry)

    if not raw_rows:
        return None

    # Header order (like the CSV path), n_prompt/n_gen first (not table cols)
    ordered: list[str] = []
    for c in ['n_prompt', 'n_gen'] + [name for _, name in param_headers]:
        if c not in ordered:
            ordered.append(c)
    varying_params = [c for c in ordered if len(col_values.get(c, ())) > 1]
    constant_params = {c: next(iter(v)) for c, v in col_values.items() if len(v) == 1}
    if build_commit:
        constant_params.setdefault('build_commit', build_commit)
    if build_number:
        constant_params.setdefault('build_number', build_number)

    return {
        'source': str(md_path),
        'varying_params': varying_params,
        'constant_params': {k: str(v) for k, v in constant_params.items()},
        'raw_rows': raw_rows,
    }


def parse_bench_file(path: Path) -> Optional[dict]:
    """
    Dispatch to the CSV or Markdown parser based on content probes.
    Accepts misnamed files (md table with `.csv` extension and vice versa).
    """
    path = Path(path)
    if is_llama_bench_csv(path):
        return parse_bench_csv(path)
    if is_llama_bench_md(path):
        return parse_bench_md(path)
    # Fall back to trying both parsers directly (covers probe edge cases)
    parsed = parse_bench_csv(path)
    if parsed is not None:
        return parsed
    return parse_bench_md(path)