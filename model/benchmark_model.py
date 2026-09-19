"""
model/benchmark_model.py

The Model layer of llamagraph's MVP architecture.

Responsibilities:
  - Load and parse llama-bench CSV files via utils.csv_parser
  - Merge data from multiple files into a unified in-memory store
  - Detect which parameters vary (potential axes) vs. are fixed
  - Provide filtered/sliced views of the raw measurement data
  - Expose clean query methods that the Presenter calls

No Tkinter, no Matplotlib imports here — pure data logic.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from utils.csv_parser import parse_bench_file
from utils.colors import normalize_series

# Type alias used throughout
RawRow = dict[str, Any]


class BenchmarkModel:
    """
    Central data store for all loaded llama-bench CSV measurements.

    The presenter calls:
      load_files(paths)     – replace the current dataset
      get_dimensions()      – list of parameter names that vary
      get_dim_values(dim)   – sorted list of unique values for one dimension
      get_all_params()      – full dimension→values mapping
      apply_filters(fdict)  – set which dimension values are kept for plotting
      get_filter_state()    – current filter dict
      get_plot_data(...)    – aggregated series ready for the plot engines
    """

    def __init__(self) -> None:
        # Loaded dataset entries – each entry wraps one CSV file
        self._datasets: list[dict] = []

        # Computed once after load: {param_name: sorted_list_of_values}
        self._dim_values: dict[str, list] = {}

        # Active dimension filters: {param_name: set_of_allowed_values}
        # An empty set means "show all values for this dim"
        self._active_filters: dict[str, set] = {}

        # Categorical mappings for string-valued dimensions used in 3D plots
        # {dim_name: {string_value: integer_code}}
        self._cat_mappings: dict[str, dict[str, int]] = {}

        # Registered observers (Presenter callbacks)
        self._observers: list[Callable[[], None]] = []

    # ── Observer / notification ───────────────────────────────────────────────

    def add_observer(self, callback: Callable[[], None]) -> None:
        """Register a zero-arg callback to call whenever the model changes."""
        self._observers.append(callback)

    def _notify(self) -> None:
        for cb in self._observers:
            try:
                cb()
            except Exception as exc:
                print(f"[Model] Observer error: {exc}")

    # ── File loading ──────────────────────────────────────────────────────────

    def load_files(self, selected_paths: list[Path]) -> list[str]:
        """
        Parse and store the given CSV files.

        Returns a list of error messages for files that could not be parsed
        (empty list on full success).
        """
        self._datasets = []
        errors: list[str] = []

        for path in selected_paths:
            parsed = parse_bench_file(Path(path))
            if parsed is None:
                errors.append(f"Could not parse: {Path(path).name}")
                continue
            self._datasets.append({
                'path': Path(path),
                'data': parsed,
            })

        self._recompute_dimensions()
        self._notify()
        return errors

    def clear(self) -> None:
        """Remove all loaded data."""
        self._datasets = []
        self._dim_values = {}
        self._active_filters = {}
        self._notify()

    # ── Dimension / parameter discovery ──────────────────────────────────────

    def _recompute_dimensions(self) -> None:
        """
        After loading, collect all parameter dimensions and their unique
        values across all loaded datasets.  Reset filters.
        """
        dim_vals: dict[str, set] = defaultdict(set)

        for entry in self._datasets:
            data = entry['data']
            # Collect values that actually appear in raw rows
            for row in data['raw_rows']:
                for param in data['varying_params']:
                    val = row.get(param)
                    if val is not None:
                        dim_vals[param].add(val)
            # Also include constant params (single value) as dimensions
            for param, val in data['constant_params'].items():
                try:
                    dim_vals[param].add(float(val))
                except (ValueError, TypeError):
                    dim_vals[param].add(val)

        # Sort numerically where possible, else lexicographically
        self._dim_values = {}
        for dim, vals in dim_vals.items():
            try:
                self._dim_values[dim] = sorted(vals, key=float)
            except (TypeError, ValueError):
                self._dim_values[dim] = sorted(vals, key=str)

        # Reset filters: all values selected by default
        self._active_filters = {
            dim: set(vals)
            for dim, vals in self._dim_values.items()
        }

    def get_dimensions(self) -> list[str]:
        """
        Return parameter names that vary across the loaded data (candidate axes).
        Only dims with >1 unique value are returned.
        """
        return [d for d, vals in self._dim_values.items() if len(vals) > 1]

    def get_all_dim_values(self) -> dict[str, list]:
        """Return {dim: sorted_value_list} for every known dimension."""
        return dict(self._dim_values)

    def get_dim_values(self, dim: str) -> list:
        """Return the sorted list of unique values for one dimension."""
        return list(self._dim_values.get(dim, []))

    # ── Filter management ─────────────────────────────────────────────────────

    def apply_filters(self, filter_dict: dict[str, set]) -> None:
        """
        Set active filters.  *filter_dict* maps dim_name → set_of_allowed_values.
        An absent dim keeps its previous filter state.
        """
        for dim, allowed in filter_dict.items():
            if dim in self._dim_values:
                self._active_filters[dim] = set(allowed)
        self._notify()

    def reset_filters(self) -> None:
        """Re-allow all values for all dimensions."""
        self._active_filters = {
            dim: set(vals)
            for dim, vals in self._dim_values.items()
        }
        self._notify()

    def get_filter_state(self) -> dict[str, set]:
        """Return a snapshot of the current filter state."""
        return {dim: set(vals) for dim, vals in self._active_filters.items()}

    # ── Data access ───────────────────────────────────────────────────────────

    def has_data(self) -> bool:
        return len(self._datasets) > 0

    def get_dataset_count(self) -> int:
        return len(self._datasets)

    def get_dataset_paths(self) -> list[Path]:
        return [entry['path'] for entry in self._datasets]

    def get_datasets_raw(self) -> list[dict]:
        """Return the raw dataset list (read-only intent)."""
        return self._datasets

    def get_filtered_rows(
        self,
        row_type: Optional[str] = None,   # 'pp', 'tg', or None for both
        extra_filters: Optional[dict[str, Any]] = None,
    ) -> list[RawRow]:
        """
        Return all raw measurement rows that pass the current dimension filters.

        Parameters
        ----------
        row_type:
            If given, restrict to 'pp' or 'tg' rows only.
        extra_filters:
            Additional equality constraints beyond the stored filters,
            e.g. {'n_gpu_layers': 18.0}.
        """
        result: list[RawRow] = []

        for entry in self._datasets:
            data = entry['data']
            consts = data['constant_params']

            for row in data['raw_rows']:
                if row_type and row.get('type') != row_type:
                    continue

                if not self._row_passes_filters(row, consts):
                    continue

                if extra_filters:
                    if not self._row_passes_extra(row, consts, extra_filters):
                        continue

                result.append(row)

        return result

    def _row_passes_filters(
        self, row: RawRow, consts: dict[str, str]
    ) -> bool:
        """
        Return True if the row's parameter values are all within the active
        filter sets.
        """
        for dim, allowed in self._active_filters.items():
            if not allowed:
                continue  # empty set → no restriction

            val = row.get(dim)
            if val is None:
                # Fall back to constant value for this file
                raw = consts.get(dim, '')
                try:
                    val = float(raw)
                except (ValueError, TypeError):
                    val = raw

            if val not in allowed:
                return False
        return True

    def _row_passes_extra(
        self, row: RawRow, consts: dict[str, str], extra: dict[str, Any]
    ) -> bool:
        for dim, required in extra.items():
            val = row.get(dim)
            if val is None:
                raw = consts.get(dim, '')
                try:
                    val = float(raw)
                except (ValueError, TypeError):
                    val = raw
            if val != required:
                return False
        return True

    # ── Aggregated plot data ──────────────────────────────────────────────────

    def get_2d_series(
        self,
        x_dim: str,
        show_ts: bool,
        show_pp: bool,
        show_tg: bool,
        normalize: bool,
        scale_pct: bool,
    ) -> dict:
        """
        Build aggregated 2-D series data for *x_dim*.

        Returns:
        {
          'pp': [{'x': ..., 'y': ..., 'err': ..., 'file_idx': ...}, ...],
          'tg': [...],
        }
        One list entry per (file_index, x_value) pair.
        """
        y_key = 'ts_val' if show_ts else 'ns_val'
        err_key = 'ts_err' if show_ts else 'ns_err'

        series_pp: list[dict] = []
        series_tg: list[dict] = []

        for file_idx, entry in enumerate(self._datasets):
            data = entry['data']
            consts = data['constant_params']

            # Group rows by (x_value, type)
            groups: dict[tuple, list] = defaultdict(list)

            for row in data['raw_rows']:
                # Check global filter
                if not self._row_passes_filters(row, consts):
                    continue

                x_val = self._resolve_param(row, consts, x_dim)
                if x_val is None:
                    continue

                y_val = row.get(y_key)
                e_val = row.get(err_key, 0.0) or 0.0
                if y_val is None:
                    continue

                groups[(x_val, row['type'])].append({
                    'y': float(y_val),
                    'err': float(e_val),
                    'ts': _safe_float(row.get('ts_val')),
                    'ts_err': _safe_float(row.get('ts_err')) or 0.0,
                    'ns': _safe_float(row.get('ns_val')),
                    'ns_err': _safe_float(row.get('ns_err')) or 0.0,
                })

            # Aggregate per-file pp/tg
            pp_agg = _aggregate_groups(groups, 'pp')
            tg_agg = _aggregate_groups(groups, 'tg')

            if normalize:
                pp_agg = _normalize_agg(pp_agg, scale_pct)
                tg_agg = _normalize_agg(tg_agg, scale_pct)

            if show_pp:
                for pt in pp_agg:
                    pt['file_idx'] = file_idx
                    series_pp.append(pt)
            if show_tg:
                for pt in tg_agg:
                    pt['file_idx'] = file_idx
                    series_tg.append(pt)

        return {'pp': series_pp, 'tg': series_tg}

    def get_3d_points(
        self,
        x_dim: str,
        y_dim: str,
        show_ts: bool,
        show_pp: bool,
        show_tg: bool,
        normalize: bool,
        scale_pct: bool,
    ) -> tuple[list[tuple], list[tuple], dict, dict]:
        """
        Build raw (x, y, z, err) point lists for the 3-D plot engine.

        Returns (points_pp, points_tg, stats, infos) where each point
        is a 4-tuple (x_val, y_val, z_val, z_err), stats holds the
        pre-normalization Z minima/maxima per series:
        {'pp_min', 'pp_max', 'tg_min', 'tg_max'} (None when a series
        is empty), and infos holds per-point tooltip payloads aligned
        1:1 with the point lists:
        {'pp': [...], 'tg': [...]} with
        {'x', 'y', 'ts', 'ts_err', 'ns', 'ns_err'} each.
        With *normalize* set, each series is independently
        min-max stretched to the full range, so both series fill the
        whole plot height — the same profile each shows on its own.

        String-valued dimensions are encoded to sequential integers;
        call get_dim_labels() to retrieve the label mapping.
        """
        z_key = 'ts_val' if show_ts else 'ns_val'
        e_key  = 'ts_err' if show_ts else 'ns_err'

        self._cat_mappings.clear()
        points_pp: list[tuple] = []
        points_tg: list[tuple] = []
        infos_pp: list[dict] = []
        infos_tg: list[dict] = []

        for entry in self._datasets:
            data = entry['data']
            consts = data['constant_params']

            for row in data['raw_rows']:
                if not self._row_passes_filters(row, consts):
                    continue

                x_val = self._resolve_param(row, consts, x_dim)
                y_val = self._resolve_param(row, consts, y_dim)
                z_val = row.get(z_key)
                e_val = float(row.get(e_key, 0) or 0)

                if any(v is None for v in (x_val, y_val, z_val)):
                    continue

                try:
                    fx = float(x_val)
                except (TypeError, ValueError):
                    fx = self._cat_encode(x_val, x_dim)
                try:
                    fy = float(y_val)
                except (TypeError, ValueError):
                    fy = self._cat_encode(y_val, y_dim)

                pt = (fx, fy, float(z_val), e_val)
                info = {
                    'x': x_val,
                    'y': y_val,
                    'ts': _safe_float(row.get('ts_val')),
                    'ts_err': _safe_float(row.get('ts_err')) or 0.0,
                    'ns': _safe_float(row.get('ns_val')),
                    'ns_err': _safe_float(row.get('ns_err')) or 0.0,
                }

                if row['type'] == 'pp' and show_pp:
                    points_pp.append(pt)
                    infos_pp.append(info)
                elif row['type'] == 'tg' and show_tg:
                    points_tg.append(pt)
                    infos_tg.append(info)

        # Pre-normalization minima/maxima per series, so the view can
        # label the Z axis in absolute units even though both series
        # share the stretched [0, 1] (or [0, 100]) volume when normalized.
        stats = {
            'pp_min': min((p[2] for p in points_pp), default=None),
            'pp_max': max((p[2] for p in points_pp), default=None),
            'tg_min': min((p[2] for p in points_tg), default=None),
            'tg_max': max((p[2] for p in points_tg), default=None),
        }

        # Independent per-series min-max stretch (each series fills the
        # full height); on the shared Z axis both therefore span [0, 1].
        # Infos stay aligned 1:1 (same order, untouched values).
        if normalize:
            points_pp = _normalize_3d_points(points_pp, scale_pct)
            points_tg = _normalize_3d_points(points_tg, scale_pct)

        return points_pp, points_tg, stats, {'pp': infos_pp, 'tg': infos_tg}

    def _cat_encode(self, val, dim: str) -> float:
        """Map a string value to a sequential float code for the given dimension."""
        if dim not in self._cat_mappings:
            self._cat_mappings[dim] = {}
        mapping = self._cat_mappings[dim]
        if val not in mapping:
            mapping[val] = len(mapping)
        return float(mapping[val])

    def get_dim_labels(self, dim: str) -> list[str] | None:
        """Return categorical labels for *dim* ordered by code, or None if not categorical."""
        mapping = self._cat_mappings.get(dim)
        if not mapping:
            return None
        sorted_pairs = sorted(mapping.items(), key=lambda kv: kv[1])
        return [p[0] for p in sorted_pairs]

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_param(
        row: RawRow, consts: dict[str, str], dim: str
    ) -> Optional[float | str]:
        """
        Look up a parameter value from the row first, then constant_params.
        Returns a float if convertible, else the raw string, else None.
        """
        val = row.get(dim)
        if val is None:
            raw = consts.get(dim, '')
            try:
                val = float(raw)
            except (ValueError, TypeError):
                val = raw if raw else None
        return val


# ── Module-level helpers ──────────────────────────────────────────────────────

def _safe_float(value) -> Optional[float]:
    """Float conversion returning None instead of raising."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _aggregate_groups(
    groups: dict[tuple, list], row_type: str
) -> list[dict]:
    """
    Convert (x_val, type) → [measurement dicts] groups into averaged
    data points sorted by x_val.

    Each point carries the active metric as 'y'/'err' plus the raw
    'ts'/'ts_err'/'ns'/'ns_err' values for tooltips (either metric may
    be None when the source row lacked it).
    """
    result = []
    relevant = {k: v for k, v in groups.items() if k[1] == row_type}
    for (x_val, _), measurements in sorted(relevant.items()):
        y_vals = [m['y'] for m in measurements]
        e_vals = [m['err'] for m in measurements]
        mean_y = sum(y_vals) / len(y_vals)
        max_e = max(e_vals)
        point: dict = {'x': x_val, 'y': mean_y, 'err': max_e}
        for key in ('ts', 'ts_err', 'ns', 'ns_err'):
            vals = [m[key] for m in measurements if m[key] is not None]
            if key.endswith('_err'):
                point[key] = max(vals) if vals else 0.0
            else:
                point[key] = sum(vals) / len(vals) if vals else None
        result.append(point)
    return result


def _normalize_agg(points: list[dict], scale_pct: bool) -> list[dict]:
    """Normalize a list of {'x', 'y', 'err', ...} points in-place (returns new list)."""
    y_vals = [p['y'] for p in points]
    e_vals = [p['err'] for p in points]
    ny, ne, _ = normalize_series(y_vals, e_vals, scale_pct)
    out = []
    for i, p in enumerate(points):
        q = dict(p)
        q['y'], q['err'] = ny[i], ne[i]
        out.append(q)
    return out


def _normalize_3d_points(
    points: list[tuple], scale_pct: bool
) -> list[tuple]:
    """
    Stretch the z component of (x, y, z, err) tuples to the full range.

    Each series is min-max scaled to [0, 1] (or [0, 100] when *scale_pct*
    is True), so every visible series fills the whole plot height — the
    same profile Matplotlib's autoscaling shows for a lone series.
    Errors are scaled by the same span. A constant series maps to the
    top (1.0); an empty list stays empty.
    """
    if not points:
        return points
    z_vals = [p[2] for p in points]
    z_min, z_max = min(z_vals), max(z_vals)
    span = z_max - z_min
    factor = 100.0 if scale_pct else 1.0
    if span == 0:
        # Constant series: pin to the top, keep errors relative to the
        # level (same convention as max-normalization) instead of dropping
        # them.
        base = z_max if z_max != 0 else 1.0
        return [(p[0], p[1], factor, p[3] / base * factor) for p in points]
    return [
        (p[0], p[1], (p[2] - z_min) / span * factor, p[3] / span * factor)
        for p in points
    ]