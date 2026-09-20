"""
model/ranking.py

Weighted PP/TG ranking over a defined, filtered dataset.

Responsibilities:
  - Group raw PP/TG measurements that share the same parameter
    combination into comparable entries (best PP + best TG each).
  - Min-max normalize PP and TG over the given dataset so the two
    metrics (which live on very different scales) become comparable.
  - Score every entry per TG/PP weighting and return the Top-N ranks
    per weighting as a structured result set.

No Tkinter, no Matplotlib, no llama-bench, no subprocess imports here —
pure data logic. The result set is a plain-data 3-D structure with
dimensions (weighting, result type, rank) that each consumer renders
on its own (llama-optimizer renders Markdown, LlamaGraph will render
its own view; see ranking.README.md).

Weight convention (TG-first, matching the table headers):
  A header like "70/30" means 70% TG + 30% PP. The extremes map to
  "TG" (100/0) and "PP" (0/100). Weights are stored as
  (w_tg, w_pp) tuples summing to 1.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

#: Default weighting columns, stored TG-first: pure TG, then blended,
#: then pure PP. Renders as: TG | 70/30 | 50/50 | 30/70 | PP
DEFAULT_WEIGHTS_SPEC = "1/0,0.7/0.3,0.5/0.5,0.3/0.7,0/1"

#: Result types stored per cell. ``par`` holds the copy-paste-ready
#: parameter dict, ``pp``/``tg`` the raw t/s values and ``score`` the
#: weighted normalized score that produced the rank.
RESULT_TYPES = ("par", "pp", "tg", "score")


# ── Weights ────────────────────────────────────────────────────────────────

def parse_weights(spec: str) -> list[tuple[float, float]]:
    """Parse a weighting list like ``"1/0,0.7/0.3,0.5/0.5,0.3/0.7,0/1"``.

    Each item is ``TG/PP`` — the TG share first, the PP share second —
    either as fractions (``0.7/0.3``) or as percentages (``70/30``).
    Values are normalized so they sum to 1 and returned as
    ``(w_tg, w_pp)`` tuples in input order. Duplicates are kept (they
    render as repeated columns, which is the caller's choice).

    Raises:
        ValueError: on empty input, malformed items, negative shares
            or a zero total.
    """
    if spec is None or not str(spec).strip():
        raise ValueError("empty weighting specification")
    weights: list[tuple[float, float]] = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split("/")
        if len(parts) != 2:
            raise ValueError(f"malformed weighting {item!r} (expected TG/PP)")
        try:
            tg_share = float(parts[0].strip())
            pp_share = float(parts[1].strip())
        except ValueError:
            raise ValueError(f"malformed weighting {item!r} (not numeric)")
        if tg_share < 0 or pp_share < 0:
            raise ValueError(f"malformed weighting {item!r} (negative share)")
        total = tg_share + pp_share
        if total <= 0:
            raise ValueError(f"malformed weighting {item!r} (zero total)")
        weights.append((tg_share / total, pp_share / total))
    if not weights:
        raise ValueError("empty weighting specification")
    return weights


def default_weights() -> list[tuple[float, float]]:
    """Return the default weighting columns (TG .. PP)."""
    return parse_weights(DEFAULT_WEIGHTS_SPEC)


def header_for_weight(weight: tuple[float, float]) -> str:
    """Map a ``(w_tg, w_pp)`` tuple to its table header.

    Pure TG renders as ``"TG"``, pure PP as ``"PP"``, anything blended
    as ``"<tg_pct>/<pp_pct>"`` (e.g. ``"70/30"``).
    """
    w_tg, w_pp = weight
    if w_tg >= 1.0 - 1e-9 and w_pp <= 1e-9:
        return "TG"
    if w_pp >= 1.0 - 1e-9 and w_tg <= 1e-9:
        return "PP"
    return f"{w_tg * 100:g}/{w_pp * 100:g}"


# ── Entries ────────────────────────────────────────────────────────────────

def make_entry(
    params: dict[str, str],
    pp: Optional[float],
    tg: Optional[float],
) -> dict[str, Any]:
    """Build a single ranking entry from a parameter dict and raw t/s values."""
    return {"params": dict(params), "pp": pp, "tg": tg}


def group_measurements(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group single-kind measurements into comparable entries.

    Each item is ``{"params": dict, "kind": "pp"|"tg", "value": float}``.
    Measurements sharing the exact same parameter dict are merged; the
    best (maximum) value per kind wins (mirrors the optimizer's
    per-value ``max(t/s)`` aggregation). Combinations lacking either a
    PP or a TG measurement are skipped (incomplete configs).

    Returns a list of entries (see :func:`make_entry`), sorted by
    params for determinism.
    """
    best: dict[tuple, dict[str, Any]] = {}
    for item in items:
        params = item.get("params") or {}
        kind = str(item.get("kind", "")).lower()
        if kind not in ("pp", "tg"):
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue
        key = tuple(sorted((str(k), str(v)) for k, v in params.items()))
        slot = best.get(key)
        if slot is None:
            slot = {"params": {str(k): str(v) for k, v in params.items()},
                    "pp": None, "tg": None}
            best[key] = slot
        current = slot[kind]
        if current is None or value > current:
            slot[kind] = value
    entries = [e for e in best.values()
               if e["pp"] is not None and e["tg"] is not None]
    entries.sort(key=lambda e: tuple(sorted(e["params"].items())))
    return entries


def _row_kind(row: dict[str, str]) -> Optional[str]:
    """Classify a raw bench CSV row as ``"pp"``, ``"tg"`` or None.

    Header lookup is case-insensitive (``DictReader`` preserves the
    file's header case, so ``TEST`` must match ``test``).
    """
    lowered = {str(k).lower(): v for k, v in row.items()}
    test = (lowered.get("test") or "").lower()
    if "pp" in test:
        return "pp"
    if "tg" in test:
        return "tg"
    try:
        n_prompt = int(lowered.get("n_prompt", -1))
        n_gen = int(lowered.get("n_gen", -1))
    except (ValueError, TypeError):
        return None
    if n_prompt > 0 and n_gen == 0:
        return "pp"
    if n_prompt == 0 and n_gen > 0:
        return "tg"
    return None


def parse_grid_csv(
    csv_path: Path,
    param_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Build ranking entries from a grid-search bench CSV.

    Args:
        csv_path: bench ``*.csv`` file containing PP and TG rows for
            every tested combination.
        param_map: maps the entry's parameter name (e.g. the optimizer
            CLI flag ``"--ubatch-size"``) to the CSV column holding its
            value (e.g. ``"n_ubatch"``). Entry identity is the plain
            parameter dict — no file/model metadata is included.

    Incomplete combinations (no PP or no TG row) are skipped.
    Returns ``[]`` when the file cannot be parsed or holds no usable rows.

    Note: only throughput columns (``avg_ts`` / ``t/s``) are evaluated;
    latency columns (``avg_ns``) are not scored.
    """
    try:
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                return []
            # Case-insensitive match, consistent with the optimizer's
            # sequential parser (parse_bench_output): a header-case
            # variant must not silently yield "no complete configs".
            actual = {c.lower(): c for c in reader.fieldnames}
            perf_col = next(
                (actual[want] for want in ("avg_ts", "t/s")
                 if want in actual),
                None,
            )
            if perf_col is None:
                return []
            resolved: dict[str, str] = {}
            missing: list[str] = []
            for name, col in param_map.items():
                hit = actual.get(col.lower())
                if hit is None:
                    missing.append(f"{name} (column {col!r})")
                else:
                    resolved[name] = hit
            if missing:
                print(f"[ranking] Warning: column(s) not found in "
                      f"{Path(csv_path).name}: {', '.join(missing)}")
            if not resolved:
                return []
            items: list[dict[str, Any]] = []
            for row in reader:
                kind = _row_kind(row)
                if kind is None:
                    continue
                try:
                    value = float(row.get(perf_col, ""))
                except (TypeError, ValueError):
                    continue
                params = {name: str(row.get(col, "")).strip()
                          for name, col in resolved.items()}
                if any(v == "" for v in params.values()):
                    continue
                items.append({"params": params, "kind": kind,
                              "value": value})
    except (OSError, csv.Error):
        return []
    return group_measurements(items)


# ── Normalize / score / rank ───────────────────────────────────────────────

def normalize(
    entries: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute the min-max bounds of *entries* (PP and TG separately).

    Normalization always spans exactly the dataset handed in — in the
    optimizer that is the whole grid CSV, in LlamaGraph it will be the
    currently filtered view dataset. A zero span (all values equal)
    maps to ``1.0`` for every entry so equal candidates tie instead of
    dividing by zero. Raises ValueError on empty input or missing
    (None) metrics.
    """
    if not entries:
        raise ValueError("no entries to normalize")
    try:
        pps = [float(e["pp"]) for e in entries]
        tgs = [float(e["tg"]) for e in entries]
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid PP/TG metric: {exc}")
    return {"pp_min": min(pps), "pp_max": max(pps),
            "tg_min": min(tgs), "tg_max": max(tgs)}


def _norm_value(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    return (value - low) / (high - low)


def score(
    npp: float,
    ntg: float,
    w_tg: float,
    w_pp: float,
) -> float:
    """Weighted score from normalized PP/TG values (all in [0, 1])."""
    return w_tg * ntg + w_pp * npp


def rank(
    entries: list[dict[str, Any]],
    weights: list[tuple[float, float]],
    top_n: int = 1,
) -> dict[str, Any]:
    """Rank *entries* for every weighting, Top-N each.

    Returns the result set — a plain-data structure with dimensions
    (weighting, result type, rank)::

        {
          "weights": [(w_tg, w_pp), ...],
          "headers": ["TG", "70/30", ...],
          "norm": {"pp_min", "pp_max", "tg_min", "tg_max"},
          "top_n": top_n,
          "cells": cells[weight_idx][rank]
                   = {"params", "pp", "tg", "score"},
        }

    ``cells[w]`` holds at most ``top_n`` cells, best score first; rank
    ``0`` is the winner (the Markdown renderer displays it 1-based).
    Ties break deterministically by params. Raises ValueError on empty
    entries/weights or ``top_n < 1``.
    """
    if not entries:
        raise ValueError("no entries to rank")
    if not weights:
        raise ValueError("no weightings given")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    bounds = normalize(entries)
    cells: list[list[dict[str, Any]]] = []
    for w_tg, w_pp in weights:
        scored = []
        for entry in entries:
            npp = _norm_value(float(entry["pp"]),
                              bounds["pp_min"], bounds["pp_max"])
            ntg = _norm_value(float(entry["tg"]),
                              bounds["tg_min"], bounds["tg_max"])
            scored.append({
                "params": dict(entry["params"]),
                "pp": float(entry["pp"]),
                "tg": float(entry["tg"]),
                "score": score(npp, ntg, w_tg, w_pp),
            })
        scored.sort(key=lambda c: (-c["score"],
                                   tuple(sorted(c["params"].items()))))
        cells.append(scored[:top_n])
    return {"weights": list(weights),
            "headers": [header_for_weight(w) for w in weights],
            "norm": bounds, "top_n": top_n, "cells": cells}


# ── Rendering ──────────────────────────────────────────────────────────────

def format_params(params: dict[str, str]) -> str:
    """Format a param dict as a copy-paste-ready CLI fragment.

    Insertion order is preserved (the optimizer builds it in
    ``::optimize`` order); e.g.
    ``{"--ubatch-size": "256", "--batch-size": "1024"}`` becomes
    ``"--ubatch-size 256 --batch-size 1024"``.
    """
    return " ".join(f"{flag} {value}" for flag, value in params.items())


def render_markdown(result: dict[str, Any]) -> str:
    """Render a result set as a Markdown table (one 4-row block per rank).

    Layout (form 2 from the design discussion)::

        | # | Type  | TG  | 70/30 | ... | PP |
        | 1 | par   | ... | ...   |     |    |
        |   | pp    | ... | ...   |     |    |
        |   | tg    | ... | ...   |     |    |
        |   | score | ... | ...   |     |    |

    The ``par`` row stays copy-paste-ready; ``pp``/``tg`` carry raw t/s
    and ``score`` the weighted normalized score behind the rank.
    """
    headers: list[str] = result.get("headers", [])
    cells: list[list[dict[str, Any]]] = result.get("cells", [])
    if not headers or not cells:
        return "_No complete (pp+tg) configurations found._\n"
    depth = max((len(col) for col in cells), default=0)
    if depth == 0:
        return "_No complete (pp+tg) configurations found._\n"
    lines = ["| # | Type | " + " | ".join(headers) + " |",
             "|---|---|" + "|".join(["---"] * len(headers)) + "|"]
    for rank in range(depth):
        for row_idx, rtype in enumerate(RESULT_TYPES):
            num = str(rank + 1) if row_idx == 0 else ""
            label = rtype
            vals = []
            for col in cells:
                if rank >= len(col):
                    vals.append("")
                    continue
                cell = col[rank]
                if rtype == "par":
                    vals.append(format_params(cell["params"]))
                elif rtype == "pp":
                    vals.append(f"{cell['pp']:.2f}")
                elif rtype == "tg":
                    vals.append(f"{cell['tg']:.2f}")
                else:
                    vals.append(f"{cell['score']:.4f}")
            lines.append(f"| {num} | {label} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"
