# `model/ranking.py`

> **🔗 Used by:** `tools/llama-optimizer/llama-optimizer.py` (grid mode) via `--top` / `--table-weights`.
> Planned, not yet implemented: LlamaGraph views (see below).

## 🔍 Overview
`model/ranking.py` answers one question: *given a defined set of benchmark measurements, which parameter combination is best — when prompt processing (PP) and token generation (TG) matter to different degrees?*

It groups raw PP/TG measurements into comparable entries, min-max normalizes both metrics over the dataset, scores every entry per TG/PP weighting, and returns the Top-N ranks per weighting as a plain-data result set. It renders nothing itself except an optional Markdown table string; each consumer owns its display.

## 🧠 Core Philosophy
- **Zero Assumptions about parameters:** entries carry an opaque `params` dict (`{"--ubatch-size": "256", ...}`). The module never validates flag names — callers map their own columns/flags to it.
- **Normalization before weighting:** PP throughput (`~1000 t/s`) dwarfs TG (`~100 t/s`). A raw weighted sum would always elect the PP champion, so both metrics are min-max normalized over the handed-in dataset first, then scored as `score = w_tg·ntg + w_pp·npp`.
- **Incomplete configs are skipped:** a combination without both a PP and a TG measurement cannot be ranked fairly and is dropped (never imputed).
- **No UI, no I/O side effects:** pure data logic, stdlib only — same layering as `model/benchmark_model.py`.

## ⚙️ How It Works
1. **Group:** measurements sharing the exact same parameter dict are merged (`max(t/s)` per kind wins, mirroring the optimizer's per-value aggregation).
2. **Normalize:** PP and TG bounds (`min`/`max`) are computed over the given dataset. Zero span maps to `1.0` (equal candidates tie).
3. **Score & rank:** per weighting `(w_tg, w_pp)` every entry is scored and sorted (ties break deterministically by params); each weighting keeps its Top-N.
4. **Render (optional):** `render_markdown()` turns the result set into a Markdown table; other consumers read `cells` directly.

## 📐 Why This Dataset Shape

The result set has three dimensions — **(weighting, result type, rank)**:

```python
{
  "weights": [(1.0, 0.0), (0.7, 0.3), ...],  # (w_tg, w_pp), TG share first
  "headers": ["TG", "70/30", ...],           # display labels
  "norm": {"pp_min", "pp_max", "tg_min", "tg_max"},
  "top_n": 1,
  "cells": cells[weight_idx][rank]
           = {"params", "pp", "tg", "score"},
}
```

Alternatives considered and rejected:
- **Wide columns** (`TG-par / TG-pp / TG-tg / 70/30-par / …`): explodes with every weighting and hard-codes the display into the data.
- **Type-as-rows storage** (`Type | TG | …` with `par/pp/tg` row groups): mixes storage with one particular table layout.

The 3-D form keeps data and display separate: the same `cells` render as the optimizer's Markdown block table today and as a native LlamaGraph view tomorrow, without re-scoring.

## 🏷️ Weight Convention

Headers and `--table-weights` items are **`TG/PP`** — the TG share first:

| Header | Meaning |
|---|---|
| `TG` | pure TG (`1/0`) |
| `70/30` | 70% TG + 30% PP |
| `50/50` | balanced |
| `30/70` | 30% TG + 70% PP |
| `PP` | pure PP (`0/1`) |

Specs accept fractions (`0.7/0.3`) or percentages (`70/30`); shares are normalized to sum 1. Default: `1/0,0.7/0.3,0.5/0.5,0.3/0.7,0/1`.

## 🖥️ API Usage
```python
from model import ranking as ranking

entries = ranking.parse_grid_csv(csv_path, {"--ubatch-size": "n_ubatch"})
# ... or ranking.group_measurements([{"params": {...}, "kind": "pp"|"tg", "value": ...}])
weights = ranking.parse_weights("1/0,0.7/0.3,0.5/0.5,0.3/0.7,0/1")
result = ranking.rank(entries, weights, top_n=3)
print(ranking.render_markdown(result))
```

The Markdown layout uses one 4-row block per rank (`par` = copy-paste-ready params, `pp`/`tg` = raw t/s, `score` = weighted normalized score); rank `0` in data displays as `1`.

## 🔮 Planned LlamaGraph Use (not yet implemented)

LlamaGraph will feed the *currently defined view dataset* — already filtered, possibly multi-file, possibly higher-dimensional than the 2D/3D view — through `group_measurements()` + `rank()`. Entry identity stays the plain parameter dict; file/model metadata is not part of ranking. LlamaGraph will provide its own native display instead of `render_markdown()`.

## 📦 Requirements
- Python 3.12+ (same as the repo root), stdlib only (`csv`, `pathlib`, `typing`).
- No `tkinter`, no `matplotlib`, no `llama-bench` dependency.

## ⚠️ Notes
- Normalization spans exactly the dataset handed in: grid CSV in the optimizer, filtered view rows in LlamaGraph. Scores from different datasets are not comparable.
- Scores are unitless ranks aids, not throughputs — always read them next to the raw `pp`/`tg` values.
