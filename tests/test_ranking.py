"""Tests for model/ranking.py - weighted PP/TG ranking (pure data logic)."""
from __future__ import annotations

import pytest

from model import ranking as r


def _entries():
    # A: best PP, weak TG. B: balanced. C: best TG, weak PP.
    return [
        r.make_entry({"--ubatch-size": "512"}, pp=2000.0, tg=60.0),
        r.make_entry({"--ubatch-size": "256"}, pp=1500.0, tg=90.0),
        r.make_entry({"--ubatch-size": "128"}, pp=1000.0, tg=100.0),
    ]


def test_parse_weights_fractions_and_percentages():
    assert r.parse_weights("1/0,0/1") == [(1.0, 0.0), (0.0, 1.0)]
    assert r.parse_weights("70/30") == [(0.7, 0.3)]
    assert r.parse_weights("0.7/0.3") == pytest.approx([(0.7, 0.3)])


def test_parse_weights_invalid():
    for bad in ["", "foo", "1", "1/2/3", "-1/2", "0/0", "a/b"]:
        with pytest.raises(ValueError):
            r.parse_weights(bad)


def test_header_for_weight():
    assert r.header_for_weight((1.0, 0.0)) == "TG"
    assert r.header_for_weight((0.0, 1.0)) == "PP"
    assert r.header_for_weight((0.7, 0.3)) == "70/30"
    assert r.header_for_weight((0.5, 0.5)) == "50/50"


def test_default_weights_headers():
    weights = r.default_weights()
    assert [r.header_for_weight(w) for w in weights] == [
        "TG", "70/30", "50/50", "30/70", "PP"]


def test_normalize_bounds():
    bounds = r.normalize(_entries())
    assert bounds == {"pp_min": 1000.0, "pp_max": 2000.0,
                      "tg_min": 60.0, "tg_max": 100.0}


def test_normalize_rejects_missing_metrics():
    with pytest.raises(ValueError):
        r.normalize([r.make_entry({"a": "1"}, pp=None, tg=5.0)])
    with pytest.raises(ValueError):
        r.normalize([])


def test_normalize_zero_span():
    bounds = r.normalize([r.make_entry({"a": "1"}, pp=5.0, tg=5.0)])
    assert bounds["pp_min"] == bounds["pp_max"] == 5.0


def test_rank_extremes_pick_extremes():
    result = r.rank(_entries(), [(1.0, 0.0), (0.0, 1.0)], top_n=1)
    assert result["headers"] == ["TG", "PP"]
    assert result["cells"][0][0]["params"] == {"--ubatch-size": "128"}
    assert result["cells"][1][0]["params"] == {"--ubatch-size": "512"}


def test_rank_blended_winner_and_scores():
    result = r.rank(_entries(), [(0.5, 0.5)], top_n=3)
    col = result["cells"][0]
    # B is the best compromise: npp=0.5, ntg=0.75 -> 0.625
    assert col[0]["params"] == {"--ubatch-size": "256"}
    assert col[0]["score"] == pytest.approx(0.625)
    assert len(col) == 3
    # scores descend
    assert [c["score"] for c in col] == sorted(
        [c["score"] for c in col], reverse=True)


def test_rank_top_n_clamped_to_entries():
    result = r.rank(_entries()[:2], [(0.5, 0.5)], top_n=5)
    assert len(result["cells"][0]) == 2


def test_rank_rejects_bad_input():
    with pytest.raises(ValueError):
        r.rank([], [(1.0, 0.0)])
    with pytest.raises(ValueError):
        r.rank(_entries(), [])
    with pytest.raises(ValueError):
        r.rank(_entries(), [(1.0, 0.0)], top_n=0)


def test_group_measurements_merges_and_skips_incomplete():
    items = [
        {"params": {"b": "1"}, "kind": "pp", "value": 100.0},
        {"params": {"b": "1"}, "kind": "pp", "value": 120.0},  # max wins
        {"params": {"b": "1"}, "kind": "tg", "value": 10.0},
        {"params": {"b": "2"}, "kind": "pp", "value": 200.0},  # no tg -> skip
        {"params": {"b": "3"}, "kind": "tg", "value": 30.0},  # no pp -> skip
    ]
    entries = r.group_measurements(items)
    assert entries == [{"params": {"b": "1"}, "pp": 120.0, "tg": 10.0}]


def test_parse_grid_csv_groups_and_skips(tmp_path):
    csv_file = tmp_path / "grid.csv"
    csv_file.write_text(
        "test,n_ubatch,n_batch,avg_ts\n"
        "pp512,128,512,1000.0\n"
        "tg128,128,512,100.0\n"
        "pp512,256,512,2000.0\n"
        "tg128,256,512,60.0\n"
        "pp512,512,512,1500.0\n",  # tg row missing -> skipped
        encoding="utf-8",
    )
    entries = r.parse_grid_csv(
        csv_file, {"--ubatch-size": "n_ubatch", "--batch-size": "n_batch"})
    assert len(entries) == 2
    by_ub = {e["params"]["--ubatch-size"]: e for e in entries}
    assert by_ub["128"] == {"params": {"--ubatch-size": "128",
                                       "--batch-size": "512"},
                            "pp": 1000.0, "tg": 100.0}
    assert by_ub["256"]["pp"] == 2000.0


def test_parse_grid_csv_unknown_column(tmp_path, capsys):
    csv_file = tmp_path / "grid.csv"
    csv_file.write_text("test,avg_ts\ntg128,10.0\n", encoding="utf-8")
    assert r.parse_grid_csv(csv_file, {"--x": "nope"}) == []
    assert "not found" in capsys.readouterr().out


def test_parse_grid_csv_case_insensitive_columns(tmp_path):
    csv_file = tmp_path / "grid.csv"
    csv_file.write_text(
        "TEST,N_UBATCH,AVG_TS\n"
        "pp512,128,1000.0\n"
        "tg128,128,100.0\n",
        encoding="utf-8",
    )
    entries = r.parse_grid_csv(csv_file, {"--ubatch-size": "n_ubatch"})
    assert entries == [{"params": {"--ubatch-size": "128"},
                        "pp": 1000.0, "tg": 100.0}]


def test_render_markdown_layout():
    result = r.rank(_entries(), [(1.0, 0.0), (0.0, 1.0)], top_n=1)
    md = r.render_markdown(result)
    assert md.splitlines()[0] == "| # | Type | TG | PP |"
    # one 4-row block (par/pp/tg/score) for rank 1
    assert "| 1 | par |" in md
    assert "|  | pp |" in md
    assert "|  | tg |" in md
    assert "|  | score |" in md
    # par row stays free of metric values (copy-paste-ready)
    par_line = next(line for line in md.splitlines() if "| par |" in line)
    assert "--ubatch-size" in par_line


def test_render_markdown_empty():
    assert "No complete" in r.render_markdown(
        {"headers": [], "cells": []})


def test_format_params_order_preserved():
    assert r.format_params({"--ubatch-size": "256",
                            "--batch-size": "1024"}) == \
        "--ubatch-size 256 --batch-size 1024"
