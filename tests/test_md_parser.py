"""Tests for llama-bench markdown (`-o md`) parsing in utils/csv_parser.py."""
from pathlib import Path

from model.benchmark_model import BenchmarkModel
from utils.csv_parser import (
    is_llama_bench_csv,
    is_llama_bench_md,
    parse_bench_file,
    parse_bench_md,
)

MD_SAMPLE = """\
| model | size | params | backend | ngl | n_batch | n_ubatch | type_k | type_v | fa | test | t/s |
| ----- | ---: | -----: | ------- | --: | ------: | -------: | -----: | -----: | -: | ----: | ---: |
| qwen35 27B Q6_K | 23.90 GiB | 26.90 B | CUDA | 22 | 1024 | 256 | q8_0 | q4_0 | 1 | pp2048 | 101.48 ± 2.22 |
| qwen35 27B Q6_K | 23.90 GiB | 26.90 B | CUDA | 22 | 1024 | 256 | q8_0 | q4_0 | 1 | tg512 | 1.44 ± 0.01 |

build: 400ac8e19 (8674)
"""


def _write(tmp_path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_parse_md_basic(tmp_path):
    result = parse_bench_md(_write(tmp_path, "bench.md", MD_SAMPLE))

    assert result is not None
    assert len(result["raw_rows"]) == 2

    pp = next(r for r in result["raw_rows"] if r["type"] == "pp")
    tg = next(r for r in result["raw_rows"] if r["type"] == "tg")
    assert pp["ts_val"] == 101.48
    assert pp["ts_err"] == 2.22
    assert tg["ts_val"] == 1.44

    # ns derived via ns = 1e9 * N / ts
    assert pp["ns_val"] == 1e9 * 2048 / 101.48
    assert tg["ns_val"] == 1e9 * 512 / 1.44


def test_parse_md_aliases_and_build(tmp_path):
    result = parse_bench_md(_write(tmp_path, "bench.md", MD_SAMPLE))

    assert result is not None
    # short md names are aliased to csv counterparts
    assert "n_gpu_layers" in result["raw_rows"][0]
    assert "ngl" not in result["raw_rows"][0]
    assert "flash_attn" in result["raw_rows"][0]
    assert "backends" in result["raw_rows"][0]
    # build line becomes constants (also in _IGNORE_COLS, so no dims)
    assert result["constant_params"]["build_commit"] == "400ac8e19"
    assert result["constant_params"]["build_number"] == "8674"


def test_parse_md_varying_detection(tmp_path):
    content = MD_SAMPLE.replace("| 22 | 1024 | 256 |", "| 26 | 1024 | 256 |", 1)
    # only the second row keeps ngl=22 -> n_gpu_layers varies
    result = parse_bench_md(_write(tmp_path, "bench.md", content))

    assert result is not None
    assert "n_gpu_layers" in result["varying_params"]


def test_parse_md_keeps_test_sizes(tmp_path):
    """Same-type different-size rows stay distinguishable via n_prompt/n_gen."""
    content = (
        "| model | test | t/s |\n"
        "| ----- | ----: | ---: |\n"
        "| m1 | pp512 | 50.0 ± 1.0 |\n"
        "| m1 | pp1024 | 60.0 ± 1.5 |\n"
        "| m1 | tg128 | 2.0 ± 0.1 |\n"
    )
    result = parse_bench_md(_write(tmp_path, "bench.md", content))

    assert result is not None
    assert len(result["raw_rows"]) == 3
    assert "n_prompt" in result["varying_params"]

    pp_rows = [r for r in result["raw_rows"] if r["type"] == "pp"]
    assert sorted(r["n_prompt"] for r in pp_rows) == [512.0, 1024.0]
    assert all(r["n_gen"] == 0.0 for r in pp_rows)

    # n_prompt works as a plot axis (previously yielded zero points)
    model = BenchmarkModel()
    assert model.load_files([tmp_path / "bench.md"]) == []
    series = model.get_2d_series(
        x_dim="n_prompt",
        show_ts=True,
        show_pp=True,
        show_tg=False,
        normalize=False,
        scale_pct=False,
    )
    assert sorted(p["x"] for p in series["pp"]) == [512.0, 1024.0]


def test_parse_md_misnamed_csv_extension(tmp_path):
    # md table with .csv extension (the original bug) dispatches to md parser
    p = _write(tmp_path, "bench_phase1_123.csv", MD_SAMPLE)
    assert not is_llama_bench_csv(p)
    assert is_llama_bench_md(p)
    result = parse_bench_file(p)
    assert result is not None
    assert len(result["raw_rows"]) == 2


def test_is_md_rejects_csv_and_txt(tmp_path):
    csv_file = _write(
        tmp_path, "valid.csv", "n_prompt,n_gen,avg_ts,avg_ns\n1024,0,100,50000"
    )
    assert not is_llama_bench_md(csv_file)

    txt_file = _write(tmp_path, "notes.txt", "just some text\nno tables here\n")
    assert not is_llama_bench_md(txt_file)

    missing = tmp_path / "missing.md"
    assert not is_llama_bench_md(missing)


def test_parse_md_skips_bad_rows(tmp_path):
    content = """\
| model | backend | ngl | test | t/s |
| ----- | ------- | --: | ----: | ---: |
| m1 | CUDA | 22 | pp512 | 50.0 ± 1.0 |
| m1 | CUDA | 22 | bogus | 50.0 ± 1.0 |
| m1 | CUDA | 22 | tg128 | not-a-number |
"""
    result = parse_bench_md(_write(tmp_path, "bench.md", content))
    assert result is not None
    assert len(result["raw_rows"]) == 1
    assert result["raw_rows"][0]["type"] == "pp"


def test_parse_md_dangling_ts_separator(tmp_path):
    """`t/s` cells like `50.0 ±` (no error value) skip the row, never crash."""
    content = (
        "| model | test | t/s |\n"
        "| ----- | ----: | ---: |\n"
        "| m1 | pp512 | 50.0 ± |\n"
        "| m1 | pp512 | 50.0 +/- |\n"
        "| m1 | tg128 | 2.0 ± 0.1 |\n"
    )
    result = parse_bench_md(_write(tmp_path, "bench.md", content))
    assert result is not None
    assert len(result["raw_rows"]) == 1
    assert result["raw_rows"][0]["type"] == "tg"


def test_parse_md_headers_case_insensitive(tmp_path):
    """Uppercase headers yield lowercased dims, unified with CSV names."""
    content = (
        "| MODEL | BACKEND | NGL | TEST | T/S |\n"
        "| ----- | ------- | --: | ----: | ---: |\n"
        "| m1 | CUDA | 22 | pp512 | 50.0 ± 1.0 |\n"
        "| m1 | CUDA | 26 | tg128 | 2.0 ± 0.1 |\n"
    )
    result = parse_bench_md(_write(tmp_path, "bench.md", content))
    assert result is not None
    assert len(result["raw_rows"]) == 2
    assert "n_gpu_layers" in result["varying_params"]
    assert "model" in result["raw_rows"][0]
    assert "backends" in result["raw_rows"][0]


def test_parse_md_empty_and_header_only(tmp_path):
    assert parse_bench_md(_write(tmp_path, "e.md", "")) is None
    header_only = (
        "| model | test | t/s |\n| ----- | ----: | ---: |\n"
    )
    assert parse_bench_md(_write(tmp_path, "h.md", header_only)) is None


def test_model_loads_md(tmp_path):
    md_file = _write(tmp_path, "bench.md", MD_SAMPLE)
    model = BenchmarkModel()
    errors = model.load_files([md_file])

    assert errors == []
    assert model.has_data()
    assert model.get_dataset_count() == 1
    # pp + tg rows pass the default filters
    assert len(model.get_filtered_rows()) == 2
    assert len(model.get_filtered_rows(row_type="pp")) == 1
    # 2-D aggregation works on md data
    series = model.get_2d_series(
        x_dim="n_gpu_layers",
        show_ts=True,
        show_pp=True,
        show_tg=True,
        normalize=False,
        scale_pct=False,
    )
    assert len(series["pp"]) == 1
    assert len(series["tg"]) == 1
