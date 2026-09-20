"""Tests for tools/llama-optimizer/llama-optimizer.py - pure helper functions.

The script filename contains a hyphen, so it is loaded via importlib instead
of a regular import. Only side-effect-free helpers are tested here (no
subprocess, no tkinter dialogs, no llama-bench binary needed).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools" / "llama-optimizer"


def _load():
    spec = importlib.util.spec_from_file_location(
        "llama_optimizer", TOOLS_DIR / "llama-optimizer.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


opt = _load()


def test_tool_files_exist():
    """The vendored tool directory is complete (script + both docs)."""
    assert (TOOLS_DIR / "llama-optimizer.py").is_file()
    assert (TOOLS_DIR / "llama-optimizer.README.md").is_file()
    assert (TOOLS_DIR / "llama-optimizer.params.README.md").is_file()


def test_resolve_output_format():
    assert opt.resolve_output_format(None, {}) == "both"
    assert opt.resolve_output_format("md", {}) == "md"
    assert opt.resolve_output_format("csv", {"output-format": "md"}) == "csv"
    assert opt.resolve_output_format(None, {"output-format": "csv"}) == "csv"
    assert opt.resolve_output_format(None, {"output-format": "bogus"}) == "both"
    assert opt.resolve_output_format(None, {"output-format": True}) == "both"


def test_strip_output_format_args():
    args = ["-m", "x.gguf", "-o", "md", "-b", "512", "-oe", "json", "--output=csv"]
    assert opt.strip_output_format_args(args) == ["-m", "x.gguf", "-b", "512"]
    assert opt.strip_output_format_args(["-o"]) == []
    assert opt.strip_output_format_args([]) == []


def test_output_args_for():
    assert opt.output_args_for("csv") == ["-o", "csv"]
    assert opt.output_args_for("md") == ["-o", "md"]
    assert opt.output_args_for("both") == ["-o", "csv", "-oe", "md"]


def test_get_csv_column():
    assert opt.get_csv_column("--batch-size") == "n_batch"
    assert opt.get_csv_column("-ngl") == "n_gpu_layers"
    assert opt.get_csv_column("--ubatch-size") == "n_ubatch"
    assert opt.get_csv_column("--custom-flag") == "custom_flag"


def test_load_params_recursive(tmp_path):
    params = tmp_path / "params.txt"
    params.write_text(
        "# comment line\n"
        "-m model.gguf -b 1024\n"
        "::optimize --ubatch-size 64,128,256\n"
        "::optimize-order --ubatch-size\n"
        "::output-format csv\n",
        encoding="utf-8",
    )
    configs, base_args, targets = opt.load_params_recursive(params)

    assert configs["optimize-order"] == "--ubatch-size"
    assert configs["output-format"] == "csv"
    assert "-m" in base_args and "model.gguf" in base_args
    assert list(targets) == ["--ubatch-size"]
    assert targets["--ubatch-size"] == ["64", "128", "256"]


def test_load_params_space_separated_values(tmp_path):
    """::optimize also accepts space-separated values (no commas)."""
    params = tmp_path / "params.txt"
    params.write_text(
        "--batch-size 1024\n"
        "::optimize --ubatch-size 64 128 256\n"
        "::optimize --batch-size 512 1024 2048\n",
        encoding="utf-8",
    )
    _, _, targets = opt.load_params_recursive(params)

    assert targets["--ubatch-size"] == ["64", "128", "256"]
    assert targets["--batch-size"] == ["512", "1024", "2048"]


def test_load_params_ignores_metadata(tmp_path):
    """Tokens containing '=' are metadata, in comma and space style."""
    params = tmp_path / "params.txt"
    params.write_text(
        "::optimize --ubatch-size 64,128,256 metric=tg\n"
        "::optimize --batch-size 512 1024 2048 mode=sequential\n",
        encoding="utf-8",
    )
    _, _, targets = opt.load_params_recursive(params)

    assert targets["--ubatch-size"] == ["64", "128", "256"]
    assert targets["--batch-size"] == ["512", "1024", "2048"]


def test_nested_params_including_file_wins(tmp_path):
    """Directives, targets: including file overrides the included one."""
    base = tmp_path / "base.txt"
    base.write_text(
        "-b 512\n"
        "::output-dir base_dir\n"
        "::optimize --ubatch-size 64,128\n"
        "::optimize --batch-size 512\n",
        encoding="utf-8",
    )
    top = tmp_path / "top.txt"
    top.write_text(
        f"::params-file {base}\n"
        "-b 1024\n"
        "::output-dir top_dir\n"
        "::optimize --ubatch-size 256,512\n",
        encoding="utf-8",
    )
    configs, base_args, targets = opt.load_params_recursive(top)

    assert configs["output-dir"] == "top_dir"
    assert targets["--ubatch-size"] == ["256", "512"]
    assert targets["--batch-size"] == ["512"]  # inherited from base
    # included args first, including file's appended (later CLI wins)
    assert base_args == ["-b", "512", "-b", "1024"]


def test_nested_params_chain(tmp_path):
    """Three levels: nearest including file wins, rest is inherited."""
    c = tmp_path / "c.txt"
    c.write_text("::output-dir c_dir\n::output-format md\n", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text(f"::params-file {c}\n::output-dir b_dir\n", encoding="utf-8")
    a = tmp_path / "a.txt"
    a.write_text(f"::params-file {b}\n::output-dir a_dir\n", encoding="utf-8")

    configs, _, _ = opt.load_params_recursive(a)
    assert configs["output-dir"] == "a_dir"
    assert configs["output-format"] == "md"  # inherited through b from c


def test_load_params_missing_file(tmp_path):
    configs, base_args, targets = opt.load_params_recursive(
        tmp_path / "does-not-exist.txt"
    )
    assert configs == {} and base_args == [] and len(targets) == 0


def _bench_csv(path: Path) -> Path:
    path.write_text(
        "n_batch,n_prompt,n_gen,avg_ts\n"
        "512,1024,0,100.0\n"
        "512,0,128,10.0\n"
        "256,1024,0,90.0\n"
        "256,0,128,12.0\n",
        encoding="utf-8",
    )
    return path


def test_parse_bench_output(tmp_path):
    results = opt.parse_bench_output(_bench_csv(tmp_path / "b.csv"), "--batch-size")
    # only tg rows count, best per value wins
    assert results == {"512": 10.0, "256": 12.0}


def test_parse_bench_output_unknown_column(tmp_path, capsys):
    results = opt.parse_bench_output(
        _bench_csv(tmp_path / "b.csv"), "--no-such-flag"
    )
    assert results == {}
    assert "not found" in capsys.readouterr().out


def test_parse_grid_output(tmp_path):
    best, tg = opt.parse_grid_output(
        _bench_csv(tmp_path / "b.csv"), ["--batch-size"]
    )
    assert tg == 12.0
    assert best == {"--batch-size": "256"}


def test_extract_repetitions():
    assert opt.extract_repetitions(["-r", "3"]) == 3
    assert opt.extract_repetitions([]) == 1
    assert opt.extract_repetitions(["-r"]) == 1
    assert opt.extract_repetitions(["-r", "bogus"]) == 1


def test_resolve_top():
    assert opt.resolve_top(None, {}) == 1
    assert opt.resolve_top(3, {}) == 3
    assert opt.resolve_top(None, {"top": "2"}) == 2
    assert opt.resolve_top(2, {"top": "5"}) == 2  # CLI wins
    assert opt.resolve_top(0, {}) == 1
    assert opt.resolve_top(None, {"top": "bogus"}) == 1


def test_resolve_top_warns_on_invalid(capsys):
    assert opt.resolve_top(None, {"top": "bogus"}) == 1
    assert "Invalid ::top" in capsys.readouterr().out
    assert opt.resolve_top(None, {"top": "0"}) == 1
    assert "Invalid ::top" in capsys.readouterr().out
    # missing value stays silent (plain default)
    assert opt.resolve_top(None, {}) == 1
    assert capsys.readouterr().out == ""


def test_resolve_table_weights_strict_raises():
    import pytest
    with pytest.raises(ValueError):
        opt.resolve_table_weights("bogus", {}, strict=True)
    # lenient default still falls back
    assert opt.resolve_table_weights("bogus", {}) is not None


def test_resolve_table_weights_default_headers():
    from model import ranking as ranking_mod
    weights = opt.resolve_table_weights(None, {})
    assert [ranking_mod.header_for_weight(w) for w in weights] == [
        "TG", "70/30", "50/50", "30/70", "PP"]


def test_resolve_table_weights_cli_wins_and_invalid_falls_back(capsys):
    from model import ranking as ranking_mod
    weights = opt.resolve_table_weights("1/0,0/1", {"table-weights": "0.5/0.5"})
    assert weights == [(1.0, 0.0), (0.0, 1.0)]
    weights = opt.resolve_table_weights(None, {"table-weights": "bogus"})
    assert weights == ranking_mod.default_weights()
    assert "Invalid weighting" in capsys.readouterr().out


def test_build_grid_param_map():
    assert opt.build_grid_param_map(["--batch-size", "-ngl"]) == {
        "--batch-size": "n_batch", "-ngl": "n_gpu_layers"}


def test_grid_ranking_end_to_end(tmp_path):
    """Grid CSV -> ranking entries -> weighted result set (TG..PP)."""
    from model import ranking as ranking_mod
    csv_file = tmp_path / "grid.csv"
    csv_file.write_text(
        "test,n_ubatch,avg_ts\n"
        "pp512,128,1000.0\n"
        "tg128,128,100.0\n"
        "pp512,256,2000.0\n"
        "tg128,256,60.0\n",
        encoding="utf-8",
    )
    param_map = opt.build_grid_param_map(["--ubatch-size"])
    entries = ranking_mod.parse_grid_csv(csv_file, param_map)
    assert len(entries) == 2
    weights = opt.resolve_table_weights(None, {})
    result = ranking_mod.rank(entries, weights, opt.resolve_top(None, {}))
    assert result["headers"][0] == "TG" and result["headers"][-1] == "PP"
    assert result["cells"][0][0]["params"] == {"--ubatch-size": "128"}
    assert result["cells"][-1][0]["params"] == {"--ubatch-size": "256"}
    assert "70/30" in ranking_mod.render_markdown(result)


def test_extract_metadata(tmp_path):
    csv_file = tmp_path / "b.csv"
    csv_file.write_text(
        "build_number,cpu_info,backends,model_filename,model_type,model_size,n_prompt,n_gen,avg_ts\n"
        '"1234","my cpu","CUDA","m.gguf","qwen","20G",1024,0,100.0\n',
        encoding="utf-8",
    )
    meta = opt.extract_metadata(csv_file)
    assert meta["build_number"] == "1234"
    assert meta["backends"] == "CUDA"
    assert opt.extract_metadata(tmp_path / "missing.csv") == {}
