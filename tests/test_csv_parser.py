"""Tests for utils/csv_parser.py - CSV parsing and parameter detection."""
import csv
import pathlib
from pathlib import Path
from utils.csv_parser import (
    get_bench_file_meta,
    parse_bench_csv,
    parse_bench_md,
    is_llama_bench_csv,
)


def test_parse_bench_csv(tmp_path):
    """Test parsing of llama-bench CSV format."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,gpu_name,model_name,params
1024,0,100.5,5.2,50250,2600,RTX 4090,Llama-2-70B,70B
0,256,85.3,4.1,42650,2050,RTX 4090,Llama-2-70B,70B
1024,0,120.1,6.3,60050,3150,RTX 3090,Mistral-7B,7B
""")
    
    result = parse_bench_csv(Path(csv_file))
    
    assert result is not None
    assert "source" in result
    assert "varying_params" in result
    assert "constant_params" in result
    assert "raw_rows" in result
    
    assert len(result["raw_rows"]) == 3
    assert len(result["varying_params"]) >= 1
    assert "gpu_name" in result["varying_params"] or "gpu_name" in result["constant_params"]


def test_parse_bench_csv_empty(tmp_path):
    """Test parsing of empty CSV file."""
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")
    
    result = parse_bench_csv(Path(csv_file))
    assert result is None


def test_parse_bench_csv_invalid(tmp_path):
    """Test parsing of invalid CSV file."""
    csv_file = tmp_path / "invalid.csv"
    csv_file.write_text("This is not a CSV")
    
    result = parse_bench_csv(Path(csv_file))
    assert result is None


def test_is_llama_bench_csv(tmp_path):
    """Test llama-bench CSV detection."""
    # Valid llama-bench CSV
    valid_file = tmp_path / "valid.csv"
    valid_file.write_text("n_prompt,n_gen,avg_ts,avg_ns\n1024,0,100,50000")
    assert is_llama_bench_csv(Path(valid_file))
    
    # Invalid (missing required columns)
    invalid_file = tmp_path / "invalid.csv"
    invalid_file.write_text("col1,col2,col3\na,b,c")
    assert not is_llama_bench_csv(Path(invalid_file))
    
    # Test with missing file
    missing_file = tmp_path / "missing.csv"
    assert not is_llama_bench_csv(Path(missing_file))


def test_parse_bench_csv_with_ignore_cols(tmp_path):
    """Test parsing with ignored columns."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,build_commit,model_filename,gpu_info,test_time
1024,0,100.5,5.2,50250,2600,abc123,model.gguf,RTX 4090,2024-01-01
0,256,85.3,4.1,42650,2050,def456,model2.gguf,RTX 3090,2024-01-02
""")
    
    result = parse_bench_csv(Path(csv_file))
    
    assert result is not None
    # Ignored columns should not appear in varying_params
    assert "build_commit" not in result["varying_params"]
    assert "model_filename" not in result["varying_params"]
    assert "gpu_info" not in result["varying_params"]
    assert "test_time" not in result["varying_params"]


def test_parse_bench_csv_malformed_data(tmp_path):
    """Test parsing with malformed data."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,custom_param
invalid,0,100.5,5.2,50250,2600,test1
1024,invalid,85.3,4.1,42650,2050,test2
1024,0,invalid,4.1,42650,2050,test3
1024,0,85.3,invalid,42650,2050,test4
""")
    
    result = parse_bench_csv(Path(csv_file))
    
    assert result is not None
    # Should skip malformed rows
    assert len(result["raw_rows"]) < 4


def test_parse_bench_csv_mixed_types(tmp_path):
    """Test parsing with mixed numeric/string parameters."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,str_param,num_param
1024,0,100.5,5.2,50250,2600,auto,128
0,256,85.3,4.1,42650,2050,manual,256
1024,0,120.1,6.3,60050,3150,auto,128
""")
    
    result = parse_bench_csv(Path(csv_file))
    
    assert result is not None
    assert "str_param" in result["varying_params"]
    assert "num_param" in result["varying_params"]
    assert any(row["str_param"] == "auto" for row in result["raw_rows"])
    assert any(row["num_param"] == 128.0 for row in result["raw_rows"])


def test_parse_bench_csv_directory_path(tmp_path):
    """Test parsing with a directory path (triggers file open exception)."""
    result = parse_bench_csv(tmp_path)
    assert result is None


def test_parse_bench_csv_non_pp_tg_rows(tmp_path):
    """Test rows that are neither pp nor tg (n_prompt=0, n_gen=0)."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
0,0,100.5,5.2,50250,2600,70B
1024,0,85.3,4.1,42650,2050,7B
""")
    
    result = parse_bench_csv(Path(csv_file))
    
    assert result is not None
    # Only the valid row should be present (the 0,0 row is skipped)
    assert len(result["raw_rows"]) == 1
    # Additionally, rows that result in no raw_rows should return None
    csv_file2 = tmp_path / "test2.csv"
    csv_file2.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns
0,0,100.5,5.2,50250,2600
0,0,85.3,4.1,42650,2050
""")
    assert parse_bench_csv(Path(csv_file2)) is None


def test_parse_bench_csv_empty_column_values(tmp_path):
    """Test parsing with empty column values."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,gpu_name,extra_param
1024,0,100.5,5.2,50250,2600,RTX 4090,
0,256,85.3,4.1,42650,2050,,value
""")
    
    result = parse_bench_csv(Path(csv_file))
    
    assert result is not None
    assert len(result["raw_rows"]) == 2
    # Row with empty extra_param should have None
    row0 = result["raw_rows"][0]
    assert "extra_param" in row0
    # Row with empty gpu_name should have None
    row1 = result["raw_rows"][1]
    assert row1["gpu_name"] is None


def test_parse_bench_csv_missing_ns_columns(tmp_path):
    """Test parsing when avg_ns/stddev_ns columns are missing."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts
1024,0,100.5,5.2
0,256,85.3,4.1
""")
    
    result = parse_bench_csv(Path(csv_file))
    
    assert result is not None
    assert len(result["raw_rows"]) == 2
    for row in result["raw_rows"]:
        assert row["ts_val"] is not None
        assert row["ns_val"] is None  # Should be None since column missing


def test_parse_bench_csv_invalid_ns_value(tmp_path):
    """Test parsing when avg_ns has an invalid value (triggers ValueError)."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns
1024,0,100.5,5.2,invalid,2600
0,256,85.3,4.1,42650,2050
""")
    
    result = parse_bench_csv(Path(csv_file))
    
    assert result is not None
    assert len(result["raw_rows"]) == 2
    # Row with invalid avg_ns should have ns_val=None
    assert result["raw_rows"][0]["ns_val"] is None
    # Row with valid avg_ns should have ns_val set
    assert result["raw_rows"][1]["ns_val"] == 42650.0


def test_parse_bench_csv_malformed_row_warning_print(capsys, tmp_path):
    """Malformed rows with invalid n_prompt/n_gen print warning to stdout."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns
invalid,0,100.5,5.2,50250,2600
1024,invalid,85.3,4.1,42650,2050
1024,0,100.5,5.2,50250,2600
""")

    result = parse_bench_csv(Path(csv_file))

    assert result is not None
    assert len(result["raw_rows"]) == 1  # only valid row
    captured = capsys.readouterr()
    assert captured.out
    assert "csv_parser" in captured.out
    assert "Skipping" in captured.out


def test_parse_bench_csv_non_pp_tg_warning_print(capsys, tmp_path):
    """Rows that are neither PP nor TG print warning to stdout."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns
0,0,100.5,5.2,50250,2600
1024,0,85.3,4.1,42650,2050
""")

    result = parse_bench_csv(Path(csv_file))

    assert result is not None
    assert len(result["raw_rows"]) == 1
    captured = capsys.readouterr()
    assert captured.out
    assert "csv_parser" in captured.out
    assert "Skipping" in captured.out


def test_parse_bench_csv_invalid_ts_warning_print(capsys, tmp_path):
    """Invalid avg_ts/stddev_ts values print warning to stdout."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns
1024,0,invalid,5.2,50250,2600
""")

    result = parse_bench_csv(Path(csv_file))

    assert result is not None
    captured = capsys.readouterr()
    assert captured.out
    assert "csv_parser" in captured.out
    assert "Invalid" in captured.out


def test_parse_bench_csv_keeps_file_meta(tmp_path):
    """Build/model metadata survives parsing (for the comparison filter)."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "build_commit,build_number,model_filename,model_type,"
        "n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns\n"
        "972d2313b,11028,G:\\models\\foo.gguf,TestType,"
        "1024,0,100.5,5.2,50250,2600\n"
        "972d2313b,11028,G:\\models\\foo.gguf,TestType,"
        "0,256,85.3,4.1,42650,2050\n"
    )

    result = parse_bench_csv(Path(csv_file))

    assert result is not None
    meta = result["file_meta"]
    assert meta["build_commit"] == "972d2313b"
    assert meta["build_number"] == "11028"
    assert meta["model_filename"] == "G:\\models\\foo.gguf"
    # Key/label use the basename so absolute paths compare across machines
    assert meta["model_key"] == "foo.gguf"
    assert meta["model_label"] == "foo.gguf"
    # ... while the fields stay out of the plot dimensions
    assert "build_number" not in result["varying_params"]
    assert "model_filename" not in result["varying_params"]


def test_get_bench_file_meta_missing_fields(tmp_path):
    """Files without metadata columns yield None entries (never raise)."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("n_prompt,n_gen,avg_ts,avg_ns\n1024,0,100,50000")

    meta = get_bench_file_meta(Path(csv_file))

    assert meta["build_number"] is None
    assert meta["model_key"] is None
    assert get_bench_file_meta(tmp_path / "does-not-exist.csv")["build_number"] is None


def test_parse_bench_md_keeps_file_meta(tmp_path):
    """MD tables expose build info + model label as file meta."""
    md_file = tmp_path / "test.md"
    md_file.write_text(
        "| model | backend | test | t/s |\n"
        "| ----- | ------- | ----: | ---: |\n"
        "| m1 | CUDA | pp512 | 50.0 ± 1.0 |\n"
        "| m1 | CUDA | tg128 | 2.0 ± 0.1 |\n"
        "\nbuild: 972d2313b (11028)\n"
    )

    result = parse_bench_md(Path(md_file))

    assert result is not None
    assert result["file_meta"]["build_number"] == "11028"
    assert result["file_meta"]["build_commit"] == "972d2313b"
    assert result["file_meta"]["model_key"] == "m1"
