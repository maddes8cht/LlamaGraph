"""Tests for model/benchmark_model.py - Data model and filtering."""
import pytest
from pathlib import Path
from model.benchmark_model import BenchmarkModel


def test_initialization():
    """Test model initialization."""
    model = BenchmarkModel()
    assert model.has_data() == False
    assert model.get_dataset_count() == 0


def test_load_files(tmp_path):
    """Test loading CSV files."""
    model = BenchmarkModel()
    
    # Create test CSV files
    csv1 = tmp_path / "test1.csv"
    csv1.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,gpu_name,model_name,params
1024,0,100.5,5.2,50250,2600,RTX 4090,Llama-2-70B,70B
0,256,85.3,4.1,42650,2050,RTX 4090,Llama-2-70B,70B
""")
    
    csv2 = tmp_path / "test2.csv"
    csv2.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,gpu_name,model_name,params
1024,0,120.1,6.3,60050,3150,RTX 3090,Mistral-7B,7B
0,512,95.2,4.8,47600,2400,RTX 3090,Mistral-7B,7B
""")
    
    errors = model.load_files([Path(csv1), Path(csv2)])
    assert len(errors) == 0
    assert model.get_dataset_count() == 2
    assert model.has_data()


def test_get_dimensions(tmp_path):
    """Test dimension detection."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,gpu_name,model_name,params
1024,0,100.5,5.2,50250,2600,RTX 4090,Llama-2-70B,70B
0,256,85.3,4.1,42650,2050,RTX 4090,Llama-2-70B,70B
1024,0,120.1,6.3,60050,3150,RTX 3090,Mistral-7B,7B
""")
    
    model.load_files([Path(csv_file)])
    
    dimensions = model.get_dimensions()
    assert "gpu_name" in dimensions
    assert "model_name" in dimensions
    assert "params" in dimensions


def test_filter_management(tmp_path):
    """Test filter application."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,gpu_name,model_name,params
1024,0,100.5,5.2,50250,2600,RTX 4090,Llama-2-70B,70B
0,256,85.3,4.1,42650,2050,RTX 4090,Llama-2-70B,70B
1024,0,120.1,6.3,60050,3150,RTX 3090,Mistral-7B,7B
1024,0,115.0,5.8,57500,2900,RTX 3090,Mistral-7B,7B
""")
    
    model.load_files([Path(csv_file)])
    
    # Apply filter for RTX 4090
    model.apply_filters({"gpu_name": {"RTX 4090"}})
    
    # Check that filtering works
    dimensions = model.get_dimensions()
    assert "gpu_name" in dimensions


def test_get_filtered_rows(tmp_path):
    """Test filtered data retrieval."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,gpu_name,model_name,params
1024,0,100.5,5.2,50250,2600,RTX 4090,Llama-2-70B,70B
0,256,85.3,4.1,42650,2050,RTX 4090,Llama-2-70B,70B
1024,0,120.1,6.3,60050,3150,RTX 3090,Mistral-7B,7B
""")
    
    model.load_files([Path(csv_file)])
    
    # Get all rows
    all_rows = model.get_filtered_rows()
    assert len(all_rows) == 3
    
    # Filter by row type
    pp_rows = model.get_filtered_rows(row_type="pp")
    assert len(pp_rows) == 2


def test_2d_series_aggregation(tmp_path):
    """Test 2D series data aggregation."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
0,256,85.3,4.1,42650,2050,70B
1024,0,120.1,6.3,60050,3150,7B
1024,0,115.0,5.8,57500,2900,7B
""")
    
    model.load_files([Path(csv_file)])
    
    # Get time-series data
    series = model.get_2d_series("params", show_ts=True, show_pp=True, show_tg=False, normalize=False, scale_pct=False)
    
    assert "pp" in series
    assert len(series["pp"]) == 2  # Two unique param values


def test_3d_points_generation(tmp_path):
    """Test 3D point generation with numeric dimensions."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,n_prompt_batch,n_gen_batch
1024,0,100.5,5.2,50250,2600,128,64
0,256,85.3,4.1,42650,2050,128,64
1024,0,120.1,6.3,60050,3150,256,128
""")
    
    model.load_files([Path(csv_file)])
    
    # Get 3D points using numeric dimensions
    points_pp, points_tg = model.get_3d_points("n_prompt_batch", "n_gen_batch", show_ts=True, show_pp=True, show_tg=True, normalize=False, scale_pct=False)
    
    assert len(points_pp) == 2
    assert len(points_tg) == 1


def test_filter_with_extra_constraints(tmp_path):
    """Test filtering with extra constraints."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params,n_gpu_layers
1024,0,100.5,5.2,50250,2600,70B,32
0,256,85.3,4.1,42650,2050,70B,32
1024,0,120.1,6.3,60050,3150,7B,64
1024,0,115.0,5.8,57500,2900,7B,32
""")
    
    model.load_files([Path(csv_file)])
    
    # Filter with extra constraints
    filtered = model.get_filtered_rows(extra_filters={"n_gpu_layers": 32.0})
    assert len(filtered) == 3
    
    filtered_64 = model.get_filtered_rows(extra_filters={"n_gpu_layers": 64.0})
    assert len(filtered_64) == 1


def test_2d_series_with_normalization(tmp_path):
    """Test 2D series with normalization."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
0,256,85.3,4.1,42650,2050,70B
1024,0,200.0,10.0,100000,5000,7B
""")
    
    model.load_files([Path(csv_file)])
    
    # Get normalized series
    series = model.get_2d_series("params", show_ts=True, show_pp=True, show_tg=False, normalize=True, scale_pct=False)
    
    assert "pp" in series
    assert len(series["pp"]) == 2
    # Check that values are normalized (should be between 0 and 1)
    for point in series["pp"]:
        assert 0 <= point["y"] <= 1


def test_multiple_files_loading(tmp_path):
    """Test loading multiple files."""
    model = BenchmarkModel()
    
    csv1 = tmp_path / "test1.csv"
    csv1.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
""")
    
    csv2 = tmp_path / "test2.csv"
    csv2.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,120.1,6.3,60050,3150,7B
""")
    
    csv3 = tmp_path / "test3.csv"
    csv3.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,150.0,7.5,75000,3750,13B
""")
    
    errors = model.load_files([Path(csv1), Path(csv2), Path(csv3)])
    assert len(errors) == 0
    assert model.get_dataset_count() == 3
    
    # Check dimensions
    dimensions = model.get_dimensions()
    assert "params" in dimensions
    
    # Check that all parameter values are detected
    param_values = model.get_dim_values("params")
    assert len(param_values) == 3


def test_filter_reset(tmp_path):
    """Test filter reset functionality."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
0,256,85.3,4.1,42650,2050,7B
1024,0,120.1,6.3,60050,3150,13B
""")
    
    model.load_files([Path(csv_file)])
    
    # Apply restrictive filter
    model.apply_filters({"params": {"70B"}})
    filtered = model.get_filtered_rows()
    assert len(filtered) == 1
    
    # Reset filters
    model.reset_filters()
    all_rows = model.get_filtered_rows()
    assert len(all_rows) == 3


def test_empty_dataset_handling(tmp_path):
    """Test handling of empty datasets."""
    model = BenchmarkModel()
    
    # Load empty file
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")
    
    errors = model.load_files([Path(csv_file)])
    assert len(errors) == 1
    assert model.get_dataset_count() == 0
    assert not model.has_data()


def test_invalid_file_handling(tmp_path):
    """Test handling of invalid files."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "invalid.csv"
    csv_file.write_text("This is not a valid CSV file")
    
    errors = model.load_files([Path(csv_file)])
    assert len(errors) == 1
    assert "Could not parse" in errors[0]


def test_observer_notification(tmp_path):
    """Test observer notification mechanism."""
    model = BenchmarkModel()
    
    notification_count = 0
    def observer_callback():
        nonlocal notification_count
        notification_count += 1
    
    model.add_observer(observer_callback)
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
""")
    
    # Should notify on load
    model.load_files([Path(csv_file)])
    assert notification_count == 1
    
    # Should notify on filter change
    model.apply_filters({"params": {"70B"}})
    assert notification_count == 2
    
    # Should notify on clear
    model.clear()
    assert notification_count == 3


def test_clear_functionality(tmp_path):
    """Test clear functionality."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
""")
    
    model.load_files([Path(csv_file)])
    assert model.has_data()
    
    model.clear()
    assert not model.has_data()
    assert model.get_dataset_count() == 0


def test_get_all_dim_values(tmp_path):
    """Test get_all_dim_values."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
0,256,85.3,4.1,42650,2050,7B
""")
    
    model.load_files([Path(csv_file)])
    dim_values = model.get_all_dim_values()
    assert "params" in dim_values
    assert "70B" in dim_values["params"]
    assert "7B" in dim_values["params"]
    
    # Before any data is loaded, should return empty dict
    model.clear()
    assert model.get_all_dim_values() == {}


def test_filter_state(tmp_path):
    """Test get_filter_state."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
""")
    
    model.load_files([Path(csv_file)])
    state = model.get_filter_state()
    assert "params" in state
    assert isinstance(state["params"], set)


def test_observer_error_handling(tmp_path):
    """Test that observer errors don't propagate."""
    model = BenchmarkModel()
    
    def broken_observer():
        raise RuntimeError("Observer failed")
    
    model.add_observer(broken_observer)
    
    # This should not raise despite the broken observer
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
""")
    model.load_files([Path(csv_file)])
    # Observer error is silently caught, load should succeed
    assert model.has_data()


def test_get_dataset_paths(tmp_path):
    """Test get_dataset_paths."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
""")
    
    model.load_files([Path(csv_file)])
    paths = model.get_dataset_paths()
    assert len(paths) == 1
    assert str(paths[0]) == str(csv_file)
    
    # Before load, should be empty
    model.clear()
    assert model.get_dataset_paths() == []


def test_get_datasets_raw(tmp_path):
    """Test get_datasets_raw."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
""")
    
    model.load_files([Path(csv_file)])
    raw = model.get_datasets_raw()
    assert len(raw) == 1
    assert 'path' in raw[0]
    assert 'data' in raw[0]


def test_2d_series_with_tg(tmp_path):
    """Test 2D series with show_tg=True."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
0,256,85.3,4.1,42650,2050,70B
0,128,90.1,3.8,45050,1900,7B
""")
    
    model.load_files([Path(csv_file)])
    
    series = model.get_2d_series("params", show_ts=True, show_pp=True, show_tg=True, normalize=False, scale_pct=False)
    
    assert "pp" in series
    assert "tg" in series
    assert len(series["pp"]) == 1
    assert len(series["tg"]) == 2


def test_2d_series_filtered_rows_excluded(tmp_path):
    """Test 2D series with rows excluded by filters."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
0,256,85.3,4.1,42650,2050,70B
1024,0,120.1,6.3,60050,3150,7B
""")
    
    model.load_files([Path(csv_file)])
    
    # Apply a filter that only allows 70B
    model.apply_filters({"params": {"70B"}})
    
    series = model.get_2d_series("params", show_ts=True, show_pp=True, show_tg=True, normalize=False, scale_pct=False)
    
    # The 7B row should be excluded by the filter
    assert "pp" in series
    assert "tg" in series
    # Only 70B rows should be in the series
    assert len(series["pp"]) + len(series["tg"]) == 2
    for pt in series["pp"] + series["tg"]:
        assert pt["x"] == "70B"


def test_2d_series_invalid_x_dim(tmp_path):
    """Test 2D series with x_dim that doesn't resolve (returns None for x_val)."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
""")
    
    model.load_files([Path(csv_file)])
    
    # Use a dimension that doesn't exist
    series = model.get_2d_series("nonexistent_dim", show_ts=True, show_pp=True, show_tg=False, normalize=False, scale_pct=False)
    
    # Should be empty but not crash
    assert "pp" in series
    assert "tg" in series
    assert len(series["pp"]) == 0


def test_2d_series_missing_y_value(tmp_path):
    """Test 2D series when y_key values are missing (ts_val is None)."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    # File without avg_ts column → ts_val will be None for all rows
    csv_file.write_text("""n_prompt,n_gen,avg_ns,stddev_ns,params
1024,0,50250,2600,70B
0,256,42650,2050,70B
""")
    
    model.load_files([Path(csv_file)])
    
    # Request ts data (show_ts=True) → y_key='ts_val' → all rows have ts_val=None
    series = model.get_2d_series("params", show_ts=True, show_pp=True, show_tg=False, normalize=False, scale_pct=False)
    
    assert "pp" in series
    assert "tg" in series
    # All rows have ts_val=None, so they should be skipped
    assert len(series["pp"]) == 0
    assert len(series["tg"]) == 0
    
    # But ns data should work (show_ts=False → y_key='ns_val')
    series_ns = model.get_2d_series("params", show_ts=False, show_pp=True, show_tg=True, normalize=False, scale_pct=False)
    assert len(series_ns["pp"]) == 1
    assert len(series_ns["tg"]) == 1


def test_3d_points_with_normalization(tmp_path):
    """Test 3D points with normalization."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,n_prompt_batch,n_gen_batch
1024,0,100.5,5.2,50250,2600,128,64
0,256,85.3,4.1,42650,2050,128,64
1024,0,120.1,6.3,60050,3150,256,128
""")
    
    model.load_files([Path(csv_file)])
    
    points_pp, points_tg = model.get_3d_points("n_prompt_batch", "n_gen_batch", show_ts=True, show_pp=True, show_tg=True, normalize=True, scale_pct=False)
    
    # Should have values, and they should be normalized to [0, 1]
    assert len(points_pp) == 2
    assert len(points_tg) == 1
    for pt in points_pp + points_tg:
        assert 0 <= pt[2] <= 1  # z (index 2) should be in [0, 1]


def test_3d_points_string_dimension_encoded(tmp_path):
    """Test 3D points with string-valued dimension (categorical encoding)."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
""")
    
    model.load_files([Path(csv_file)])
    
    points_pp, points_tg = model.get_3d_points("params", "params", show_ts=True, show_pp=True, show_tg=False, normalize=False, scale_pct=False)
    
    # String "70B" is encoded to categorical code 0.0 → row is kept
    assert len(points_pp) == 1
    assert points_pp[0][0] == 0.0
    assert points_pp[0][1] == 0.0
    assert model.get_dim_labels("params") == ["70B"]


def test_3d_points_none_dimension(tmp_path):
    """Test 3D points with non-existent dimension (x_val is None)."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,n_prompt_batch
1024,0,100.5,5.2,50250,2600,128
""")
    
    model.load_files([Path(csv_file)])
    
    # x_dim doesn't exist → _resolve_param returns None → any(... is None) → continue
    points_pp, points_tg = model.get_3d_points("nonexistent_x", "n_prompt_batch", show_ts=True, show_pp=True, show_tg=False, normalize=False, scale_pct=False)
    
    assert len(points_pp) == 0


def test_3d_points_filtered(tmp_path):
    """Test 3D points with filters applied."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,n_prompt_batch,n_gen_batch
1024,0,100.5,5.2,50250,2600,128,64
0,256,85.3,4.1,42650,2050,128,64
1024,0,120.1,6.3,60050,3150,256,128
""")
    
    model.load_files([Path(csv_file)])
    
    # Apply filter that excludes all rows
    model.apply_filters({"n_prompt_batch": {999}})
    
    points_pp, points_tg = model.get_3d_points("n_prompt_batch", "n_gen_batch", show_ts=True, show_pp=True, show_tg=True, normalize=False, scale_pct=False)
    
    assert len(points_pp) == 0
    assert len(points_tg) == 0


def test_resolve_param_fallback(tmp_path):
    """Test _resolve_param falling back to constant params."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,const_param
1024,0,100.5,5.2,50250,2600,fixed_value
""")
    
    model.load_files([Path(csv_file)])
    
    # Access internal method directly
    datasets = model.get_datasets_raw()
    data = datasets[0]['data']
    consts = data['constant_params']
    row = data['raw_rows'][0]
    
    # Resolve a param not in row but in consts
    val = model._resolve_param(row, consts, "const_param")
    assert val is not None
    
    # Resolve a param not in row or consts (should return None)
    val = model._resolve_param(row, consts, "nonexistent")
    assert val is None


def test_normalize_3d_points_edge_cases(tmp_path):
    """Test _normalize_3d_points with edge cases."""
    from model.benchmark_model import _normalize_3d_points
    
    # Empty list should return empty
    assert _normalize_3d_points([], True) == []
    assert _normalize_3d_points([], False) == []
    
    # Normal list should work
    points = [(1.0, 2.0, 100.0, 5.0), (1.0, 2.0, 50.0, 3.0)]
    result = _normalize_3d_points(points, False)
    assert len(result) == 2
    # First point has z=100, max is 100, so first z becomes 1.0
    assert result[0][2] == 1.0
    assert result[1][2] == 0.5
    
    # Scale to percent
    result_pct = _normalize_3d_points(points, True)
    assert result_pct[0][2] == 100.0
    assert result_pct[1][2] == 50.0


def test_empty_allowed_set_skip(tmp_path):
    """Test that a dimension with empty allowed set is skipped in row_passes_filters."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
0,256,85.3,4.1,42650,2050,7B
""")
    
    model.load_files([Path(csv_file)])
    
    # Apply filter with empty set for a dimension (should be skipped = no restriction)
    model.apply_filters({"params": set()})
    
    # All rows should still pass because empty set is skipped
    rows = model.get_filtered_rows()
    assert len(rows) == 2


def test_row_passes_filters_fallback_to_consts(tmp_path):
    """Test _row_passes_filters fallback to constant params when val is None."""
    model = BenchmarkModel()
    
    # File A: has "gpu_name" as a varying column
    csv_a = tmp_path / "file_a.csv"
    csv_a.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,gpu_name,params
1024,0,100.5,5.2,50250,2600,GPU_A,70B
""")
    
    # File B: no "gpu_name" column at all
    csv_b = tmp_path / "file_b.csv"
    csv_b.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,120.1,6.3,60050,3150,7B
""")
    
    model.load_files([Path(csv_a), Path(csv_b)])
    
    # "gpu_name" is now in _active_filters (from file A)
    rows = model.get_filtered_rows()
    # File A row has gpu_name="GPU_A" → stays
    # File B row has no gpu_name → falls back to consts.get("gpu_name", '') → '' → float('') fails → val=''
    # '' not in {"GPU_A"} → filtered out
    # So only 1 row should remain
    assert len(rows) == 1
    assert all("gpu_name" in row for row in rows)


def test_row_passes_extra_fallback_to_consts(tmp_path):
    """Test _row_passes_extra fallback to constant params when val is None."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
0,256,85.3,4.1,42650,2050,7B
""")
    
    model.load_files([Path(csv_file)])
    
    # Extra filter for a dimension not in any file → row.get() returns None → fallback to consts
    rows_extra = model.get_filtered_rows(extra_filters={"nonexistent_dim": "some_value"})
    # val becomes '' after fallback → '' != "some_value" → all rows filtered out
    assert len(rows_extra) == 0
    
    # Extra filter for a dimension that exists and matches
    rows_match = model.get_filtered_rows(extra_filters={"params": "70B"})
    assert len(rows_match) == 1
    
    # Extra filter for a dimension that exists but doesn't match
    rows_no_match = model.get_filtered_rows(extra_filters={"params": "13B"})
    assert len(rows_no_match) == 0


def test_get_dim_values_empty(tmp_path):
    """Test get_dim_values for non-existent dimension."""
    model = BenchmarkModel()
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,params
1024,0,100.5,5.2,50250,2600,70B
""")
    
    model.load_files([Path(csv_file)])
    
    vals = model.get_dim_values("nonexistent")
    assert vals == []


# ── get_3d_points with string params ──────────────────────────────────────


def test_get_3d_points_string_params_returns_categorical(tmp_path):
    """String-valued dimension params are encoded to categorical codes."""
    model = BenchmarkModel()

    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,gpu_name,params
1024,0,100.5,5.2,50250,2600,RTX 4090,70B
0,256,85.3,4.1,42650,2050,RTX 4090,70B
""")

    model.load_files([Path(csv_file)])

    pp, tg = model.get_3d_points(
        x_dim="gpu_name", y_dim="params",
        show_ts=True, show_pp=True, show_tg=True,
        normalize=False, scale_pct=False,
    )

    assert isinstance(pp, list)
    assert isinstance(tg, list)
    assert len(pp) == 1  # one PP row
    assert len(tg) == 1  # one TG row
    # Both points get code 0 for gpu_name="RTX 4090" and code 0 for params="70B"
    x_vals = {p[0] for p in pp + tg}
    y_vals = {p[1] for p in pp + tg}
    assert x_vals == {0.0}
    assert y_vals == {0.0}

    # Verify categorical labels are stored
    assert model.get_dim_labels("gpu_name") == ["RTX 4090"]
    assert model.get_dim_labels("params") == ["70B"]


def test_get_3d_points_mixed_params_encodes_string(tmp_path):
    """String params in one dimension get encoded; numeric dim stays numeric."""
    model = BenchmarkModel()

    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,gpu_name,n_gpu_layers
1024,0,100.5,5.2,50250,2600,RTX 4090,20
0,256,85.3,4.1,42650,2050,RTX 4090,99
""")

    model.load_files([Path(csv_file)])

    pp, tg = model.get_3d_points(
        x_dim="gpu_name", y_dim="n_gpu_layers",
        show_ts=True, show_pp=True, show_tg=True,
        normalize=False, scale_pct=False,
    )

    assert isinstance(pp, list)
    assert isinstance(tg, list)
    assert len(pp) == 1
    assert len(tg) == 1
    # x_dim is string (encoded to 0.0), y_dim is numeric (20.0 / 99.0)
    for pt in pp:
        assert pt[0] == 0.0  # encoded from "RTX 4090"
        assert pt[1] == 20.0  # n_gpu_layers as float
    for pt in tg:
        assert pt[0] == 0.0
        assert pt[1] == 99.0

    assert model.get_dim_labels("gpu_name") == ["RTX 4090"]
    assert model.get_dim_labels("n_gpu_layers") is None  # numeric


def test_get_3d_points_numeric_params_works(tmp_path):
    """Fully numeric dimensions produce valid points without encoding."""
    model = BenchmarkModel()

    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""n_prompt,n_gen,avg_ts,stddev_ts,avg_ns,stddev_ns,n_gpu_layers
1024,0,100.5,5.2,50250,2600,20
0,256,85.3,4.1,42650,2050,99
""")

    model.load_files([Path(csv_file)])

    pp, tg = model.get_3d_points(
        x_dim="n_gpu_layers", y_dim="n_gpu_layers",
        show_ts=True, show_pp=True, show_tg=True,
        normalize=False, scale_pct=False,
    )

    assert isinstance(pp, list)
    assert isinstance(tg, list)
    assert len(pp) == 1
    assert len(tg) == 1

    assert model.get_dim_labels("n_gpu_layers") is None  # numeric, no mapping
