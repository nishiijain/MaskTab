import pytest
import pandas as pd
from masktab.data.loader import load_csv, EmptyDatasetError

def test_load_csv_success(tmp_path):
    file_path = tmp_path / "data.csv"
    content = "cat_col,num_col\n"
    for i in range(25):
        content += f"val_{i % 10},{i}\n"
    file_path.write_text(content, encoding="utf-8")
    
    df, types = load_csv(str(file_path))
    assert list(df.columns) == ["cat_col", "num_col"]
    assert types["cat_col"] == "categorical"
    assert types["num_col"] == "numerical"

def test_load_csv_missing_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_csv("nonexistent.csv")

def test_load_csv_empty_rows(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("col1,col2\n", encoding="utf-8")
    with pytest.raises(EmptyDatasetError, match="zero rows"):
        load_csv(str(file_path))

def test_load_csv_empty_columns(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("", encoding="utf-8")
    with pytest.raises(EmptyDatasetError, match="zero columns"):
        load_csv(str(file_path))

def test_load_csv_duplicate_columns(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("col1,col1\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate column names found"):
        load_csv(str(file_path))

def test_load_csv_latin1_fallback(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_bytes(b"col1\n\xff\n")
    df, types = load_csv(str(file_path))
    assert not df.empty
