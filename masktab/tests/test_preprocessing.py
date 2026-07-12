import pytest
import pandas as pd
import numpy as np
from masktab.data.preprocessing import Preprocessor

def test_fit_missing_shared_columns():
    df = pd.DataFrame({"a": [1, 2]})
    p = Preprocessor()
    with pytest.raises(ValueError, match="missing shared columns"):
        p.fit(df, {"b": "numerical"}, ["b"])

def test_fit_entirely_null_numerical():
    df = pd.DataFrame({"a": [np.nan, np.nan]})
    p = Preprocessor()
    with pytest.raises(ValueError, match="entirely null"):
        p.fit(df, {"a": "numerical"}, ["a"])

def test_fit_entirely_null_categorical():
    df = pd.DataFrame({"a": [None, None]})
    p = Preprocessor()
    with pytest.raises(ValueError, match="entirely null"):
        p.fit(df, {"a": "categorical"}, ["a"])

def test_transform_before_fit():
    df = pd.DataFrame({"a": [1, 2]})
    p = Preprocessor()
    with pytest.raises(RuntimeError, match="before fit"):
        p.transform(df)

def test_transform_unseen_categorical(caplog):
    df = pd.DataFrame({"a": [f"val_{i}" for i in range(15)]})
    p = Preprocessor()
    p.fit(df, {"a": "categorical"}, ["a"])
    
    df_new = pd.DataFrame({"a": ["unseen_val"]})
    arr, cols = p.transform(df_new)
    assert arr[0, 0] == 0
    assert "unseen categories mapped to UNKNOWN" in caplog.text

def test_serialization_roundtrip():
    df = pd.DataFrame({
        "num": [float(i) for i in range(24)] + [np.nan],
        "cat_embed": [f"v_{i}" for i in range(12)] + [np.nan] * 3 + [f"v_{i}" for i in range(10)],
        "cat_onehot": ["a", "b", "a", "b", "c"] * 5
    })
    
    types = {"num": "numerical", "cat_embed": "categorical", "cat_onehot": "categorical"}
    shared = list(types.keys())
    
    p1 = Preprocessor()
    p1.fit(df, types, shared)
    arr1, cols1 = p1.transform(df)
    
    d = p1.to_dict()
    p2 = Preprocessor.from_dict(d)
    arr2, cols2 = p2.transform(df)
    
    np.testing.assert_array_almost_equal(arr1, arr2)
    assert cols1 == cols2
