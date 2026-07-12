import pytest
from masktab.data.schema_validator import validate_schema_compatibility, SchemaIncompatibleError

def test_validate_schema_compatibility_success():
    pt = {"a": "numerical", "b": "categorical", "c": "numerical"}
    t = {"a": "numerical", "b": "categorical", "d": "numerical"}
    res = validate_schema_compatibility(pt, t)
    assert sorted(res["shared_columns"]) == ["a", "b"]
    assert res["shared_column_types"]["a"] == "numerical"
    assert res["pretraining_only_columns"] == ["c"]
    assert res["task_only_columns"] == ["d"]

def test_validate_schema_compatibility_zero_shared():
    pt = {"a": "numerical"}
    t = {"b": "numerical"}
    with pytest.raises(SchemaIncompatibleError, match="empty"):
        validate_schema_compatibility(pt, t)

def test_validate_schema_compatibility_fewer_than_min():
    pt = {"a": "numerical"}
    t = {"a": "numerical"}
    with pytest.raises(SchemaIncompatibleError, match="less than MIN_SHARED_FEATURES"):
        validate_schema_compatibility(pt, t)

def test_validate_schema_compatibility_type_conflict():
    pt = {"a": "numerical", "b": "categorical"}
    t = {"a": "categorical", "b": "categorical"}
    with pytest.raises(SchemaIncompatibleError, match="Type conflicts"):
        validate_schema_compatibility(pt, t)
