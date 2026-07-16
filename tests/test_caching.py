import pytest
import numpy as np
from masktab.inference.caching import InferenceCache

def test_cache_row_hashing():
    cache = InferenceCache()
    row = {"a": 1, "b": np.float32(2.5), "c": "test"}
    h = cache.compute_row_hash(row)
    assert isinstance(h, str)
    
    # Order independence
    row2 = {"c": "test", "b": np.float32(2.5), "a": 1}
    h2 = cache.compute_row_hash(row2)
    assert h == h2

def test_cache_non_serializable():
    cache = InferenceCache()
    class CustomObj:
        pass
    row = {"a": CustomObj()}
    with pytest.raises(TypeError, match="not JSON serializable"):
        cache.compute_row_hash(row)

def test_cache_fifo_eviction():
    cache = InferenceCache(max_size=2)
    cache.set("hash1", {"res": 1})
    cache.set("hash2", {"res": 2})
    cache.set("hash3", {"res": 3})
    
    # hash1 should be evicted
    assert cache.get("hash1") is None
    assert cache.get("hash2") == {"res": 2}
    assert cache.get("hash3") == {"res": 3}
