import json
import hashlib
import numpy as np

class InferenceCache:
    """
    An in-memory, session-scoped cache for inference results.
    Note: This cache does not survive process restart. Persistent caching
    is a future extension requiring an inference_cache SQLite table.
    """
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache = {} # standard dict maintains insertion order for FIFO eviction

    def _default_serializer(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def compute_row_hash(self, row: dict) -> str:
        """
        Serializes the row dict to JSON with sort_keys=True and computes a SHA256 hash.
        Raises TypeError if the row contains non-JSON-serializable values (after trying 
        to convert known numpy scalar types).
        """
        try:
            # We explicitly handle numpy scalar conversion via default=self._default_serializer.
            serialized = json.dumps(row, sort_keys=True, default=self._default_serializer)
        except TypeError as e:
            raise TypeError(f"Row contains non-JSON-serializable values: {e}")
            
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def get(self, row_hash: str) -> dict | None:
        """Retrieves a cached result by row hash."""
        return self._cache.get(row_hash)

    def set(self, row_hash: str, result: dict) -> None:
        """Stores a result in the cache, applying FIFO eviction if max_size is exceeded."""
        if row_hash not in self._cache:
            if len(self._cache) >= self.max_size:
                # FIFO eviction: pop the first inserted key
                first_key = next(iter(self._cache))
                del self._cache[first_key]
        self._cache[row_hash] = result
