import hashlib
import os

def compute_dataset_hash(filepath: str, feature_names: list[str]) -> str:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if not feature_names:
        raise ValueError("feature_names cannot be empty")

    file_size = os.path.getsize(filepath)
    sorted_features = ",".join(sorted(feature_names))
    
    content_hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            content_hasher.update(chunk)
    
    content_hash = content_hasher.hexdigest()
    combined_string = f"{content_hash}|{file_size}|{sorted_features}"
    
    final_hasher = hashlib.sha256()
    final_hasher.update(combined_string.encode('utf-8'))
    
    return final_hasher.hexdigest()
