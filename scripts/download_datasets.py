import os
import requests
from masktab.data.loader import load_csv

class DatasetUnavailableError(Exception):
    pass

def fetch_uci_adult(output_dir: str) -> str:
    """
    Fetches the UCI Adult dataset (income prediction).
    Target column is "income" (binary classification).
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "adult.csv")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    
    try:
        if not os.path.exists(out_path):
            print(f"Downloading UCI Adult dataset to {out_path}...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(response.content)
    except Exception as e:
        print(f"Failed to download UCI Adult dataset: {e}")
        print("Please manually download the dataset from https://archive.ics.uci.edu/dataset/2/adult")
        print(f"and place the file (e.g. adult.data renamed to adult.csv) at: {out_path}")
        raise DatasetUnavailableError(f"UCI Adult dataset unavailable at {out_path}")
        
    try:
        load_csv(out_path)
    except Exception as e:
        print(f"Failed to parse {out_path}: {e}")
        print("Please manually download the dataset from https://archive.ics.uci.edu/dataset/2/adult")
        print(f"and place the correctly formatted CSV at: {out_path}")
        raise DatasetUnavailableError(f"UCI Adult dataset unavailable or corrupted at {out_path}")
        
    return out_path

def fetch_uci_covertype(output_dir: str) -> str:
    """
    Fetches the UCI Covertype dataset (forest cover type prediction).
    Target column is "Cover_Type" (multiclass classification, 7 classes).
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "covertype.csv")
    # Using the standard URL for covertype data file
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    
    try:
        if not os.path.exists(out_path):
            import gzip
            print(f"Downloading UCI Covertype dataset to {out_path}...")
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(gzip.decompress(response.content))
    except Exception as e:
        print(f"Failed to download UCI Covertype dataset: {e}")
        print("Please manually download the dataset from https://archive.ics.uci.edu/dataset/31/covertype")
        print(f"extract it, and place the CSV file at: {out_path}")
        raise DatasetUnavailableError(f"UCI Covertype dataset unavailable at {out_path}")
        
    try:
        load_csv(out_path)
    except Exception as e:
        print(f"Failed to parse {out_path}: {e}")
        print("Please manually download the dataset from https://archive.ics.uci.edu/dataset/31/covertype")
        print(f"extract it, and place the correctly formatted CSV at: {out_path}")
        raise DatasetUnavailableError(f"UCI Covertype dataset unavailable or corrupted at {out_path}")
        
    return out_path

if __name__ == "__main__":
    output_dir = "data"
    try:
        fetch_uci_adult(output_dir)
        print("Adult dataset fetched successfully.")
    except DatasetUnavailableError:
        pass
        
    try:
        fetch_uci_covertype(output_dir)
        print("Covertype dataset fetched successfully.")
    except DatasetUnavailableError:
        pass
