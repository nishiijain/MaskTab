import pandas as pd
import logging
import csv
import io

logger = logging.getLogger(__name__)

CATEGORICAL_UNIQUE_THRESHOLD = 20

class EmptyDatasetError(Exception):
    pass

def load_csv(filepath: str) -> tuple[pd.DataFrame, dict[str, str]]:
    try:
        with open(filepath, 'rb') as f:
            head = f.read(4096)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

    if head:
        try:
            header_str = head.decode('utf-8')
        except UnicodeDecodeError:
            header_str = head.decode('latin-1', errors='ignore')
            
        reader = csv.reader(io.StringIO(header_str.splitlines()[0]))
        try:
            cols = next(reader)
            cols = [str(c).strip() for c in cols]
            seen = set()
            dupes = []
            for c in cols:
                if c in seen:
                    dupes.append(c)
                seen.add(c)
            if dupes:
                raise ValueError(f"Duplicate column names found: {list(set(dupes))}")
        except StopIteration:
            pass

    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except pd.errors.EmptyDataError:
        raise EmptyDatasetError("Dataset has zero columns")
    except UnicodeDecodeError:
        logger.warning(f"Failed to read {filepath} with utf-8, retrying with latin-1")
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
        except pd.errors.EmptyDataError:
            raise EmptyDatasetError("Dataset has zero columns")

    if df.shape[1] == 0:
        raise EmptyDatasetError("Dataset has zero columns")
    if df.shape[0] == 0:
        raise EmptyDatasetError("Dataset has zero rows")

    df.columns = [str(c).strip() for c in df.columns]

    column_types_dict = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype) and dtype.name in ['int64', 'float64']:
            if df[col].nunique() > CATEGORICAL_UNIQUE_THRESHOLD:
                column_types_dict[col] = "numerical"
            else:
                column_types_dict[col] = "categorical"
        else:
            column_types_dict[col] = "categorical"

    return df, column_types_dict
