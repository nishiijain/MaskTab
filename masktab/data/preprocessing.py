import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self):
        self.fitted = False
        self.params = {}

    def fit(self, df: pd.DataFrame, column_types: dict, shared_columns: list):
        missing_cols = [col for col in shared_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"fit() called on DataFrame missing shared columns: {missing_cols}")

        self.params = {}
        for col in shared_columns:
            ctype = column_types[col]
            if ctype == "numerical":
                # Compute median from non-null
                non_null_series = df[col].dropna()
                if non_null_series.empty:
                    raise ValueError(f"Numerical column '{col}' is entirely null.")
                median_val = float(non_null_series.median())
                
                # Fit scaler
                scaler = StandardScaler()
                # Impute data just for fitting the scaler
                imputed_data = df[col].fillna(median_val).values.reshape(-1, 1)
                scaler.fit(imputed_data)
                
                self.params[col] = {
                    "type": "numerical",
                    "imputation_value": median_val,
                    "scaler_mean": float(scaler.mean_[0]),
                    "scaler_scale": float(scaler.scale_[0])
                }
            elif ctype == "categorical":
                non_null_series = df[col].dropna()
                if non_null_series.empty:
                    raise ValueError(f"Categorical column '{col}' is entirely null.")
                modes = non_null_series.mode()
                # Tie-break by picking the first in sorted order
                mode_val = sorted(modes.tolist())[0]
                if len(modes) > 1:
                    logger.info(f"Multiple modes found for {col}, selected first sorted: {mode_val}")
                
                cardinality = non_null_series.nunique()
                if cardinality > 10:
                    encoding_strategy = "embedding"
                    unique_vals = sorted(non_null_series.unique().tolist())
                    encoding_map = {val: idx + 1 for idx, val in enumerate(unique_vals)}
                    
                    self.params[col] = {
                        "type": "categorical",
                        "encoding_strategy": encoding_strategy,
                        "imputation_value": mode_val,
                        "encoding_map": encoding_map
                    }
                else:
                    encoding_strategy = "onehot"
                    unique_vals = sorted(non_null_series.unique().tolist())
                    
                    self.params[col] = {
                        "type": "categorical",
                        "encoding_strategy": encoding_strategy,
                        "imputation_value": mode_val,
                        "categories": unique_vals
                    }
            else:
                raise ValueError(f"Unknown column type {ctype} for column {col}")

        self.fitted = True

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list]:
        if not self.fitted:
            raise RuntimeError("transform() called before fit()")

        missing_cols = [col for col in self.params.keys() if col not in df.columns]
        if missing_cols:
            raise ValueError(f"transform() called on DataFrame missing shared columns: {missing_cols}")

        out_arrays = []
        out_columns = []

        for col, params in self.params.items():
            if params["type"] == "numerical":
                # Never use statistics computed from the data passed into transform()
                # Apply stored imputation values
                imputed = df[col].fillna(params["imputation_value"])
                
                # Apply stored StandardScaler transform
                scaled = (imputed.values - params["scaler_mean"]) / params["scaler_scale"]
                out_arrays.append(scaled.reshape(-1, 1))
                out_columns.append(col)
            elif params["type"] == "categorical":
                # Never use statistics computed from the data passed into transform()
                # Apply stored imputation values
                imputed = df[col].fillna(params["imputation_value"])
                
                if params["encoding_strategy"] == "embedding":
                    encoding_map = params["encoding_map"]
                    
                    def map_val(x):
                        return encoding_map.get(x, 0) # 0 is reserved for UNKNOWN
                    
                    mapped = imputed.apply(map_val)
                    unseen_count = (mapped == 0).sum()
                    if unseen_count > 0:
                        logger.warning(f"Column '{col}' has {unseen_count} unseen categories mapped to UNKNOWN (0).")
                    
                    out_arrays.append(mapped.values.reshape(-1, 1).astype(np.float32))
                    out_columns.append(col)
                elif params["encoding_strategy"] == "onehot":
                    categories = params["categories"]
                    # Map unseen to all zeros in one-hot
                    # Create dummy columns
                    onehot_arr = np.zeros((len(df), len(categories)), dtype=np.float32)
                    for i, cat in enumerate(categories):
                        onehot_arr[:, i] = (imputed == cat).astype(np.float32)
                    
                    out_arrays.append(onehot_arr)
                    for cat in categories:
                        out_columns.append(f"{col}_{cat}")

        result_array = np.hstack(out_arrays).astype(np.float32)
        return result_array, out_columns

    def fit_transform(self, df: pd.DataFrame, column_types: dict, shared_columns: list) -> tuple[np.ndarray, list]:
        self.fit(df, column_types, shared_columns)
        return self.transform(df)

    def to_dict(self) -> dict:
        return self.params.copy()

    @classmethod
    def from_dict(cls, data: dict):
        inst = cls()
        inst.params = data.copy()
        inst.fitted = len(data) > 0
        return inst
