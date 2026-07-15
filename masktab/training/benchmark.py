import logging
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from masktab.data.loader import load_csv
from masktab.data.hashing import compute_dataset_hash
from masktab.data.registry import load_preprocessing_config
from masktab.data.preprocessing import Preprocessor
from masktab.db.connection import get_connection, db_session
from masktab.models.encoder import Encoder
from masktab.training.finetune import finetune_head, train_head_internal

logger = logging.getLogger(__name__)

def run_benchmark(
    encoder_version_id: int,
    csv_path: str,
    db_path: str,
    target_column: str,
    task_type: str,
    feature_columns: list[str] | None = None,
    n_classes: int | None = None,
    labeled_fraction_partial: float = 0.2,
    random_seed: int = 42,
    epochs: int = 100,
    batch_size: int = 32
) -> dict:
    if not (0.0 < labeled_fraction_partial < 1.0):
        raise ValueError(f"labeled_fraction_partial must be in (0.0, 1.0), got {labeled_fraction_partial}")
        
    # Arm (a): Pretrained full
    res_a = finetune_head(
        encoder_version_id=encoder_version_id,
        csv_path=csv_path,
        db_path=db_path,
        target_column=target_column,
        task_type=task_type,
        feature_columns=feature_columns,
        n_classes=n_classes,
        epochs=epochs,
        batch_size=batch_size,
        random_seed=random_seed,
        task_name="benchmark_pretrained_full"
    )
    
    # Reproduce the exact same data split for (b), (c), (d)
    df, inferred_column_types = load_csv(csv_path)
    if feature_columns is not None:
        pass
    else:
        feature_columns = [col for col in df.columns if col != target_column]
        
    df = df.dropna(subset=[target_column]).copy()
    
    n_classes_used = n_classes
    if task_type == "classification":
        distinct_labels = df[target_column].unique()
        n_classes_used = len(distinct_labels)
        label_map = {label: i for i, label in enumerate(sorted(distinct_labels))}
        y_data = df[target_column].map(label_map).values
    else:
        y_data = df[target_column].astype(float).values
        
    dataset_hash = compute_dataset_hash(csv_path, feature_columns)
    conn = get_connection(db_path)
    existing_config = load_preprocessing_config(conn, dataset_hash)
    
    preprocessor = Preprocessor.from_dict(existing_config)
    transformed_array, _ = preprocessor.transform(df)
    
    X_train, X_val, y_train, y_val = train_test_split(
        transformed_array, y_data, test_size=0.15, random_state=random_seed, shuffle=True
    )
    
    # Get encoder architecture from DB to instantiate fresh ones
    cursor = conn.execute("SELECT artifact_path FROM encoder_versions WHERE id = ?", (encoder_version_id,))
    row = cursor.fetchone()
    state_dict = torch.load(row["artifact_path"], weights_only=True)
    input_dim = state_dict['network.0.weight'].shape[1]
    hidden_dims = []
    i = 0
    while f'network.{i}.weight' in state_dict:
        hidden_dims.append(state_dict[f'network.{i}.weight'].shape[0])
        i += 2
        
    # Arm (b): Random-init encoder full
    random_encoder = Encoder(input_dim=input_dim, hidden_dims=tuple(hidden_dims))
    _, val_metric_name_b, val_metric_value_b, _, _ = train_head_internal(
        encoder=random_encoder,
        X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
        task_type=task_type, n_classes=n_classes_used,
        epochs=epochs, batch_size=batch_size, learning_rate=1e-3,
        early_stopping_patience=10, random_seed=random_seed,
        freeze_encoder=True
    )
    
    # Arm (c): Supervised full
    supervised_encoder_full = Encoder(input_dim=input_dim, hidden_dims=tuple(hidden_dims))
    _, val_metric_name_c, val_metric_value_c, _, _ = train_head_internal(
        encoder=supervised_encoder_full,
        X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
        task_type=task_type, n_classes=n_classes_used,
        epochs=epochs, batch_size=batch_size, learning_rate=1e-3,
        early_stopping_patience=10, random_seed=random_seed,
        freeze_encoder=False
    )
    
    # Arm (d): Supervised partial
    np.random.seed(random_seed)
    n_partial = int(len(X_train) * labeled_fraction_partial)
    if n_partial < batch_size:
        raise ValueError(
            f"Subsampled partial dataset has {n_partial} rows, which is fewer than batch_size {batch_size}. "
            f"Suggest larger labeled_fraction_partial or smaller batch_size."
        )
        
    indices = np.random.choice(len(X_train), n_partial, replace=False)
    X_train_partial = X_train[indices]
    y_train_partial = y_train[indices]
    
    supervised_encoder_partial = Encoder(input_dim=input_dim, hidden_dims=tuple(hidden_dims))
    _, val_metric_name_d, val_metric_value_d, _, _ = train_head_internal(
        encoder=supervised_encoder_partial,
        X_train=X_train_partial, X_val=X_val, y_train=y_train_partial, y_val=y_val,
        task_type=task_type, n_classes=n_classes_used,
        epochs=epochs, batch_size=batch_size, learning_rate=1e-3,
        early_stopping_patience=10, random_seed=random_seed,
        freeze_encoder=False
    )
    
    res_a_val = res_a["val_metric_value"]
    
    if task_type == "classification":
        pretrained_within_5pct = res_a_val >= 0.95 * val_metric_value_c
        pretrained_beats_partial = res_a_val > val_metric_value_d
    else:
        pretrained_within_5pct = res_a_val <= 1.05 * val_metric_value_c
        pretrained_beats_partial = res_a_val < val_metric_value_d
        
    results_dict = {
        "pretrained_full": {
            "val_metric_name": res_a["val_metric_name"],
            "val_metric_value": res_a_val,
            "n_training_samples": len(X_train)
        },
        "random_encoder_full": {
            "val_metric_name": val_metric_name_b,
            "val_metric_value": val_metric_value_b,
            "n_training_samples": len(X_train)
        },
        "supervised_full": {
            "val_metric_name": val_metric_name_c,
            "val_metric_value": val_metric_value_c,
            "n_training_samples": len(X_train)
        },
        "supervised_partial": {
            "val_metric_name": val_metric_name_d,
            "val_metric_value": val_metric_value_d,
            "n_training_samples": len(X_train_partial)
        },
        "pretrained_within_5pct_of_supervised": bool(pretrained_within_5pct),
        "pretrained_beats_partial_supervised": bool(pretrained_beats_partial)
    }
    
    with db_session(db_path) as session_conn:
        for arm_name, metrics in results_dict.items():
            if not isinstance(metrics, dict):
                continue
            session_conn.execute(
                """
                INSERT INTO benchmark_results (encoder_version_id, arm_name, val_metric_name, val_metric_value, n_training_samples, labeled_dataset_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (encoder_version_id, arm_name, metrics["val_metric_name"], float(metrics["val_metric_value"]), metrics["n_training_samples"], dataset_hash)
            )
            
    return results_dict
