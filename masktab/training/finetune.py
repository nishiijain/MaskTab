import os
import math
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error

from masktab.data.loader import load_csv
from masktab.data.hashing import compute_dataset_hash
from masktab.data.registry import load_preprocessing_config, save_preprocessing_config
from masktab.data.preprocessing import Preprocessor
from masktab.db.connection import get_connection, db_session
from masktab.models.encoder import Encoder
from masktab.models.task_head import TaskHead

logger = logging.getLogger(__name__)

class EncoderVersionNotFoundError(Exception):
    pass

class FeatureDimensionMismatchError(Exception):
    pass

def train_head_internal(
    encoder: nn.Module,
    X_train, X_val, y_train, y_val,
    task_type: str, n_classes: int | None,
    epochs: int, batch_size: int, learning_rate: float,
    early_stopping_patience: int, random_seed: int,
    freeze_encoder: bool = True
):
    torch.manual_seed(random_seed)
    
    if freeze_encoder:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad_(False)
    else:
        encoder.train()
        for param in encoder.parameters():
            param.requires_grad_(True)
            
    encoder_output_dim = encoder.get_output_dim()
    head = TaskHead(input_dim=encoder_output_dim, task_type=task_type, n_classes=n_classes)
    
    if freeze_encoder:
        optimizer = optim.Adam(head.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=learning_rate)
        
    loss_fn = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopped = False
    epochs_run = 0
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
    if task_type == "classification":
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    else:
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        
    num_batches = len(X_train) // batch_size
    val_num_batches = max(1, len(X_val) // batch_size)
    val_batch_size = len(X_val) if len(X_val) < batch_size else batch_size
    
    for epoch in range(epochs):
        if freeze_encoder:
            encoder.eval()
        else:
            encoder.train()
        head.train()
        
        epoch_train_loss = 0.0
        
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            x_b = X_train_tensor[batch_start:batch_end]
            y_b = y_train_tensor[batch_start:batch_end]
            
            optimizer.zero_grad()
            
            if freeze_encoder:
                with torch.no_grad():
                    embeddings = encoder(x_b)
            else:
                embeddings = encoder(x_b)
                
            preds = head(embeddings)
            loss = loss_fn(preds, y_b)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        encoder.eval()
        head.eval()
        epoch_val_loss = 0.0
        
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for i in range(val_num_batches):
                batch_start = i * val_batch_size
                batch_end = (i + 1) * val_batch_size if i < val_num_batches - 1 else len(X_val)
                x_val_b = X_val_tensor[batch_start:batch_end]
                y_val_b = y_val_tensor[batch_start:batch_end]
                
                embeddings = encoder(x_val_b)
                preds = head(embeddings)
                v_loss = loss_fn(preds, y_val_b)
                epoch_val_loss += v_loss.item()
                
                if task_type == "classification":
                    all_val_preds.append(preds.softmax(dim=-1))
                else:
                    all_val_preds.append(preds)
                all_val_targets.append(y_val_b)
                
        avg_val_loss = epoch_val_loss / val_num_batches
        epochs_run += 1
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            early_stopped = True
            break
            
    # Calculate final metrics
    encoder.eval()
    head.eval()
    all_preds_tensor = torch.cat(all_val_preds, dim=0).numpy()
    all_targets_tensor = torch.cat(all_val_targets, dim=0).numpy()
    
    val_metric_name = ""
    val_metric_value = 0.0
    
    if task_type == "classification":
        pred_classes = all_preds_tensor.argmax(axis=1)
        f1 = f1_score(all_targets_tensor, pred_classes, average="macro")
        try:
            if n_classes == 2:
                # binary AUC
                auc = roc_auc_score(all_targets_tensor, all_preds_tensor[:, 1])
            else:
                auc = roc_auc_score(all_targets_tensor, all_preds_tensor, average="macro", multi_class="ovr")
        except ValueError as e:
            logger.warning(f"AUC calculation failed: {e}")
            auc = None
        val_metric_name = "F1" # primary metric
        val_metric_value = f1
    else:
        mse = mean_squared_error(all_targets_tensor, all_preds_tensor)
        rmse = math.sqrt(mse)
        val_metric_name = "RMSE"
        val_metric_value = rmse
        
    return head, val_metric_name, val_metric_value, epochs_run, early_stopped


def finetune_head(
    encoder_version_id: int,
    csv_path: str,
    db_path: str,
    target_column: str,
    task_type: str,
    feature_columns: list[str] | None = None,
    n_classes: int | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_split: float = 0.15,
    artifact_dir: str = "artifacts/heads",
    random_seed: int = 42,
    early_stopping_patience: int = 10,
    task_name: str = None
) -> dict:
    if not task_name:
        raise ValueError("task_name must be provided and cannot be empty")
        
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type}")

    # 1. DB Query and loading encoder
    conn = get_connection(db_path)
    cursor = conn.execute("SELECT artifact_path FROM encoder_versions WHERE id = ?", (encoder_version_id,))
    row = cursor.fetchone()
    if not row:
        raise EncoderVersionNotFoundError(f"Encoder version {encoder_version_id} not found in database")
        
    encoder_artifact_path = row["artifact_path"]
    if not os.path.exists(encoder_artifact_path):
        raise FileNotFoundError(f"Encoder artifact missing on disk: {encoder_artifact_path}")
        
    state_dict = torch.load(encoder_artifact_path, weights_only=True)
    input_dim = state_dict['network.0.weight'].shape[1]
    
    hidden_dims = []
    i = 0
    while f'network.{i}.weight' in state_dict:
        hidden_dims.append(state_dict[f'network.{i}.weight'].shape[0])
        i += 2
        
    encoder = Encoder(input_dim=input_dim, hidden_dims=tuple(hidden_dims))
    encoder.load_state_dict(state_dict)
    
    # 2. Load CSV and validate target
    df, inferred_column_types = load_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in CSV columns")
        
    if feature_columns is not None:
        if target_column in feature_columns:
            raise ValueError(f"target_column '{target_column}' cannot be in feature_columns")
    else:
        feature_columns = [col for col in df.columns if col != target_column]
        
    # 3. Clean target column
    orig_len = len(df)
    df = df.dropna(subset=[target_column]).copy()
    dropped = orig_len - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with null target values")
        if dropped > 0.5 * orig_len:
            raise ValueError("Dropped more than 50% of rows due to missing target values")
            
    if task_type == "classification":
        distinct_labels = df[target_column].unique()
        if len(distinct_labels) < 2:
            raise ValueError(f"Found {len(distinct_labels)} distinct classes, need at least 2 for classification")
        if n_classes is not None and len(distinct_labels) != n_classes:
            raise ValueError(f"n_classes provided as {n_classes} but found {len(distinct_labels)} distinct classes")
        n_classes = len(distinct_labels)
        
        label_map = {label: i for i, label in enumerate(sorted(distinct_labels))}
        y_data = df[target_column].map(label_map).values
    else:
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            raise ValueError(f"Target column '{target_column}' must be numeric for regression")
        y_data = df[target_column].astype(float).values

    # 4. Preprocessing
    dataset_hash = compute_dataset_hash(csv_path, feature_columns)
    existing_config = load_preprocessing_config(conn, dataset_hash)
    
    preprocessor = Preprocessor()
    if existing_config:
        logger.info(f"Reusing existing preprocessing config for dataset_hash={dataset_hash}")
        preprocessor = Preprocessor.from_dict(existing_config)
    else:
        logger.info(f"Fitting new preprocessing config for dataset_hash={dataset_hash}")
        preprocessor.fit(df, inferred_column_types, feature_columns)
        with db_session(db_path) as session_conn:
            for col, config in preprocessor.to_dict().items():
                save_preprocessing_config(session_conn, dataset_hash, col, config)
                
    transformed_array, _ = preprocessor.transform(df)
    
    # 5. Dimension check
    if transformed_array.shape[1] != input_dim:
        raise FeatureDimensionMismatchError(
            f"Preprocessed feature dimension {transformed_array.shape[1]} does not match encoder input_dim {input_dim}"
        )
        
    # 6. Train/Val split
    if not (0.0 < val_split < 0.5):
        raise ValueError("val_split must be between 0.0 and 0.5")
        
    X_train, X_val, y_train, y_val = train_test_split(
        transformed_array, y_data, test_size=val_split, random_state=random_seed, shuffle=True
    )
    
    if len(X_train) < batch_size:
        raise ValueError(f"Training set has {len(X_train)} rows, fewer than batch_size {batch_size}")
        
    # 7. Train
    head, val_metric_name, val_metric_value, epochs_run, early_stopped = train_head_internal(
        encoder=encoder,
        X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
        task_type=task_type, n_classes=n_classes,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience, random_seed=random_seed,
        freeze_encoder=True
    )
    
    # 8. Save artifact and update DB atomically
    os.makedirs(artifact_dir, exist_ok=True)
    
    with db_session(db_path) as session_conn:
        cursor = session_conn.execute("SELECT MAX(id) as max_id FROM task_head_versions")
        row = cursor.fetchone()
        next_version = (row["max_id"] or 0) + 1
        
        artifact_path = os.path.join(artifact_dir, f"{task_name}_head_v{next_version}.pt")
        
        try:
            torch.save(head.state_dict(), artifact_path)
            
            # Deactivate previous
            session_conn.execute(
                "UPDATE task_head_versions SET is_active = 0 WHERE task_name = ? AND is_active = 1",
                (task_name,)
            )
            # Insert new
            session_conn.execute(
                """
                INSERT INTO task_head_versions (encoder_version_id, task_name, artifact_path, labeled_dataset_hash, n_samples, val_metric_name, val_metric_value, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (encoder_version_id, task_name, artifact_path, dataset_hash, len(df), val_metric_name, val_metric_value)
            )
        except Exception as e:
            if os.path.exists(artifact_path):
                os.remove(artifact_path)
            raise e
            
    return {
        "head_version_id": next_version,
        "task_name": task_name,
        "encoder_version_id": encoder_version_id,
        "artifact_path": artifact_path,
        "val_metric_name": val_metric_name,
        "val_metric_value": val_metric_value,
        "epochs_run": epochs_run,
        "early_stopped": early_stopped
    }
