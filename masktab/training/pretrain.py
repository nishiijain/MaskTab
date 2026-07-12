import os
import logging
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

from masktab.data.loader import load_csv, EmptyDatasetError
from masktab.data.hashing import compute_dataset_hash
from masktab.data.registry import load_preprocessing_config, save_preprocessing_config
from masktab.data.preprocessing import Preprocessor
from masktab.db.connection import get_connection, db_session
from masktab.models.encoder import Encoder
from masktab.models.decoder import Decoder
from masktab.models.masking import generate_feature_mask, apply_mask
from masktab.models.losses import masked_reconstruction_loss

logger = logging.getLogger(__name__)

def pretrain_encoder(
    csv_path: str,
    db_path: str,
    feature_columns: list[str] | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    mask_prob: float = 0.3,
    alpha: float = 0.5,
    val_split: float = 0.15,
    artifact_dir: str = "artifacts/encoders",
    random_seed: int = 42,
    early_stopping_patience: int = 8
) -> dict:
    # 1. Load CSV
    df, inferred_column_types = load_csv(csv_path)

    # 2. Validate feature_columns
    if feature_columns is None:
        feature_columns = list(df.columns)
    else:
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"feature_columns contains missing columns: {missing}")

    # 3. Compute hash
    dataset_hash = compute_dataset_hash(csv_path, feature_columns)

    # 4. Check registry
    conn = get_connection(db_path)
    existing_config = load_preprocessing_config(conn, dataset_hash)
    
    preprocessor = Preprocessor()
    if existing_config:
        logger.info(f"Reusing existing preprocessing config for dataset_hash={dataset_hash}")
        preprocessor = Preprocessor.from_dict(existing_config)
    else:
        logger.info(f"Fitting new preprocessing config for dataset_hash={dataset_hash}")
        preprocessor.fit(df, inferred_column_types, feature_columns)
        # Save to registry
        with db_session(db_path) as session_conn:
            for col, config in preprocessor.to_dict().items():
                save_preprocessing_config(session_conn, dataset_hash, col, config)
                
    # 5. Transform and split
    transformed_array, output_columns = preprocessor.transform(df)
    
    if not (0.0 < val_split < 0.5):
        raise ValueError(f"val_split must be in (0.0, 0.5), got {val_split}")
        
    X_train, X_val = train_test_split(transformed_array, test_size=val_split, random_state=random_seed, shuffle=True)
    
    # 6. Check training set size
    if len(X_train) < batch_size:
        raise ValueError(f"Training set has {len(X_train)} rows, which is fewer than batch_size {batch_size}")
        
    # 7. Seed Torch
    torch.manual_seed(random_seed)
    
    # 8. Instantiate models
    input_dim = transformed_array.shape[1]
    encoder = Encoder(input_dim=input_dim)
    decoder = Decoder(latent_dim=encoder.get_output_dim(), output_dim=input_dim)
    
    # 9. Build feature_types list
    feature_types = []
    for out_col in output_columns:
        # Preprocessor output_columns format: original_col for numerical/embedding, original_col_category for onehot
        # Let's deduce from preprocessor.params
        matched = False
        for col, params in preprocessor.params.items():
            if out_col == col:
                feature_types.append(params["type"])
                matched = True
                break
            elif params["type"] == "categorical" and params.get("encoding_strategy") == "onehot":
                if out_col.startswith(f"{col}_"):
                    # Onehot expanded columns are treated as categorical for loss purposes
                    feature_types.append("categorical")
                    matched = True
                    break
        if not matched:
            raise RuntimeError(f"Could not determine feature type for output column {out_col}")
            
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    
    # 10. Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopped = False
    epochs_run = 0
    final_train_loss = 0.0
    
    try:
        for epoch in range(epochs):
            encoder.train()
            decoder.train()
            
            # Batching: drop last incomplete batch
            num_batches = len(X_train) // batch_size
            epoch_train_loss = 0.0
            
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = (i + 1) * batch_size
                x_batch = torch.tensor(X_train[batch_start:batch_end])
                
                optimizer.zero_grad()
                
                mask = generate_feature_mask(batch_size, input_dim, mask_prob=mask_prob)
                x_masked = apply_mask(x_batch, mask)
                
                latent = encoder(x_masked)
                reconstructed = decoder(latent)
                
                loss, _ = masked_reconstruction_loss(reconstructed, x_batch, mask, feature_types, alpha)
                
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                
            avg_train_loss = epoch_train_loss / num_batches
            final_train_loss = avg_train_loss
            
            # Validation
            encoder.eval()
            decoder.eval()
            
            val_num_batches = len(X_val) // batch_size
            if val_num_batches == 0:
                # If val set is smaller than batch_size, evaluate on all of it
                val_num_batches = 1
                val_batch_size = len(X_val)
            else:
                val_batch_size = batch_size
                
            epoch_val_loss = 0.0
            
            with torch.no_grad():
                # Use a fixed generator for validation set determinism based on epoch
                val_gen = torch.Generator()
                val_gen.manual_seed(random_seed + epoch)
                
                for i in range(val_num_batches):
                    if val_batch_size == len(X_val):
                        x_val_batch = torch.tensor(X_val)
                    else:
                        batch_start = i * val_batch_size
                        batch_end = (i + 1) * val_batch_size
                        x_val_batch = torch.tensor(X_val[batch_start:batch_end])
                        
                    val_mask = generate_feature_mask(len(x_val_batch), input_dim, mask_prob=mask_prob, generator=val_gen)
                    x_val_masked = apply_mask(x_val_batch, val_mask)
                    
                    latent = encoder(x_val_masked)
                    reconstructed = decoder(latent)
                    
                    v_loss, _ = masked_reconstruction_loss(reconstructed, x_val_batch, val_mask, feature_types, alpha)
                    epoch_val_loss += v_loss.item()
                    
            avg_val_loss = epoch_val_loss / val_num_batches
            
            epochs_run += 1
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                early_stopped = True
                break
                
        # 11. Save artifact (only if completed or early stopped without raising an exception)
        os.makedirs(artifact_dir, exist_ok=True)
        
        with db_session(db_path) as session_conn:
            cursor = session_conn.execute("SELECT MAX(id) as max_id FROM encoder_versions")
            row = cursor.fetchone()
            next_version = (row["max_id"] or 0) + 1
            
            artifact_path = os.path.join(artifact_dir, f"encoder_v{next_version}.pt")
            torch.save(encoder.state_dict(), artifact_path)
            
            # 12. Insert DB row
            session_conn.execute(
                """
                INSERT INTO encoder_versions (artifact_path, pretraining_dataset_hash, n_samples, n_epochs, val_reconstruction_loss)
                VALUES (?, ?, ?, ?, ?)
                """,
                (artifact_path, dataset_hash, len(df), epochs_run, best_val_loss)
            )
            
        return {
            "encoder_version_id": next_version,
            "artifact_path": artifact_path,
            "dataset_hash": dataset_hash,
            "epochs_run": epochs_run,
            "best_val_loss": best_val_loss,
            "early_stopped": early_stopped,
            "final_train_loss": final_train_loss
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e
