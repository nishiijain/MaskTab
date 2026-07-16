import sqlite3
import torch
from typing import Dict, List, Any
import os

class EncoderDimensionMismatchError(Exception):
    pass

def rollback_head(task_name: str, target_head_version_id: int, db_path: str) -> Dict[str, int]:
    """
    Rolls back the active head for a given task to a previously trained version.
    This safely swaps the active metadata flag in a single transaction.
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Validate target_head_version_id belongs to task_name
        cursor.execute(
            "SELECT id, is_active FROM task_head_versions WHERE id = ? AND task_name = ?", 
            (target_head_version_id, task_name)
        )
        target_row = cursor.fetchone()
        
        if not target_row:
            # Check if it exists for another task
            cursor.execute("SELECT task_name FROM task_head_versions WHERE id = ?", (target_head_version_id,))
            wrong_task_row = cursor.fetchone()
            if wrong_task_row:
                raise ValueError(
                    f"Head version {target_head_version_id} exists but belongs to task '{wrong_task_row['task_name']}', "
                    f"not '{task_name}'."
                )
            raise ValueError(f"Head version {target_head_version_id} not found for task '{task_name}'.")

        # Get current active
        cursor.execute("SELECT id FROM task_head_versions WHERE task_name = ? AND is_active = 1", (task_name,))
        current_active_rows = cursor.fetchall()
        
        # If target is already active and no other active rows, it's a no-op
        if target_row["is_active"] == 1 and len(current_active_rows) == 1:
            return {
                "previous_active_id": target_head_version_id,
                "new_active_id": target_head_version_id
            }
            
        previous_active_id = current_active_rows[0]["id"] if current_active_rows else None
        
        # Deactivate all
        cursor.execute("UPDATE task_head_versions SET is_active = 0 WHERE task_name = ?", (task_name,))
        # Activate target
        cursor.execute("UPDATE task_head_versions SET is_active = 1 WHERE id = ?", (target_head_version_id,))
        
        conn.commit()
        
        return {
            "previous_active_id": previous_active_id,
            "new_active_id": target_head_version_id
        }


def rollback_encoder(task_name: str, target_encoder_version_id: int, db_path: str) -> Dict[str, Any]:
    """
    Re-points the active head for a task to a different encoder version by creating a NEW task_head_versions row.
    This does NOT retrain the head against the new encoder's representations. 
    It only repoints metadata. If the new encoder's representations are meaningfully different, 
    a real finetune_head() call is required. This is for reverting to a previously-validated pairing.
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Validate target_encoder_version_id exists
        cursor.execute("SELECT artifact_path FROM encoder_versions WHERE id = ?", (target_encoder_version_id,))
        target_encoder_row = cursor.fetchone()
        if not target_encoder_row:
            raise ValueError(f"Encoder version {target_encoder_version_id} not found in database.")
            
        target_encoder_path = target_encoder_row["artifact_path"]
        if not os.path.exists(target_encoder_path):
            raise FileNotFoundError(f"Target encoder artifact missing at {target_encoder_path}")

        # Get active head
        cursor.execute("""
            SELECT id, encoder_version_id, artifact_path, labeled_dataset_hash, n_samples, val_metric_name, val_metric_value
            FROM task_head_versions 
            WHERE task_name = ? AND is_active = 1
        """, (task_name,))
        active_head_row = cursor.fetchone()
        
        if not active_head_row:
            raise ValueError(f"No active head found for task '{task_name}'. Cannot rollback encoder without an active head.")
            
        current_encoder_id = active_head_row["encoder_version_id"]
        if current_encoder_id == target_encoder_version_id:
            # Already active
            return {
                "previous_head_id": active_head_row["id"],
                "new_head_id": active_head_row["id"],
                "target_encoder_version_id": target_encoder_version_id
            }

        # Check dimension match
        cursor.execute("SELECT artifact_path FROM encoder_versions WHERE id = ?", (current_encoder_id,))
        current_encoder_row = cursor.fetchone()
        if not current_encoder_row or not os.path.exists(current_encoder_row["artifact_path"]):
            raise FileNotFoundError(f"Current encoder artifact missing for ID {current_encoder_id}")
            
        # Load encoders and check output dim
        old_encoder_state = torch.load(current_encoder_row["artifact_path"], map_location="cpu", weights_only=True)
        new_encoder_state = torch.load(target_encoder_path, map_location="cpu", weights_only=True)
        
        old_hidden_dims = old_encoder_state["config"]["hidden_dims"]
        new_hidden_dims = new_encoder_state["config"]["hidden_dims"]
        
        if old_hidden_dims[-1] != new_hidden_dims[-1]:
            raise EncoderDimensionMismatchError(
                f"Encoder output dimension mismatch: old encoder has {old_hidden_dims[-1]}, "
                f"new encoder has {new_hidden_dims[-1]}. Re-pointing would produce invalid predictions."
            )

        # Deactivate current head
        cursor.execute("UPDATE task_head_versions SET is_active = 0 WHERE id = ?", (active_head_row["id"],))
        
        # Insert new head row pointing to the target encoder but keeping the same head weights
        cursor.execute("""
            INSERT INTO task_head_versions (
                encoder_version_id, task_name, artifact_path, 
                labeled_dataset_hash, n_samples, val_metric_name, val_metric_value, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            target_encoder_version_id,
            task_name,
            active_head_row["artifact_path"],
            active_head_row["labeled_dataset_hash"],
            active_head_row["n_samples"],
            active_head_row["val_metric_name"],
            active_head_row["val_metric_value"]
        ))
        
        new_head_id = cursor.lastrowid
        conn.commit()
        
        return {
            "previous_head_id": active_head_row["id"],
            "new_head_id": new_head_id,
            "target_encoder_version_id": target_encoder_version_id
        }


def list_versions(task_name: str, db_path: str) -> Dict[str, Any]:
    """
    Returns full version history for a task (all head versions with timestamps, metrics, active status)
    and its linked encoder versions, for operator visibility.
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                h.id as head_id, 
                h.is_active,
                h.created_at as head_created_at,
                h.val_metric_name,
                h.val_metric_value,
                h.encoder_version_id,
                e.created_at as encoder_created_at,
                e.val_reconstruction_loss
            FROM task_head_versions h
            JOIN encoder_versions e ON h.encoder_version_id = e.id
            WHERE h.task_name = ?
            ORDER BY h.id DESC
        """, (task_name,))
        
        rows = cursor.fetchall()
        
        versions = []
        for r in rows:
            versions.append({
                "head_id": r["head_id"],
                "is_active": bool(r["is_active"]),
                "head_created_at": r["head_created_at"],
                "val_metric_name": r["val_metric_name"],
                "val_metric_value": r["val_metric_value"],
                "encoder_version_id": r["encoder_version_id"],
                "encoder_created_at": r["encoder_created_at"],
                "encoder_val_reconstruction_loss": r["val_reconstruction_loss"],
            })
            
        return {
            "task_name": task_name,
            "history": versions
        }
