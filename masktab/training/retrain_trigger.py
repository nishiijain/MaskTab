import logging
from masktab.training.finetune import finetune_head
from masktab.training.pretrain import pretrain_encoder
from masktab.db.connection import get_connection

logger = logging.getLogger(__name__)

def evaluate_and_trigger_retrain(
    task_name: str, 
    db_path: str, 
    csv_path_for_retrain: str, 
    target_column: str, 
    task_type: str, 
    drift_evaluation: dict, 
    n_classes: int | None = None
) -> dict:
    
    recommendation = drift_evaluation.get("retrain_recommendation")
    if recommendation not in ("none", "head_only", "full"):
        raise ValueError(f"Invalid retrain_recommendation: {recommendation}")
        
    conn = get_connection(db_path)
    try:
        # Get current head to log previous head version and get current encoder_version_id
        cursor = conn.execute(
            "SELECT id, encoder_version_id FROM task_head_versions WHERE task_name = ? AND is_active = 1",
            (task_name,)
        )
        row = cursor.fetchone()
        
        prev_head_id = row["id"] if row else None
        current_encoder_id = row["encoder_version_id"] if row else None
        
        if recommendation != "none" and current_encoder_id is None:
            # If we need to retrain but there's no active head, we can't do head_only effectively 
            # without an encoder. Actually, if there is no active head, maybe we just pretrain?
            # The prompt implies we call finetune_head with existing encoder_version_id.
            # If none exists, we will raise an error.
            if recommendation == "head_only":
                raise RuntimeError(f"Cannot perform head_only retrain: no active task head found for {task_name}")

        new_head_id = None
        new_encoder_id = None
        action_taken = "none"
        
        if recommendation == "head_only":
            logger.info(f"Triggering head_only retrain for task {task_name}")
            result = finetune_head(
                encoder_version_id=current_encoder_id,
                csv_path=csv_path_for_retrain,
                db_path=db_path,
                target_column=target_column,
                task_type=task_type,
                n_classes=n_classes,
                task_name=task_name
            )
            new_head_id = result["head_version_id"]
            action_taken = "head_only_retrain"
            
        elif recommendation == "full":
            logger.info(f"Triggering full retrain (encoder + head) for task {task_name}")
            encoder_result = pretrain_encoder(
                csv_path=csv_path_for_retrain,
                db_path=db_path
            )
            new_encoder_id = encoder_result["encoder_version_id"]
            
            result = finetune_head(
                encoder_version_id=new_encoder_id,
                csv_path=csv_path_for_retrain,
                db_path=db_path,
                target_column=target_column,
                task_type=task_type,
                n_classes=n_classes,
                task_name=task_name
            )
            new_head_id = result["head_version_id"]
            action_taken = "full_retrain"

        # Log event
        # Note: we log "none" events intentionally for audit completeness
        conn.execute(
            """
            INSERT INTO fine_tuning_log (task_name, trigger_reason, previous_head_version_id, new_head_version_id)
            VALUES (?, ?, ?, ?)
            """,
            (task_name, recommendation, prev_head_id, new_head_id)
        )
        conn.commit()
        
    finally:
        conn.close()
        
    return {
        "action_taken": action_taken,
        "recommendation": recommendation,
        "previous_head_version_id": prev_head_id,
        "new_head_version_id": new_head_id,
        "new_encoder_version_id": new_encoder_id
    }
