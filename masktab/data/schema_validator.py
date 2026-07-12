import logging

logger = logging.getLogger(__name__)

MIN_SHARED_FEATURES = 2

class SchemaIncompatibleError(Exception):
    pass

def validate_schema_compatibility(pretraining_columns: dict, task_columns: dict) -> dict:
    shared_cols = set(pretraining_columns.keys()).intersection(task_columns.keys())
    
    if not shared_cols:
        raise SchemaIncompatibleError("Intersection of column names is empty")
        
    if len(shared_cols) < MIN_SHARED_FEATURES:
        raise SchemaIncompatibleError(f"Intersection size {len(shared_cols)} is less than MIN_SHARED_FEATURES ({MIN_SHARED_FEATURES})")

    conflicts = []
    shared_column_types = {}
    for col in shared_cols:
        pt_type = pretraining_columns[col]
        t_type = task_columns[col]
        if pt_type != t_type:
            conflicts.append(f"{col} (pretraining: {pt_type}, task: {t_type})")
        else:
            shared_column_types[col] = pt_type

    if conflicts:
        raise SchemaIncompatibleError(f"Type conflicts on shared columns: {', '.join(conflicts)}")

    pretraining_only = list(set(pretraining_columns.keys()) - shared_cols)
    task_only = list(set(task_columns.keys()) - shared_cols)
    
    if pretraining_only or task_only:
        logger.warning(f"Non-shared columns will be dropped downstream. Pretraining only: {pretraining_only}, Task only: {task_only}")

    return {
        "shared_columns": list(shared_cols),
        "pretraining_only_columns": pretraining_only,
        "task_only_columns": task_only,
        "shared_column_types": shared_column_types
    }
