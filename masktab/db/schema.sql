CREATE TABLE IF NOT EXISTS preprocessing_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_hash TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    scaler_type TEXT,
    scaler_params_json TEXT,
    encoding_map_json TEXT,
    imputation_value TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_hash, feature_name)
);

CREATE INDEX IF NOT EXISTS idx_dataset_hash ON preprocessing_configs(dataset_hash);

CREATE TABLE IF NOT EXISTS encoder_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_path TEXT NOT NULL,
    pretraining_dataset_hash TEXT NOT NULL,
    n_samples INTEGER NOT NULL,
    n_epochs INTEGER NOT NULL,
    val_reconstruction_loss REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_encoder_versions_hash ON encoder_versions(pretraining_dataset_hash);

CREATE TABLE IF NOT EXISTS task_head_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    encoder_version_id INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    labeled_dataset_hash TEXT NOT NULL,
    n_samples INTEGER NOT NULL,
    val_metric_name TEXT NOT NULL,
    val_metric_value REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (encoder_version_id) REFERENCES encoder_versions(id)
);

CREATE INDEX IF NOT EXISTS idx_task_head_active ON task_head_versions(task_name, is_active);

CREATE TABLE IF NOT EXISTS benchmark_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    encoder_version_id INTEGER,
    arm_name TEXT NOT NULL,
    val_metric_name TEXT NOT NULL,
    val_metric_value REAL NOT NULL,
    n_training_samples INTEGER NOT NULL,
    labeled_dataset_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
