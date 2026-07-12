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
