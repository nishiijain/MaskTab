import json
import sqlite3

def save_preprocessing_config(conn: sqlite3.Connection, dataset_hash: str, feature_name: str, config: dict) -> None:
    # Serialize config to json
    # Determine type of config
    scaler_type = None
    scaler_params_json = None
    encoding_map_json = None
    imputation_value = json.dumps(config.get("imputation_value"))
    
    if config.get("type") == "numerical":
        scaler_type = "StandardScaler"
        scaler_params = {
            "mean": config.get("scaler_mean"),
            "scale": config.get("scaler_scale")
        }
        scaler_params_json = json.dumps(scaler_params)
    elif config.get("type") == "categorical":
        encoding_strategy = config.get("encoding_strategy")
        if encoding_strategy == "embedding":
            encoding_map = config.get("encoding_map")
            encoding_map_json = json.dumps({"strategy": "embedding", "map": encoding_map})
        elif encoding_strategy == "onehot":
            categories = config.get("categories")
            encoding_map_json = json.dumps({"strategy": "onehot", "categories": categories})

    query = """
    INSERT OR REPLACE INTO preprocessing_configs 
    (dataset_hash, feature_name, scaler_type, scaler_params_json, encoding_map_json, imputation_value) 
    VALUES (?, ?, ?, ?, ?, ?)
    """
    conn.execute(query, (dataset_hash, feature_name, scaler_type, scaler_params_json, encoding_map_json, imputation_value))

def load_preprocessing_config(conn: sqlite3.Connection, dataset_hash: str) -> dict | None:
    query = """
    SELECT feature_name, scaler_type, scaler_params_json, encoding_map_json, imputation_value 
    FROM preprocessing_configs 
    WHERE dataset_hash = ?
    """
    cursor = conn.execute(query, (dataset_hash,))
    rows = cursor.fetchall()
    
    if not rows:
        return None
        
    config = {}
    for row in rows:
        feature_name = row["feature_name"]
        feature_config = {}
        
        imputation_value = json.loads(row["imputation_value"]) if row["imputation_value"] else None
        
        if row["scaler_type"] == "StandardScaler":
            feature_config["type"] = "numerical"
            feature_config["imputation_value"] = imputation_value
            scaler_params = json.loads(row["scaler_params_json"])
            feature_config["scaler_mean"] = scaler_params["mean"]
            feature_config["scaler_scale"] = scaler_params["scale"]
        else:
            feature_config["type"] = "categorical"
            feature_config["imputation_value"] = imputation_value
            encoding_map = json.loads(row["encoding_map_json"])
            if encoding_map["strategy"] == "embedding":
                feature_config["encoding_strategy"] = "embedding"
                feature_config["encoding_map"] = encoding_map["map"]
            elif encoding_map["strategy"] == "onehot":
                feature_config["encoding_strategy"] = "onehot"
                feature_config["categories"] = encoding_map["categories"]
                
        config[feature_name] = feature_config
        
    return config
