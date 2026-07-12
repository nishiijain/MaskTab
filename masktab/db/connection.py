import sqlite3
import contextlib
import os

def get_connection(db_path: str = "masktab.db") -> sqlite3.Connection:
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    
    # Check if table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='preprocessing_configs'")
    if not cursor.fetchone():
        with open(schema_path, "r") as f:
            schema_script = f.read()
        cursor.executescript(schema_script)
    
    return conn

@contextlib.contextmanager
def db_session(db_path: str = "masktab.db"):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
