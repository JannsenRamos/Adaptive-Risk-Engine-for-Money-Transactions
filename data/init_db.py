import sqlite3
import os

def initialize_database(db_path="data/ledger.db"):
    """
    MLE Role: Initializes the persistent state layer for the SAVE engine.
    Ensures tables and indexes exist for sub-millisecond sequential queries.
    """
    # Ensure the data directory exists before creating the DB
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Stores the raw sequence of user actions to calculate entropy.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    ''')

    # MLE Best Practice: Create a composite index for extremely fast retrieval.
    # We will constantly query "Get all events for User X sorted by Time".
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_time 
        ON session_logs(user_id, timestamp)
    ''')

    # --- FR3: The Velocity Circuit Breaker ---
    # Stores the exact state of the Token Bucket for rate limiting.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS velocity_state (
            user_id TEXT PRIMARY KEY,
            current_tokens REAL NOT NULL,
            last_update_time REAL NOT NULL
        )
    ''')

    conn.commit()
    conn.close()
    print(f" SAVE Database initialized successfully at {db_path}")

if __name__ == "__main__":
    initialize_database()