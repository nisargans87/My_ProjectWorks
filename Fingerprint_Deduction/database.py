import sqlite3
from datetime import datetime

DB_PATH = "fingerprint_group.db"


def init_db():
    """Create the database and table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            predicted_group TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def add_detection(name, image_path, predicted_group, confidence):
    """
    Insert a new fingerprint detection record.
    Returns the inserted record ID.
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    current_time = datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
    # Example: 30-07-2026 09:45:18 PM

    cursor.execute("""
        INSERT INTO detections
        (
            name,
            image_path,
            predicted_group,
            confidence,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        name,
        image_path,
        predicted_group,
        confidence,
        current_time
    ))

    conn.commit()

    new_id = cursor.lastrowid

    conn.close()

    return new_id


def get_all_detections():
    """Return all detections ordered by latest first."""

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM detections
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]


def get_detection_by_id(record_id):
    """Return one detection by its ID."""

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM detections
        WHERE id = ?
    """, (record_id,))

    row = cursor.fetchone()

    conn.close()

    if row:
        return dict(row)

    return None


def clear_all_detections():
    """Delete every detection from the database."""

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("DELETE FROM detections")

    conn.commit()

    conn.close()


def delete_detection(record_id):
    """Delete a single detection by ID."""

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM detections WHERE id=?",
        (record_id,)
    )

    conn.commit()

    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized successfully: {DB_PATH}")