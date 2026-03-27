import sqlite3
from pathlib import Path


def initialize_sqlite(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                owner TEXT NOT NULL,
                status TEXT NOT NULL,
                due_date TEXT NOT NULL
            );
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        seed_rows = [
            (1, "Prepare quarterly report", "Alice", "in_progress", "2026-04-05"),
            (2, "Security review for payment service", "Bob", "todo", "2026-04-10"),
            (3, "Migrate API gateway to v2", "Carol", "done", "2026-03-20"),
        ]

        connection.executemany(
            """
            INSERT OR IGNORE INTO tasks (id, name, owner, status, due_date)
            VALUES (?, ?, ?, ?, ?);
            """,
            seed_rows,
        )
        connection.commit()


if __name__ == "__main__":
    initialize_sqlite(Path(__file__).resolve().parent / "app.db")
    print("SQLite initialized")
