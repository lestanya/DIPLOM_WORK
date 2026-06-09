import csv
import io
import sqlite3
from collections import Counter
from pathlib import Path

DB_PATH = 'D:/JKH_Diplom/jkh.db'



def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            address TEXT,
            category TEXT,
            emotion TEXT,
            urgency TEXT,
            name TEXT,
            phone TEXT,
            status TEXT NOT NULL DEFAULT 'новая'
        )
    """)
    conn.commit()
    conn.close()


def get_all_requests(sort_by="timestamp", order="desc"):
    allowed_sort = {"timestamp", "urgency", "status"}
    if sort_by not in allowed_sort:
        sort_by = "timestamp"

    order = "ASC" if str(order).lower() == "asc" else "DESC"

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM complaints ORDER BY {sort_by} {order}")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_request_by_id(request_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM complaints WHERE id = ?", (request_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_request(request_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM complaints WHERE id = ?", (request_id,))
    conn.commit()
    conn.close()


def update_request_status(request_id, new_status):
    if new_status not in {"новая", "в_работе", "решена"}:
        return
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE complaints SET status = ? WHERE id = ?", (new_status, request_id))
    conn.commit()
    conn.close()


def update_request(request_id, data):
    allowed_fields = {"text", "address", "category", "emotion", "urgency", "name", "phone", "status"}
    fields = []
    values = []

    for key, value in data.items():
        if key in allowed_fields and value is not None:
            fields.append(f"{key} = ?")
            values.append(value)

    if not fields:
        return

    values.append(request_id)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"UPDATE complaints SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()
    conn.close()


def get_statistics():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT category, urgency, status, emotion FROM complaints")
    rows = cur.fetchall()
    conn.close()

    categories = Counter()
    urgencies = Counter()
    statuses = Counter()
    emotions = Counter()

    for row in rows:
        categories[row["category"] or "не определена"] += 1
        urgencies[row["urgency"] or "не определена"] += 1
        statuses[row["status"] or "не определена"] += 1
        emotions[row["emotion"] or "не определена"] += 1

    return {
        "categories": categories,
        "urgencies": urgencies,
        "statuses": statuses,
        "emotions": emotions,
    }


def export_to_csv_bytes():
    rows = get_all_requests(sort_by="timestamp", order="asc")
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["id", "timestamp", "user_id", "text", "address", "category", "emotion", "urgency", "name", "phone", "status"])

    for row in rows:
        writer.writerow([
            row.get("id"),
            row.get("timestamp"),
            row.get("user_id"),
            row.get("text"),
            row.get("address"),
            row.get("category"),
            row.get("emotion"),
            row.get("urgency"),
            row.get("name"),
            row.get("phone"),
            row.get("status"),
        ])

    return output.getvalue()