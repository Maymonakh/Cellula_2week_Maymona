import sqlite3
from datetime import datetime


# Create database
def create_database():

    connection = sqlite3.connect("database.db")

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            type TEXT,
            input TEXT,
            classification TEXT
        )
    """)

    connection.commit()
    connection.close()


# Save result
def save_result(input_type, input_text, result):

    connection = sqlite3.connect("database.db")

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO classifications
        (timestamp, type, input, classification)
        VALUES (?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        input_type,
        input_text,
        result
    ))

    connection.commit()
    connection.close()


# Get all results
def get_results():

    connection = sqlite3.connect("database.db")

    cursor = connection.cursor()

    cursor.execute("""
        SELECT timestamp, type, input, classification
        FROM classifications
        ORDER BY id DESC
    """)

    results = cursor.fetchall()

    connection.close()

    return results