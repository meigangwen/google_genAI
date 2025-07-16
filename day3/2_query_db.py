import os
from google import genai
from google.genai import types
import sqlite3
import pandas as pd

# setup a local sql database & populate some data to play with

conn = sqlite3.connect('day3/store/example.db')
cursor = conn.cursor()

def list_tables() -> list[str]:
    """Retrieve the names of all tables in the database."""
    # Include print logging statements so you can see when functions are being called.
    print(' - DB CALL: list_tables()')

    # Fetch the table names.
    # cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [t[0] for t in tables]

print(list_tables())

def describe_table(table_name: str) -> list[tuple[str, str]]:
    """Look up the table schema.

    Returns:
      List of columns, where each entry is a tuple of (column, type).
    """
    print(f' - DB CALL: describe_table({table_name})')

    cursor.execute(f"PRAGMA table_info({table_name});")

    schema = cursor.fetchall()
    # [column index, column name, column type, ...]
    return [(col[1], col[2]) for col in schema]

print(describe_table("products"))

def execute_query(sql: str) -> list[list[str]]:
    """Execute an SQL statement, returning the results."""
    print(f' - DB CALL: execute_query({sql})')

    cursor.execute(sql)
    return cursor.fetchall()

print(execute_query("select * from products"))
print(execute_query("select * from staff"))
print(execute_query("select * from orders"))

# Close connection
conn.close()