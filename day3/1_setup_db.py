import os
from google import genai
from google.genai import types
import sqlite3
import pandas as pd

# setup a local sql database & populate some data to play with

conn = sqlite3.connect('day3/store/example.db')
cursor = conn.cursor()

# Create the 'products' table
cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
  	product_id INTEGER PRIMARY KEY AUTOINCREMENT,
  	product_name VARCHAR(255) NOT NULL,
  	price DECIMAL(10, 2) NOT NULL
  );
''')

# Create the 'staff' table
cursor.execute('''
CREATE TABLE IF NOT EXISTS staff (
  	staff_id INTEGER PRIMARY KEY AUTOINCREMENT,
  	first_name VARCHAR(255) NOT NULL,
  	last_name VARCHAR(255) NOT NULL
  );
''')

# Create the 'orders' table
cursor.execute('''
CREATE TABLE IF NOT EXISTS orders (
  	order_id INTEGER PRIMARY KEY AUTOINCREMENT,
  	customer_name VARCHAR(255) NOT NULL,
  	staff_id INTEGER NOT NULL,
  	product_id INTEGER NOT NULL,
  	FOREIGN KEY (staff_id) REFERENCES staff (staff_id),
  	FOREIGN KEY (product_id) REFERENCES products (product_id)
  );
''')

# Insert data into the 'products' table
products = [
    ('Laptop', 799.99),
  	('Keyboard', 129.99),
  	('Mouse', 29.99)
]

cursor.executemany("INSERT INTO products (product_name, price) VALUES (?, ?)", products)
conn.commit()

# Insert data into the 'staff' table
staff = [
	('Alice', 'Smith'),
  	('Bob', 'Johnson'),
  	('Charlie', 'Williams')
]

cursor.executemany("INSERT INTO staff (first_name, last_name) VALUES (?, ?)", staff)
conn.commit()

# Insert data into the 'orders' table
orders = [
	('David Lee', 1, 1),
  	('Emily Chen', 2, 2),
	('Frank Brown', 1, 3)
]

cursor.executemany("INSERT INTO orders (customer_name, staff_id, product_id) VALUES (?, ?, ?)", orders)
conn.commit()

print('Database setup finished')

# Close connection
conn.close()