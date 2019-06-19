import sqlite3

def create_db():
    db     = sqlite3.connect('mos.db')
    cursor = db.cursor()
    return db, cursor

def create_table(db, cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS study(
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            file_name TEXT,
            model TEXT,
            rate INTEGER
        )
    ''')
    db.commit()

def add_entry(db, cursor, data):
    cursor.execute('''
        INSERT INTO study(session_id, file_name, model, rate) VALUES(?, ?, ?, ?)
    ''', (data['session_id'], data['file_name'], data['model'], data['rate']))
    db.commit()
