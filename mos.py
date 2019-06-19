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

def get_results(db, cursor):
    cursor.execute('SELECT model, ROUND(AVG(rate), 2) FROM study GROUP BY model')
    _means  = cursor.fetchall()
    means   = { row[0]: row[1]  for row in _means }

    cursor.execute('''
        SELECT
            model, SUM(
                (
                    (SELECT rate FROM study GROUP BY model) -
                    (SELECT AVG(rate) FROM study GROUP BY model)
                ) *
                (
                    (SELECT rate FROM study GROUP BY model) -
                    (SELECT AVG(rate) FROM study GROUP BY model)
                )
            ) / (COUNT((SELECT rate FROM study GROUP BY model)) - 1)
        FROM
            study
        GROUP BY
            model
    ''')
    _vars  = cursor.fetchall()
    vars   = { row[0]: row[1]  for row in _vars }

    cursor.execute('SELECT COUNT(rate) FROM study')
    counts = cursor.fetchall()
    counts = row[0][0]

    results = {
        'paper'    : {
            'mean': means['paper'],
            'std' : vars['paper'],
        },
        'our'      : {
            'mean': means['out'],
            'std' : vars['out'],
        },
        'out_final': {
            'mean': means['out_final'],
            'std' : vars['out_final'],
        },
        'count': counts
    }

    return results
