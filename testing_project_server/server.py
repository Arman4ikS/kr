from flask import Flask, request, jsonify
import sqlite3
import numpy as np
import json
from model import processing



def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS useless (
                        id INTEGER PRIMARY KEY AUTOINCREMENT
                   )
                   ''')
    conn.commit()
    conn.close()


def save_data(json_data):
    required_fields = {
        'unit_ID', 'cycles', 'setting_1', 'setting_2', 'setting_3',
        'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc',
        'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed',
        'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32'
    }

    conn = None
    try:
        if not isinstance(json_data, dict):
            raise ValueError("Неверный формат данных. Ожидается словарь")

        missing_fields = required_fields - set(json_data.keys())
        if missing_fields:
            raise ValueError(f"Отсутствуют обязательные поля: {missing_fields}")

        unit_ID = json_data['unit_ID']

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()

        cursor.execute(f'''CREATE TABLE IF NOT EXISTS unit_ID_{unit_ID}
                          (
                              cycles    INTEGER PRIMARY KEY,
                              setting_1 REAL,
                              setting_2 REAL,
                              setting_3 REAL,
                              T2        REAL,
                              T24       REAL,
                              T30       REAL,
                              T50       REAL,
                              P2        REAL,
                              P15       REAL,
                              P30       REAL,
                              Nf        REAL,
                              Nc        REAL,
                              epr       REAL,
                              Ps30      REAL,
                              phi       REAL,
                              NRf       REAL,
                              NRc       REAL,
                              BPR       REAL,
                              farB      REAL,
                              htBleed   REAL,
                              Nf_dmd    REAL,
                              PCNfR_dmd REAL,
                              W31       REAL,
                              W32       REAL
                          )''')
        json_data.pop('unit_ID')

        columns = ', '.join(json_data.keys())
        placeholders = ', '.join(['?'] * len(json_data))
        values = tuple(json_data.values())

        cursor.execute(
            f"INSERT INTO unit_ID_{unit_ID} ({columns}) VALUES ({placeholders})",
            values
        )

        conn.commit()
        print("Данные успешно сохранены!")
        return True

    except ValueError as ve:
        print(f"Ошибка валидации данных: {ve}")
        return False

    except sqlite3.Error as e:
        print(f"Ошибка базы данных: {e}")
        if conn:
            conn.rollback()
        return False

    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        return False

    finally:
        if conn:
            conn.close()

app = Flask(__name__)

@app.route('/data', methods=['POST'])
def handle_data():
    data = request.json
    print("Сервер получил:", data)
    unit_ID = data.get("unit_ID")
    save_data(data)
    rul = processing(unit_ID)
    return jsonify({"status": "OK", "answer": rul})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)  # Только локальные подключения
    init_db()