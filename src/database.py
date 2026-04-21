import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "data" / "scoring.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Tabla de auditoría — cada score calculado queda registrado.
    En entornos financieros regulados es obligatorio mantener
    un registro de todas las decisiones crediticias tomadas.
    """
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS score_history (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            score                 INTEGER NOT NULL,
            category              TEXT NOT NULL,
            decision              TEXT NOT NULL,
            default_prob          REAL NOT NULL,
            age                   INTEGER NOT NULL,
            income_monthly        REAL NOT NULL,
            loan_amount           REAL NOT NULL,
            total_debt            REAL,
            credit_history_years  REAL,
            num_late_payments     INTEGER,
            employment_years      REAL,
            debt_to_income_ratio  REAL,
            explanation           TEXT,
            created_at            DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_score(score_result: dict, input_data: dict, explanation: str) -> int:
    """
    Guarda el resultado completo en el historial.
    Retorna el ID del registro — se incluye en la respuesta al cliente.
    """
    conn = get_connection()
    cur  = conn.execute("""
        INSERT INTO score_history (
            score, category, decision, default_prob,
            age, income_monthly, loan_amount, total_debt,
            credit_history_years, num_late_payments,
            employment_years, debt_to_income_ratio, explanation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        score_result["score"],
        score_result["category"],
        score_result["decision"],
        score_result["default_prob"],
        input_data["age"],
        input_data["income_monthly"],
        input_data["loan_amount"],
        input_data.get("total_debt", 0),
        input_data.get("credit_history_years", 0),
        input_data.get("num_late_payments", 0),
        input_data.get("employment_years", 0),
        score_result["features"].debt_to_income_ratio,
        explanation
    ))
    conn.commit()
    request_id = cur.lastrowid or 0
    conn.close()
    return request_id


def get_history(limit: int = 20) -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM score_history ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_stats() -> dict:
    """Estadísticas generales del sistema de scoring."""
    conn = get_connection()
    row  = conn.execute("""
        SELECT
            COUNT(*)                            AS total_requests,
            AVG(score)                          AS avg_score,
            SUM(CASE WHEN decision = 'Aprobado'    THEN 1 ELSE 0 END) AS approved,
            SUM(CASE WHEN decision = 'Rechazado'   THEN 1 ELSE 0 END) AS rejected,
            SUM(CASE WHEN decision = 'Condicional' THEN 1 ELSE 0 END) AS conditional
        FROM score_history
    """).fetchone()
    conn.close()
    return dict(row) if row else {}