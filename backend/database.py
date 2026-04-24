import sqlite3
import json
import logging
from contextlib import contextmanager
from datetime import datetime

logger  = logging.getLogger(__name__)
DB_PATH = "financial.db"
KEEP_N  = 5  # max rows to retain per user in analysis_results / predictions / behavior_profiles


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id        TEXT PRIMARY KEY,
    name           TEXT NOT NULL DEFAULT 'User',
    email          TEXT,
    monthly_budget REAL NOT NULL DEFAULT 50000.0,
    currency       TEXT NOT NULL DEFAULT 'INR',
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id      TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    date        TEXT NOT NULL,
    month       TEXT NOT NULL,
    description TEXT NOT NULL,
    amount      REAL NOT NULL,
    type        TEXT NOT NULL CHECK(type IN ('debit','credit')),
    category    TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_txn_user     ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_txn_month    ON transactions(user_id, month);
CREATE INDEX IF NOT EXISTS idx_txn_category ON transactions(user_id, category);

CREATE TABLE IF NOT EXISTS analysis_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    computed_at TEXT NOT NULL DEFAULT (datetime('now')),
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analysis_user ON analysis_results(user_id);

CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    computed_at TEXT NOT NULL DEFAULT (datetime('now')),
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions(user_id);

CREATE TABLE IF NOT EXISTS behavior_profiles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    computed_at TEXT NOT NULL DEFAULT (datetime('now')),
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profile_user ON behavior_profiles(user_id);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id   TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    alert_type TEXT NOT NULL,
    severity   TEXT NOT NULL CHECK(severity IN ('info','warning','critical')),
    title      TEXT NOT NULL,
    message    TEXT NOT NULL,
    metadata   TEXT,
    is_read    INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_alerts_user     ON alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_alerts_unread   ON alerts(user_id, is_read);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(user_id, severity);
"""


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
    logger.info("DB ready at %s", DB_PATH)


# Data retention: keep only the last KEEP_N rows per user per table 

def _prune(conn, table: str, user_id: str):
    conn.execute(f"""
        DELETE FROM {table}
        WHERE user_id = ?
          AND id NOT IN (
              SELECT id FROM {table}
              WHERE user_id = ?
              ORDER BY id DESC
              LIMIT {KEEP_N}
          )
    """, (user_id, user_id))


# Users 

def upsert_user(user_id: str, name: str = "User", email: str = None,
                monthly_budget: float = 50000.0) -> dict:
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO users (user_id, name, email, monthly_budget)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                name           = excluded.name,
                email          = excluded.email,
                monthly_budget = excluded.monthly_budget,
                updated_at     = datetime('now')
        """, (user_id, name, email, monthly_budget))
        row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
    return dict(row)


def get_user(user_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def update_budget(user_id: str, monthly_budget: float) -> bool:
    with get_conn() as conn:
        r = conn.execute(
            "UPDATE users SET monthly_budget=?, updated_at=datetime('now') WHERE user_id=?",
            (monthly_budget, user_id),
        )
    return r.rowcount > 0


# Transactions 

def save_transactions(user_id: str, transactions: list[dict]) -> int:
    import hashlib, uuid as _uuid
    rows = []
    for t in transactions:
        raw_key = f"{user_id}:{t['date']}:{t['description']}:{t['amount']}:{t['type']}"
        txn_id  = str(_uuid.UUID(hashlib.md5(raw_key.encode()).hexdigest()))
        rows.append((txn_id, user_id, t["date"], t["month"],
                     t["description"], t["amount"], t["type"], t["category"]))

    with get_conn() as conn:
        conn.executemany("""
            INSERT OR IGNORE INTO transactions
              (txn_id, user_id, date, month, description, amount, type, category)
            VALUES (?,?,?,?,?,?,?,?)
        """, rows)
        count = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE user_id=?", (user_id,)
        ).fetchone()[0]
    return count


def get_transactions(user_id: str, months: list[str] | None = None) -> list[dict]:
    with get_conn() as conn:
        if months:
            ph   = ",".join("?" * len(months))
            rows = conn.execute(
                f"SELECT * FROM transactions WHERE user_id=? AND month IN ({ph}) ORDER BY date",
                [user_id] + months,
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM transactions WHERE user_id=? ORDER BY date", (user_id,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_current_month_transactions(user_id: str) -> list[dict]:
    return get_transactions(user_id, [datetime.now().strftime("%Y-%m")])


def get_month_spend_so_far(user_id: str) -> float:
    month = datetime.now().strftime("%Y-%m")
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount),0) as total FROM transactions "
            "WHERE user_id=? AND month=? AND type='debit'",
            (user_id, month),
        ).fetchone()
    return float(row["total"])


def clear_transactions(user_id):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM transactions WHERE user_id = ?",
        (user_id,)
    )

    conn.commit()
    conn.close()


# Analysis / Prediction / Behaviour profile 

def save_payload(table: str, user_id: str, payload: dict):
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO {table} (user_id, payload) VALUES (?, ?)",
            (user_id, json.dumps(payload)),
        )
        _prune(conn, table, user_id)


def get_latest_payload(table: str, user_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT payload FROM {table} WHERE user_id=? ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()

    return json.loads(row["payload"]) if row else None


def save_analysis(user_id: str, payload: dict):
    save_payload("analysis_results", user_id, payload)


def get_latest_analysis(user_id: str) -> dict | None:
    return get_latest_payload("analysis_results", user_id)


def save_prediction(user_id: str, payload: dict):
    save_payload("predictions", user_id, payload)


def get_latest_prediction(user_id: str) -> dict | None:
    return get_latest_payload("predictions", user_id)


def save_behavior_profile(user_id: str, payload: dict):
    save_payload("behavior_profiles", user_id, payload)


def get_latest_behavior_profile(user_id: str) -> dict | None:
    return get_latest_payload("behavior_profiles", user_id)

#  Alerts 

def save_alert(user_id: str, alert_type: str, severity: str,
               title: str, message: str, metadata: dict = None) -> str:
    import uuid as _uuid
    alert_id = str(_uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO alerts (alert_id, user_id, alert_type, severity, title, message, metadata) "
            "VALUES (?,?,?,?,?,?,?)",
            (alert_id, user_id, alert_type, severity, title, message,
             json.dumps(metadata) if metadata else None),
        )
    return alert_id


def get_alerts(user_id: str, unread_only: bool = False, limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        q      = "SELECT * FROM alerts WHERE user_id=?"
        params = [user_id]
        if unread_only:
            q += " AND is_read=0"
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(q, params).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else None
        result.append(d)
    return result


def mark_alert_read(alert_id: str, user_id: str) -> bool:
    with get_conn() as conn:
        r = conn.execute(
            "UPDATE alerts SET is_read=1 WHERE alert_id=? AND user_id=?",
            (alert_id, user_id),
        )
    return r.rowcount > 0


def clear_old_alerts(user_id: str, alert_type: str):
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM alerts WHERE user_id=? AND alert_type=? AND is_read=0",
            (user_id, alert_type),
        )
