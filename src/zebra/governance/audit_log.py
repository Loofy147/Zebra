# src/zebra/governance/audit_log.py
from __future__ import annotations
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import pathlib
import logging

logger = logging.getLogger("zebra.audit")

DB_PATH = pathlib.Path("zebra_audit.db")

class AuditLog:
    """
    PoC: sqlite-backed audit log for decisions and violations.
    In production: replace with a hardened event store (Timescale, ES, BigQuery) with access control.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(DB_PATH)
        self._ensure_db()

    def _ensure_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                kind TEXT,
                payload TEXT
            )
        """)
        conn.commit()
        conn.close()

    def store(self, report: Dict[str, Any]):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("INSERT INTO audit(created_at, kind, payload) VALUES (?, ?, ?)",
                    (datetime.utcnow().isoformat(), "explanation", json.dumps(report)))
        conn.commit()
        conn.close()
        logger.info("AuditLog: stored explanation for %s", report.get("decision_id"))
        return True

    def log_violation(self, decision: Dict[str, Any], failed_checks: List[str]):
        payload = {
            "decision": decision,
            "failed_checks": failed_checks
        }
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("INSERT INTO audit(created_at, kind, payload) VALUES (?, ?, ?)",
                    (datetime.utcnow().isoformat(), "violation", json.dumps(payload)))
        conn.commit()
        conn.close()
        logger.warning("AuditLog: violation logged for %s: %s", decision.get("id"), failed_checks)

    def log_approval(self, decision: Dict[str, Any], checks: Dict[str, bool]):
        payload = {"decision": decision, "checks": checks}
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("INSERT INTO audit(created_at, kind, payload) VALUES (?, ?, ?)",
                    (datetime.utcnow().isoformat(), "approval", json.dumps(payload)))
        conn.commit()
        conn.close()
        logger.info("AuditLog: approval recorded for %s", decision.get("id"))