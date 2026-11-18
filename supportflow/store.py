from __future__ import annotations

import csv
import hashlib
import io
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .taxonomy import priority_for_intent, sla_for_priority, team_for_intent

ALLOWED_STATUSES = {"Needs review", "Routed", "Resolved"}


class DuplicateBatchError(ValueError):
    """Raised when an identical batch has already been imported."""


class TicketStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    message TEXT NOT NULL,
                    predicted_intent TEXT NOT NULL,
                    final_intent TEXT NOT NULL,
                    department TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    threshold REAL NOT NULL,
                    priority TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reviewed INTEGER NOT NULL DEFAULT 0,
                    inference_ms REAL NOT NULL,
                    priority_reason TEXT,
                    sla_hours INTEGER,
                    sla_reason TEXT,
                    sla_due_at TEXT,
                    resolved_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_imports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT NOT NULL UNIQUE,
                    imported_at TEXT NOT NULL DEFAULT (datetime('now')),
                    row_count INTEGER NOT NULL
                )
                """
            )
            existing_columns = {
                str(row["name"]) for row in connection.execute("PRAGMA table_info(tickets)").fetchall()
            }
            migrations = {
                "priority_reason": "ALTER TABLE tickets ADD COLUMN priority_reason TEXT",
                "sla_hours": "ALTER TABLE tickets ADD COLUMN sla_hours INTEGER",
                "sla_reason": "ALTER TABLE tickets ADD COLUMN sla_reason TEXT",
                "sla_due_at": "ALTER TABLE tickets ADD COLUMN sla_due_at TEXT",
                "resolved_at": "ALTER TABLE tickets ADD COLUMN resolved_at TEXT",
            }
            for column, statement in migrations.items():
                if column not in existing_columns:
                    connection.execute(statement)
            self._backfill_sla_fields(connection)

    def _backfill_sla_fields(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT id, created_at, final_intent FROM tickets
            WHERE sla_hours IS NULL OR sla_due_at IS NULL OR priority_reason IS NULL OR sla_reason IS NULL
            """
        ).fetchall()
        for row in rows:
            priority, priority_reason = priority_for_intent(str(row["final_intent"]))
            sla_hours, sla_reason = sla_for_priority(priority)
            connection.execute(
                """
                UPDATE tickets
                SET priority = ?, priority_reason = ?, sla_hours = ?, sla_reason = ?, sla_due_at = ?
                WHERE id = ?
                """,
                (
                    priority,
                    priority_reason,
                    sla_hours,
                    sla_reason,
                    _deadline_text(str(row["created_at"]), sla_hours),
                    int(row["id"]),
                ),
            )

    def add_prediction(self, message: str, prediction: dict[str, Any]) -> int:
        status = "Routed" if prediction["decision"] == "Automatically route" else "Needs review"
        created_at = _utc_now_text()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO tickets (
                    message, predicted_intent, final_intent, department, confidence,
                    threshold, priority, decision, status, inference_ms, created_at,
                    priority_reason, sla_hours, sla_reason, sla_due_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message,
                    prediction["intent"],
                    prediction["intent"],
                    prediction["team"],
                    float(prediction["confidence"]),
                    float(prediction["threshold"]),
                    prediction["priority"],
                    prediction["decision"],
                    status,
                    float(prediction["inference_ms"]),
                    created_at,
                    prediction["priority_reason"],
                    int(prediction["sla_hours"]),
                    prediction["sla_reason"],
                    _deadline_text(created_at, int(prediction["sla_hours"])),
                ),
            )
            return int(cursor.lastrowid)

    def add_batch(
        self,
        messages: list[str],
        predictions: list[dict[str, Any]],
        fingerprint: str,
    ) -> list[int]:
        if not messages or len(messages) != len(predictions):
            raise ValueError("Batch messages and predictions must be non-empty and aligned.")
        ticket_ids: list[int] = []
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO batch_imports (fingerprint, row_count) VALUES (?, ?)",
                    (fingerprint, len(messages)),
                )
                for message, prediction in zip(messages, predictions, strict=True):
                    status = "Routed" if prediction["decision"] == "Automatically route" else "Needs review"
                    created_at = _utc_now_text()
                    cursor = connection.execute(
                        """
                        INSERT INTO tickets (
                            message, predicted_intent, final_intent, department, confidence,
                            threshold, priority, decision, status, inference_ms, created_at,
                            priority_reason, sla_hours, sla_reason, sla_due_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            message,
                            prediction["intent"],
                            prediction["intent"],
                            prediction["team"],
                            float(prediction["confidence"]),
                            float(prediction["threshold"]),
                            prediction["priority"],
                            prediction["decision"],
                            status,
                            float(prediction["inference_ms"]),
                            created_at,
                            prediction["priority_reason"],
                            int(prediction["sla_hours"]),
                            prediction["sla_reason"],
                            _deadline_text(created_at, int(prediction["sla_hours"])),
                        ),
                    )
                    ticket_ids.append(int(cursor.lastrowid))
        except sqlite3.IntegrityError as exc:
            raise DuplicateBatchError("This exact CSV batch has already been imported.") from exc
        return ticket_ids

    def has_batch(self, fingerprint: str) -> bool:
        with self._connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM batch_imports WHERE fingerprint = ? LIMIT 1", (fingerprint,)
                ).fetchone()
                is not None
            )

    def count_existing_messages(self, messages: list[str]) -> int:
        if not messages:
            return 0
        unique_messages = list(set(messages))
        placeholders = ",".join("?" for _ in unique_messages)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(DISTINCT message) FROM tickets WHERE message IN ({placeholders})",
                unique_messages,
            ).fetchone()
            return int(row[0])

    def list_tickets(
        self,
        status: str | None = None,
        department: str | None = None,
        priority: str | None = None,
        sla_state: str | None = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        parameters: list[str] = []
        if status and status != "All":
            clauses.append("status = ?")
            parameters.append(status)
        if department and department != "All":
            clauses.append("department = ?")
            parameters.append(department)
        if priority and priority != "All":
            clauses.append("priority = ?")
            parameters.append(priority)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = "SELECT * FROM tickets" + where + " ORDER BY id DESC"
        with self._connect() as connection:
            tickets = pd.read_sql_query(query, connection, params=parameters)
        tickets = _with_sla_state(tickets)
        if sla_state and sla_state != "All" and not tickets.empty:
            tickets = tickets[tickets["sla_state"] == sla_state].reset_index(drop=True)
        return tickets

    def get_ticket(self, ticket_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
            if row is None:
                return None
            ticket = dict(row)
            ticket["sla_state"] = sla_state_for_ticket(ticket)
            return ticket

    def review_ticket(self, ticket_id: int, final_intent: str) -> None:
        priority, priority_reason = priority_for_intent(final_intent)
        sla_hours, sla_reason = sla_for_priority(priority)
        with self._connect() as connection:
            ticket = connection.execute(
                "SELECT created_at FROM tickets WHERE id = ? AND status = 'Needs review'", (ticket_id,)
            ).fetchone()
            if ticket is None:
                raise ValueError("Ticket is missing or is no longer waiting for review.")
            cursor = connection.execute(
                """
                UPDATE tickets
                SET final_intent = ?, department = ?, reviewed = 1, status = 'Routed',
                    priority = ?, priority_reason = ?, sla_hours = ?, sla_reason = ?, sla_due_at = ?
                WHERE id = ? AND status = 'Needs review'
                """,
                (
                    final_intent,
                    team_for_intent(final_intent),
                    priority,
                    priority_reason,
                    sla_hours,
                    sla_reason,
                    _deadline_text(str(ticket["created_at"]), sla_hours),
                    ticket_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Ticket is missing or is no longer waiting for review.")

    def mark_resolved(self, ticket_id: int) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE tickets SET status = 'Resolved', resolved_at = ?
                WHERE id = ? AND status = 'Routed'
                """,
                (_utc_now_text(), ticket_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Only routed tickets can be marked resolved.")

    def summary(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute("SELECT status, COUNT(*) AS count FROM tickets GROUP BY status").fetchall()
            status_counts = {str(row["status"]): int(row["count"]) for row in rows}
            total = int(connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0])
            corrections = int(
                connection.execute(
                    "SELECT COUNT(*) FROM tickets WHERE reviewed = 1 AND final_intent != predicted_intent"
                ).fetchone()[0]
            )
        return {
            "total": total,
            "needs_review": status_counts.get("Needs review", 0),
            "routed": status_counts.get("Routed", 0),
            "resolved": status_counts.get("Resolved", 0),
            "corrections": corrections,
        }

    def analytics(self) -> dict[str, Any]:
        tickets = self.list_tickets()
        if tickets.empty:
            return {
                "tickets": tickets,
                "total": 0,
                "automation_rate": 0.0,
                "initial_review_rate": 0.0,
                "resolution_rate": 0.0,
                "reviewed_count": 0,
                "correction_count": 0,
                "duplicate_message_groups": 0,
                "overdue_count": 0,
                "sla_breached_count": 0,
            }
        total = len(tickets)
        reviewed = int(tickets["reviewed"].sum())
        corrections = int(
            ((tickets["reviewed"] == 1) & (tickets["final_intent"] != tickets["predicted_intent"])).sum()
        )
        with self._connect() as connection:
            duplicate_groups = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT message FROM tickets GROUP BY message HAVING COUNT(*) > 1
                    )
                    """
                ).fetchone()[0]
            )
        return {
            "tickets": tickets,
            "total": total,
            "automation_rate": float((tickets["decision"] == "Automatically route").mean()),
            "initial_review_rate": float((tickets["decision"] == "Human review required").mean()),
            "resolution_rate": float((tickets["status"] == "Resolved").mean()),
            "reviewed_count": reviewed,
            "correction_count": corrections,
            "duplicate_message_groups": duplicate_groups,
            "overdue_count": int((tickets["sla_state"] == "Overdue").sum()),
            "sla_breached_count": int((tickets["sla_state"] == "Breached").sum()),
        }

    def correction_history(self) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query(
                """
                SELECT id, created_at, message, predicted_intent, final_intent, department, confidence
                FROM tickets WHERE reviewed = 1 ORDER BY id DESC
                """,
                connection,
            )


def tickets_csv(frame: pd.DataFrame) -> bytes:
    export = frame.copy()
    text_columns = export.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        export[column] = export[column].map(_escape_spreadsheet_formula)
    buffer = io.StringIO(newline="")
    export.to_csv(buffer, index=False, quoting=csv.QUOTE_MINIMAL)
    return buffer.getvalue().encode("utf-8")


def batch_fingerprint(messages: list[str]) -> str:
    digest = hashlib.sha256()
    for message in sorted(messages):
        digest.update(message.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _escape_spreadsheet_formula(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def sla_state_for_ticket(ticket: dict[str, Any], now: datetime | None = None) -> str:
    due_at = _parse_timestamp(str(ticket["sla_due_at"]))
    if ticket["status"] == "Resolved":
        if not ticket.get("resolved_at"):
            return "Resolved (legacy)"
        return "Met" if _parse_timestamp(str(ticket["resolved_at"])) <= due_at else "Breached"
    current = now or datetime.now(timezone.utc).replace(tzinfo=None)
    if current > due_at:
        return "Overdue"
    warning_window = timedelta(hours=max(1.0, float(ticket["sla_hours"]) * 0.25))
    return "Due soon" if due_at - current <= warning_window else "On track"


def _with_sla_state(tickets: pd.DataFrame) -> pd.DataFrame:
    if tickets.empty:
        tickets["sla_state"] = pd.Series(dtype="object")
        return tickets
    tickets = tickets.copy()
    tickets["sla_state"] = [sla_state_for_ticket(row) for row in tickets.to_dict("records")]
    return tickets


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0).isoformat(sep=" ")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def _deadline_text(created_at: str, sla_hours: int) -> str:
    return (_parse_timestamp(created_at) + timedelta(hours=sla_hours)).isoformat(sep=" ", timespec="seconds")
