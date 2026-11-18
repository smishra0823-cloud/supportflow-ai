import pandas as pd

import pytest

from supportflow.store import DuplicateBatchError, TicketStore, batch_fingerprint, tickets_csv


def prediction(decision: str = "Human review required") -> dict:
    return {
        "intent": "pending_transfer",
        "team": "Transfers & Beneficiaries",
        "confidence": 0.4,
        "threshold": 0.55,
        "priority": "Medium",
        "priority_reason": "Rule-based test priority.",
        "sla_hours": 8,
        "sla_reason": "Demo policy: first response is due within 8 clock hours.",
        "decision": decision,
        "inference_ms": 4.0,
    }


def test_review_and_resolve_workflow(tmp_path) -> None:
    store = TicketStore(tmp_path / "tickets.db")
    ticket_id = store.add_prediction("Where is my transfer?", prediction())
    assert store.get_ticket(ticket_id)["status"] == "Needs review"
    store.review_ticket(ticket_id, "out_of_scope")
    reviewed = store.get_ticket(ticket_id)
    assert reviewed["status"] == "Routed"
    assert reviewed["reviewed"] == 1
    assert reviewed["final_intent"] == "out_of_scope"
    assert reviewed["department"] == "Manual Triage / Out of Scope"
    store.mark_resolved(ticket_id)
    assert store.get_ticket(ticket_id)["status"] == "Resolved"


def test_automatic_prediction_starts_routed(tmp_path) -> None:
    store = TicketStore(tmp_path / "tickets.db")
    ticket_id = store.add_prediction("Where is my transfer?", prediction("Automatically route"))
    assert store.get_ticket(ticket_id)["status"] == "Routed"
    assert store.get_ticket(ticket_id)["sla_state"] == "On track"


def test_csv_export_escapes_formulas() -> None:
    frame = pd.DataFrame([{"message": "=HYPERLINK(\"bad\")", "status": "Routed"}])
    content = tickets_csv(frame).decode("utf-8")
    assert "'=HYPERLINK" in content


def test_batch_fingerprint_is_order_independent() -> None:
    assert batch_fingerprint(["one", "two"]) == batch_fingerprint(["two", "one"])


def test_duplicate_batch_is_rejected_atomically(tmp_path) -> None:
    store = TicketStore(tmp_path / "tickets.db")
    messages = ["Where is my transfer?", "My card is missing"]
    predictions = [prediction("Automatically route"), prediction("Automatically route")]
    fingerprint = batch_fingerprint(messages)
    ids = store.add_batch(messages, predictions, fingerprint)
    assert len(ids) == 2
    with pytest.raises(DuplicateBatchError):
        store.add_batch(messages, predictions, fingerprint)
    assert store.summary()["total"] == 2


def test_analytics_reports_duplicate_messages(tmp_path) -> None:
    store = TicketStore(tmp_path / "tickets.db")
    store.add_prediction("Repeated message", prediction("Automatically route"))
    store.add_prediction("Repeated message", prediction("Automatically route"))
    analytics = store.analytics()
    assert analytics["duplicate_message_groups"] == 1
    assert analytics["automation_rate"] == 1.0


def test_review_recalculates_priority_and_sla_from_original_creation(tmp_path) -> None:
    store = TicketStore(tmp_path / "tickets.db")
    ticket_id = store.add_prediction("My phone was stolen", prediction())
    original = store.get_ticket(ticket_id)
    store.review_ticket(ticket_id, "lost_or_stolen_phone")
    reviewed = store.get_ticket(ticket_id)
    assert reviewed["priority"] == "Critical"
    assert reviewed["sla_hours"] == 1
    assert reviewed["created_at"] == original["created_at"]


def test_existing_schema_is_migrated_with_sla_fields(tmp_path) -> None:
    import sqlite3

    database = tmp_path / "legacy.db"
    store = TicketStore(database)
    ticket_id = store.add_prediction("Where is my transfer?", prediction())
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE tickets SET priority = 'Normal', priority_reason = NULL, sla_hours = NULL, sla_reason = NULL, sla_due_at = NULL WHERE id = ?",
            (ticket_id,),
        )
    migrated = TicketStore(database).get_ticket(ticket_id)
    assert migrated["priority"] == "Medium"
    assert migrated["sla_hours"] == 8
    assert migrated["sla_due_at"] is not None
