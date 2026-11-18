from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from supportflow.model import SupportFlowModel
from supportflow.store import DuplicateBatchError, TicketStore, batch_fingerprint, tickets_csv
from supportflow.text import TicketTextError, clean_ticket_text, humanize_intent


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "model_bundle"
METRICS_PATH = ROOT / "reports" / "generated" / "metrics.json"
DATABASE_PATH = ROOT / "data" / "supportflow.db"
SAMPLE_BATCH_PATH = ROOT / "samples" / "ticket_batch.csv"

DEMO_MESSAGES = {
    "ATM cash problem": "The ATM declined my cash withdrawal.",
    "Unknown card payment": "There is a card transaction on my statement that I do not recognize.",
    "Pending transfer": "I sent money yesterday but the recipient still has not received it.",
    "Forgotten passcode": "I forgot the passcode for my banking app and cannot sign in.",
    "Outside banking scope": "Can you tell me whether it will rain tomorrow?",
}

st.set_page_config(page_title="SupportFlow AI", page_icon="🎫", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2rem;}
    [data-testid="stMetric"] {background:#f5f3ff; border:1px solid #ddd6fe; padding:12px; border-radius:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("SupportFlow AI")
st.caption("Banking support ticket classification, confidence-aware routing, priority, SLA tracking, and similar-case retrieval")
st.info("Demo only. Do not enter account numbers, card numbers, passwords, or other real personal banking information.")

if not (ARTIFACT_DIR / "metadata.json").exists():
    st.error("The trained model is missing. Run `python -m scripts.train_and_evaluate`, then refresh the app.")
    st.stop()


@st.cache_resource
def load_model(path: str) -> SupportFlowModel:
    return SupportFlowModel(Path(path))


model = load_model(str(ARTIFACT_DIR))
store = TicketStore(DATABASE_PATH)
single_tab, operations_tab, batch_tab, analytics_tab = st.tabs(
    ["Classify one ticket", "Operations dashboard", "Batch intake", "Analytics"]
)

with single_tab:
    st.subheader("1. Enter a customer support message")
    demo_name = st.selectbox("Try an example", list(DEMO_MESSAGES))
    ticket_text = st.text_area(
        "Customer message", value=DEMO_MESSAGES[demo_name], height=120, max_chars=2_000
    )
    classify = st.button("Classify and route ticket", type="primary", width="stretch")
    if classify:
        try:
            result = model.predict(ticket_text)
            st.session_state["last_result"] = asdict(result)
            st.session_state["last_message"] = clean_ticket_text(ticket_text)
            st.session_state.pop("last_queued_id", None)
        except TicketTextError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"The ticket could not be classified: {exc}")

    if "last_result" in st.session_state:
        payload = st.session_state["last_result"]
        st.subheader("2. Routing result")
        if payload["decision"] == "Automatically route":
            st.success(f"Automatically route to: {payload['team']}")
        else:
            st.warning("Human review required — confidence is below the validation-selected threshold.")
        first, second, third, fourth, fifth = st.columns(5)
        first.metric("Predicted issue", payload["intent_display"])
        second.metric("Confidence", f"{payload['confidence']:.1%}")
        third.metric("Business priority", payload["priority"])
        fourth.metric("First-response SLA", f"{payload['sla_hours']}h")
        fifth.metric("Total model time", f"{payload['inference_ms']:.0f} ms")
        st.caption(
            payload["priority_reason"]
            + " "
            + payload["sla_reason"]
            + " Priority and SLA are transparent business rules, not ML predictions."
        )
        left, right = st.columns(2)
        with left:
            st.markdown("#### Top model predictions")
            st.dataframe(
                [
                    {"Issue": row["intent_display"], "Confidence": f"{row['confidence']:.1%}"}
                    for row in payload["top_predictions"]
                ],
                hide_index=True,
                width="stretch",
            )
        with right:
            st.markdown("#### Similar resolved examples")
            st.dataframe(
                [
                    {
                        "Previous message": row["message"],
                        "Known issue": row["intent"],
                        "Similarity": f"{row['similarity']:.1%}",
                    }
                    for row in payload["similar_tickets"]
                ],
                hide_index=True,
                width="stretch",
            )
        if st.button("Add ticket to support queue", width="stretch"):
            ticket_id = store.add_prediction(st.session_state["last_message"], payload)
            st.session_state["last_queued_id"] = ticket_id
        if "last_queued_id" in st.session_state:
            st.success(f"Ticket #{st.session_state['last_queued_id']} was added to the operations queue.")
        st.download_button(
            "Download routing result (JSON)",
            json.dumps(payload, indent=2),
            file_name="supportflow_routing_result.json",
            mime="application/json",
            width="stretch",
        )

with operations_tab:
    st.subheader("Support operations")
    summary = store.summary()
    metric_columns = st.columns(5)
    metric_columns[0].metric("All tickets", summary["total"])
    metric_columns[1].metric("Needs review", summary["needs_review"])
    metric_columns[2].metric("Routed", summary["routed"])
    metric_columns[3].metric("Resolved", summary["resolved"])
    metric_columns[4].metric("Agent corrections", summary["corrections"])
    all_tickets = store.list_tickets()
    departments = ["All"] + sorted(all_tickets["department"].unique().tolist()) if not all_tickets.empty else ["All"]
    priorities = ["All", "Critical", "High", "Medium", "Low"]
    sla_states = ["All", "Overdue", "Due soon", "On track", "Met", "Breached", "Resolved (legacy)"]
    filter_one, filter_two, filter_three, filter_four = st.columns(4)
    status_filter = filter_one.selectbox("Status", ["All", "Needs review", "Routed", "Resolved"])
    department_filter = filter_two.selectbox("Department", departments)
    priority_filter = filter_three.selectbox("Priority", priorities)
    sla_filter = filter_four.selectbox("SLA state", sla_states)
    tickets = store.list_tickets(status_filter, department_filter, priority_filter, sla_filter)
    if tickets.empty:
        st.info("No tickets match this queue. Classify a ticket or process a batch first.")
    else:
        display = tickets[
            [
                "id",
                "created_at",
                "message",
                "final_intent",
                "department",
                "confidence",
                "priority",
                "sla_due_at",
                "sla_state",
                "status",
            ]
        ].copy()
        display["final_intent"] = display["final_intent"].map(humanize_intent)
        display["confidence"] = display["confidence"].map(lambda value: f"{value:.1%}")
        display.columns = [
            "ID",
            "Created (UTC)",
            "Message",
            "Issue",
            "Department",
            "Confidence",
            "Priority",
            "SLA due (UTC)",
            "SLA state",
            "Status",
        ]
        st.dataframe(display, hide_index=True, width="stretch")
        st.download_button(
            "Download current queue (CSV)",
            tickets_csv(tickets),
            file_name="supportflow_ticket_queue.csv",
            mime="text/csv",
            width="stretch",
        )
        selected_id = st.selectbox("Open ticket", tickets["id"].astype(int).tolist())
        selected = store.get_ticket(int(selected_id))
        if selected is not None:
            st.write(selected["message"])
            st.caption(
                f"Model prediction: {humanize_intent(selected['predicted_intent'])} · "
                f"Confidence: {selected['confidence']:.1%} · Current team: {selected['department']} · "
                f"Priority: {selected['priority']} · SLA due: {selected['sla_due_at']} UTC · "
                f"SLA state: {selected['sla_state']}"
            )
            st.caption(selected["priority_reason"] + " " + selected["sla_reason"])
            if selected["status"] == "Needs review":
                intent_options = ["out_of_scope"] + list(model.classifier.classes_)
                default_index = intent_options.index(selected["predicted_intent"])
                corrected_intent = st.selectbox(
                    "Confirm or correct the issue",
                    intent_options,
                    index=default_index,
                    format_func=humanize_intent,
                )
                if st.button("Confirm review and route", type="primary"):
                    store.review_ticket(int(selected_id), corrected_intent)
                    st.success("Review saved. The ticket is now routed to the selected department.")
                    st.rerun()
            elif selected["status"] == "Routed":
                if st.button("Mark ticket resolved", type="primary"):
                    store.mark_resolved(int(selected_id))
                    st.success("Ticket marked resolved.")
                    st.rerun()
            else:
                st.success("This ticket is resolved.")

with batch_tab:
    st.subheader("Batch ticket intake")
    st.write("Upload a CSV with a `message` column. A maximum of 200 tickets is processed per batch.")
    st.download_button(
        "Download sample CSV",
        SAMPLE_BATCH_PATH.read_bytes(),
        file_name="supportflow_sample_tickets.csv",
        mime="text/csv",
    )
    uploaded_batch = st.file_uploader("Ticket CSV", type=["csv"], max_upload_size=5)
    if uploaded_batch is not None:
        try:
            batch = pd.read_csv(uploaded_batch)
            if "message" not in batch.columns:
                raise ValueError("The CSV must contain a column named `message`.")
            if not 1 <= len(batch) <= 200:
                raise ValueError("A batch must contain between 1 and 200 rows.")
            cleaned_messages = [clean_ticket_text(str(message)) for message in batch["message"]]
            fingerprint = batch_fingerprint(cleaned_messages)
            exact_batch_seen = store.has_batch(fingerprint)
            existing_count = store.count_existing_messages(cleaned_messages)
            st.dataframe(batch.head(20), hide_index=True, width="stretch")
            duplicate_detected = exact_batch_seen or existing_count == len(set(cleaned_messages))
            if duplicate_detected:
                st.warning(
                    "This batch appears to be already imported. It was not processed again, preventing duplicate tickets."
                )
            elif existing_count:
                st.info(
                    f"{existing_count} distinct message(s) already exist, but this is not an identical prior batch."
                )
            if st.button(
                "Classify batch and add to queues",
                type="primary",
                width="stretch",
                disabled=duplicate_detected,
            ):
                progress = st.progress(0)
                batch_predictions = []
                for index, message in enumerate(cleaned_messages, start=1):
                    prediction = asdict(model.predict(message))
                    batch_predictions.append(prediction)
                    progress.progress(index / len(cleaned_messages))
                try:
                    store.add_batch(cleaned_messages, batch_predictions, fingerprint)
                    st.success(f"Processed and queued {len(cleaned_messages)} tickets.")
                except DuplicateBatchError as exc:
                    st.warning(str(exc))
        except (ValueError, TicketTextError, pd.errors.ParserError, DuplicateBatchError) as exc:
            st.error(str(exc))

with analytics_tab:
    st.subheader("Support analytics")
    analytics = store.analytics()
    if analytics["total"] == 0:
        st.info("Analytics will appear after tickets are added to the operations queue.")
    else:
        analytics_columns = st.columns(6)
        analytics_columns[0].metric("Tickets", analytics["total"])
        analytics_columns[1].metric("Auto-routed initially", f"{analytics['automation_rate']:.1%}")
        analytics_columns[2].metric("Needed review initially", f"{analytics['initial_review_rate']:.1%}")
        analytics_columns[3].metric("Resolution rate", f"{analytics['resolution_rate']:.1%}")
        analytics_columns[4].metric("Duplicate messages", analytics["duplicate_message_groups"])
        analytics_columns[5].metric(
            "Overdue / breached",
            analytics["overdue_count"] + analytics["sla_breached_count"],
        )
        ticket_data = analytics["tickets"].copy()
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown("#### Department workload")
            department_counts = (
                ticket_data.groupby("department").size().sort_values(ascending=False).rename("Tickets")
            )
            st.bar_chart(department_counts, horizontal=True, width="stretch")
        with chart_right:
            st.markdown("#### Ticket status")
            status_counts = ticket_data.groupby("status").size().sort_values(ascending=False).rename("Tickets")
            st.bar_chart(status_counts, horizontal=True, width="stretch")
        st.markdown("#### Most common issues")
        issue_counts = (
            ticket_data.groupby("final_intent").size().sort_values(ascending=False).head(12).rename("Tickets")
        )
        issue_counts.index = [humanize_intent(value) for value in issue_counts.index]
        st.bar_chart(issue_counts, horizontal=True, width="stretch")
        priority_left, sla_right = st.columns(2)
        with priority_left:
            st.markdown("#### Priority mix")
            priority_counts = ticket_data.groupby("priority").size().sort_values(ascending=False).rename("Tickets")
            st.bar_chart(priority_counts, horizontal=True, width="stretch")
        with sla_right:
            st.markdown("#### SLA health")
            sla_counts = ticket_data.groupby("sla_state").size().sort_values(ascending=False).rename("Tickets")
            st.bar_chart(sla_counts, horizontal=True, width="stretch")
        corrections = store.correction_history()
        st.markdown("#### Agent review history")
        if corrections.empty:
            st.info("No tickets have been reviewed by an agent yet.")
        else:
            corrections["predicted_intent"] = corrections["predicted_intent"].map(humanize_intent)
            corrections["final_intent"] = corrections["final_intent"].map(humanize_intent)
            corrections["confidence"] = corrections["confidence"].map(lambda value: f"{value:.1%}")
            corrections.columns = [
                "ID",
                "Created",
                "Message",
                "Model prediction",
                "Agent decision",
                "Final department",
                "Confidence",
            ]
            st.dataframe(corrections, hide_index=True, width="stretch")
            st.download_button(
                "Download review history (CSV)",
                tickets_csv(corrections),
                file_name="supportflow_review_history.csv",
                mime="text/csv",
            )

with st.expander("Held-out evaluation and model details"):
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        cols = st.columns(4)
        cols[0].metric("Test accuracy", f"{metrics['accuracy']:.1%}")
        cols[1].metric("Macro F1", f"{metrics['macro_f1']:.1%}")
        cols[2].metric("Auto-route accuracy", f"{metrics['auto_route_accuracy']:.1%}")
        cols[3].metric("Auto-route coverage", f"{metrics['auto_route_coverage']:.1%}")
        st.json(metrics)
    st.write(
        "The sentence encoder is pretrained, while the 77-class logistic-regression router is trained on BANKING77. "
        "The confidence threshold is selected on validation data; the official test split is held out for final evaluation."
    )
    st.warning("BANKING77 is a research benchmark. These results do not establish production banking performance.")
