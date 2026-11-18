from __future__ import annotations

CRITICAL_PRIORITY_INTENTS = {
    "compromised_card",
    "lost_or_stolen_phone",
}

HIGH_PRIORITY_INTENTS = {
    "card_payment_not_recognised",
    "cash_withdrawal_not_recognised",
    "direct_debit_payment_not_recognised",
    "lost_or_stolen_card",
}

MEDIUM_PRIORITY_TOKENS = (
    "declined",
    "failed",
    "not_received",
    "not_working",
    "pending",
    "swallowed",
    "unable_to_verify",
)


def team_for_intent(intent: str) -> str:
    name = intent.lower().strip().strip("?")
    if name == "out_of_scope":
        return "Manual Triage / Out of Scope"
    if any(token in name for token in ("cash_withdrawal", "cash_received", "atm_", "card_swallowed")):
        return "ATM & Cash Withdrawals"
    if any(token in name for token in ("transfer", "beneficiary", "receiving_money")):
        return "Transfers & Beneficiaries"
    if "top_up" in name or "cash_or_cheque" in name or "cheque_or_cash" in name:
        return "Top Ups & Deposits"
    if any(token in name for token in ("card_payment", "direct_debit", "transaction_charged", "refund", "extra_charge")):
        return "Payments & Disputes"
    if any(token in name for token in ("verify", "identity", "source_of_funds", "age_limit", "country_support")):
        return "Verification & Compliance"
    if any(token in name for token in ("exchange", "fiat_currency")):
        return "Currency & Exchange"
    if any(token in name for token in ("virtual_card", "disposable", "apple_pay", "google_pay")):
        return "Digital Cards & Wallets"
    if "card" in name or "pin" in name or "visa_or_mastercard" in name or "contactless" in name:
        return "Card Services"
    if any(token in name for token in ("passcode", "personal_details", "phone", "terminate_account")):
        return "Account Access & Profile"
    return "General Banking Support"


def priority_for_intent(intent: str) -> tuple[str, str]:
    name = intent.lower().strip().strip("?")
    if name in CRITICAL_PRIORITY_INTENTS:
        return "Critical", "Rule-based: possible account compromise or loss of the customer's access device."
    if name in HIGH_PRIORITY_INTENTS:
        return "High", "Rule-based: possible fraud, unrecognised money movement, or a lost payment card."
    if any(token in name for token in MEDIUM_PRIORITY_TOKENS):
        return "Medium", "Rule-based: blocked, failed, delayed, or incomplete customer action."
    return "Low", "Rule-based: standard informational or service request."


SLA_HOURS = {
    "Critical": 1,
    "High": 4,
    "Medium": 8,
    "Low": 24,
}


def sla_for_priority(priority: str) -> tuple[int, str]:
    try:
        hours = SLA_HOURS[priority]
    except KeyError as exc:
        raise ValueError(f"Unsupported priority: {priority}") from exc
    return hours, f"Demo policy: first response is due within {hours} clock hour{'s' if hours != 1 else ''}."
