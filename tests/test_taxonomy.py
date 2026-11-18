from supportflow.taxonomy import priority_for_intent, sla_for_priority, team_for_intent


def test_routes_representative_intents() -> None:
    assert team_for_intent("out_of_scope") == "Manual Triage / Out of Scope"
    assert team_for_intent("cash_withdrawal_not_recognised") == "ATM & Cash Withdrawals"
    assert team_for_intent("pending_transfer") == "Transfers & Beneficiaries"
    assert team_for_intent("card_payment_not_recognised") == "Payments & Disputes"
    assert team_for_intent("verify_my_identity") == "Verification & Compliance"
    assert team_for_intent("contactless_not_working") == "Card Services"
    assert team_for_intent("balance_not_updated_after_cheque_or_cash_deposit") == "Top Ups & Deposits"


def test_high_priority_is_rule_based() -> None:
    priority, reason = priority_for_intent("compromised_card")
    assert priority == "Critical"
    assert reason.startswith("Rule-based")


def test_priority_levels_and_sla_policy() -> None:
    assert priority_for_intent("lost_or_stolen_card")[0] == "High"
    assert priority_for_intent("pending_transfer")[0] == "Medium"
    assert priority_for_intent("card_arrival")[0] == "Low"
    assert sla_for_priority("Critical")[0] == 1
    assert sla_for_priority("High")[0] == 4
    assert sla_for_priority("Medium")[0] == 8
    assert sla_for_priority("Low")[0] == 24
