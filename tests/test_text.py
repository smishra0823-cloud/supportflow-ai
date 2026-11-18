import pytest

from supportflow.text import TicketTextError, clean_ticket_text, humanize_intent


def test_clean_ticket_text_normalizes_whitespace() -> None:
    assert clean_ticket_text("  card\n  is\tmissing ") == "card is missing"


def test_clean_ticket_text_rejects_empty() -> None:
    with pytest.raises(TicketTextError):
        clean_ticket_text(" \n\t ")


def test_humanize_intent() -> None:
    assert humanize_intent("cash_withdrawal_not_recognised") == "Cash withdrawal not recognised"
    assert humanize_intent("reverted_card_payment?") == "Reverted card payment"
