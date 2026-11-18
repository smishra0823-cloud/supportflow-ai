from __future__ import annotations

import re

MAX_TICKET_CHARS = 2_000
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE = re.compile(r"\s+")


class TicketTextError(ValueError):
    """Raised when submitted ticket text is empty or outside safe limits."""


def clean_ticket_text(value: str) -> str:
    if not isinstance(value, str):
        raise TicketTextError("Ticket text must be a string.")
    cleaned = _CONTROL_CHARS.sub(" ", value)
    cleaned = _SPACE.sub(" ", cleaned).strip()
    if not cleaned:
        raise TicketTextError("Enter a customer message before classifying it.")
    if len(cleaned) > MAX_TICKET_CHARS:
        raise TicketTextError(f"Ticket text must be {MAX_TICKET_CHARS:,} characters or fewer.")
    return cleaned


def humanize_intent(intent: str) -> str:
    normalized = intent.strip().strip("?").replace("_", " ")
    return normalized[:1].upper() + normalized[1:]
