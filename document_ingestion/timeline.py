"""Stage 5: generate an evidence-backed chronological timeline.

Each entry carries, where available: date (or "undetermined"), action,
party/actor, court/jurisdiction, source document, page/chunk reference,
the verbatim excerpt relied upon, and the evidence ID (the event_id, which
doubles as the ledger reference into the `events` table).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Iterable

from .database import EventStore

UNDETERMINED = "undetermined"


@dataclass
class TimelineEntry:
    date: str                  # ISO yyyy-mm-dd or "undetermined"
    event_id: str              # evidence ID / ledger reference
    action: str
    party: str
    court: str
    source_document: str
    page_reference: str
    excerpt: str
    other_actors: str = ""
    # role-validation gate fields (may be empty on un-enriched databases)
    actor_id: str = ""
    actor_canonical: str = ""
    forum_id: str = ""
    forum_role: str = ""
    role_status: str = ""
    canonical_party: str = ""

    def as_dict(self) -> dict:
        """Criteria-shaped record for JSON timeline output."""
        return {
            "date": self.date,
            "event": self.action,
            "party": self.party,
            "canonical_party": self.canonical_party or self.party,
            "actor_id": self.actor_id,
            "other_actors": self.other_actors,
            "court": self.court,
            "forum_id": self.forum_id,
            "role_status": self.role_status,
            "source_document": self.source_document,
            "page_reference": self.page_reference,
            "excerpt": self.excerpt,
            "evidence_id": self.event_id,
        }


def _val(row, key, default=UNDETERMINED):
    """Tolerantly read a column that may be absent on legacy databases."""
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    return value if value not in (None, "") else default


def generate_timeline(store: EventStore) -> list[TimelineEntry]:
    """Return every event in chronological order, oldest first.

    Dated events sort ascending; "undetermined" dates collate last.
    """
    return [
        TimelineEntry(
            date=row["date"],
            event_id=row["event_id"],
            action=_val(row, "action", ""),
            party=_val(row, "party"),
            court=_val(row, "court"),
            source_document=_val(row, "source_documents", ""),
            page_reference=_val(row, "page_reference"),
            excerpt=_val(row, "excerpt", ""),
            other_actors=_val(row, "other_actors", ""),
            actor_id=_val(row, "actor_id", ""),
            actor_canonical=_val(row, "actor_canonical", ""),
            forum_id=_val(row, "forum_id", ""),
            forum_role=_val(row, "forum_role", ""),
            role_status=_val(row, "role_status", ""),
            canonical_party=_val(row, "canonical_party", ""),
        )
        for row in store.iter_events_chronologically()
    ]


def render_json(entries: Iterable[TimelineEntry]) -> str:
    """Render the timeline as structured JSON matching the output criteria.

    Each entry carries: date (ISO or "undetermined"), event, party, court,
    source_document, page_reference, excerpt (verbatim), evidence_id.
    """
    return json.dumps([e.as_dict() for e in entries], indent=2, ensure_ascii=False)


def render_markdown(entries: Iterable[TimelineEntry]) -> str:
    """Render the timeline as a structured, evidence-backed markdown report."""
    lines = ["# Document Timeline\n"]
    current_date = None
    for entry in entries:
        if entry.date != current_date:
            lines.append(f"\n## {entry.date or UNDETERMINED}\n")
            current_date = entry.date
        lines.append(f"### {entry.action}")
        lines.append(f"- **Date:** {entry.date or UNDETERMINED}")
        lines.append(f"- **Party/Actor:** {entry.party}")
        lines.append(f"- **Court/Jurisdiction:** {entry.court}")
        lines.append(f"- **Source document:** {entry.source_document}")
        lines.append(f"- **Page/chunk reference:** {entry.page_reference}")
        if entry.excerpt:
            excerpt = " ".join(entry.excerpt.split())
            lines.append(f"- **Excerpt relied upon:** “{excerpt}”")
        lines.append(f"- **Evidence ID:** `{entry.event_id}`")
        lines.append("")
    return "\n".join(lines) + "\n"
