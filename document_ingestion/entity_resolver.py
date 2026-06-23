"""Entity resolver: resolve actors and validate roles from extracted events.

Provides a generic framework for resolving free-text person names to canonical
actor IDs, and for validating stated roles (e.g. Petitioner/Respondent) against
the forum they appear in.

To use with your own case data, create a reference/case_entities.json file
with the following structure:
{
  "actors": [{"id": "A001", "name": "John Doe", "aliases": ["john doe"]}],
  "forums": [{"id": "F001", "name": "District Court", "petitioner": "A001", "respondent": "A002"}],
  "source_forum_map": {"document-name": "F001"}
}
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_REF_PATH = Path(__file__).with_name("reference") / "case_entities.json"

_ROLE_RE = re.compile(r"\b(petitioner|respondent|plaintiff|defendant|movant)\b", re.I)
_COURT_RE = re.compile(r"\b(court|clerk|magistrate|judge|tribunal)\b", re.I)

_FORUM_KEYWORDS: list[tuple[re.Pattern, str]] = []


@lru_cache(maxsize=1)
def _ref() -> dict:
    if not _REF_PATH.exists():
        return {"actors": [], "forums": [], "source_forum_map": {}}
    return json.loads(_REF_PATH.read_text(encoding="utf-8"))


_TITLES = {"mr", "mrs", "ms", "dr", "hon", "honorable", "judge", "justice",
           "gm", "drho", "magistrate", "counsel", "esq", "attorney", "the",
           "petitioner", "respondent", "father", "mother", "plaintiff",
           "defendant", "movant", "clerk", "minor", "child", "via", "pro", "se"}


def _person_tokens(text: str) -> list[str]:
    low = re.sub(r"[^a-z\s]", " ", (text or "").lower())
    return [t for t in low.split() if t not in _TITLES and len(t) > 1]


@lru_cache(maxsize=1)
def _person_records() -> list[dict]:
    recs = []
    for a in _ref()["actors"]:
        toks = _person_tokens(a["name"])
        if not toks:
            continue
        recs.append({
            "id": a["id"], "name": a["name"],
            "first": toks[0], "surname": toks[-1], "tokens": set(toks),
        })
    return recs


@lru_cache(maxsize=1)
def _surname_to_actors() -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = {}
    for r in _person_records():
        idx.setdefault(r["surname"], []).append(r)
    return idx


def _forum(fid: str | None) -> dict | None:
    if not fid:
        return None
    return next((f for f in _ref()["forums"] if f["id"] == fid), None)


def _actor_name(aid: str | None) -> str | None:
    if not aid:
        return None
    return next((a["name"] for a in _ref()["actors"] if a["id"] == aid), None)


def resolve_person(text: str) -> tuple[str | None, str | None]:
    """Resolve a free-text name to a canonical actor.

    Returns (actor_id, canonical_name) or (None, None).
    """
    toks = _person_tokens(text)
    if not toks:
        return None, None
    tokset = set(toks)

    for r in _person_records():
        if r["first"] in tokset and r["surname"] in tokset:
            return r["id"], r["name"]

    for sn in toks:
        cands = _surname_to_actors().get(sn)
        if not cands:
            continue
        others = [t for t in toks if t != sn]
        if not others:
            if len(cands) == 1:
                return cands[0]["id"], cands[0]["name"]
            continue
        for r in cands:
            if r["first"] in others:
                return r["id"], r["name"]
    return None, None


def resolve_party(party: str) -> tuple[str | None, str | None]:
    return resolve_person(party)


def normalize_name(name: str) -> str:
    return " ".join(_person_tokens(name))


_DETERMINERS = {"the", "a", "an", "each", "both", "either", "another", "every",
                "all", "any", "no", "this", "that", "these", "those", "his",
                "her", "their", "our", "your", "its", "said", "such", "same"}
_GENERIC_NOUNS = {"parent", "parents", "party", "parties", "child", "children",
                  "kid", "kids", "person", "people", "individual", "minor",
                  "minors", "petitioner", "respondent", "mother", "father",
                  "plaintiff", "defendant", "movant", "spouse", "witness",
                  "witnesses", "counsel", "attorney", "court", "courts", "clerk",
                  "judge", "officer", "sheriff", "deputy", "guardian"}
_INSTITUTION_WORDS = {"court", "courts", "nation", "state", "county", "circuit",
                      "district", "department", "office", "bar", "committee",
                      "services", "service", "agency", "llc", "inc", "school",
                      "university", "tribunal", "commission", "division",
                      "clerk", "republic", "government", "council"}


def is_person_name(name: str) -> bool:
    """Heuristic: does this string name an actual person (vs a role/institution)?"""
    if resolve_person(name)[0]:
        return True
    raw = (name or "").strip()
    if not raw:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z.'-]*", raw)
    if not words:
        return False
    low = [w.lower().strip(".'-") for w in words]
    if low[0] in _DETERMINERS:
        return False
    if any(w in _INSTITUTION_WORDS for w in low):
        return False
    content = [w for w in low if w not in _TITLES and len(w) > 1]
    if not content:
        return False
    if all(w in _GENERIC_NOUNS for w in content):
        return False
    return True


def actor_key(name: str) -> tuple[str, str, bool]:
    """Map a name to (key, display_name, is_canonical)."""
    import hashlib

    aid, canon = resolve_person(name)
    if aid:
        return aid, canon, True
    norm = normalize_name(name)
    if not norm:
        return "", name, False
    h = hashlib.sha1(norm.encode()).hexdigest()[:6]
    return f"X{h}", name.strip(), False


def classify_forum(source_document: str, court: str = "") -> str | None:
    stem = source_document.replace(".clean.md", "").replace(".md", "")
    fid = _ref()["source_forum_map"].get(stem)
    if fid:
        return fid
    blob = f"{source_document} {court}"
    for pat, f in _FORUM_KEYWORDS:
        if pat.search(blob):
            return f
    return None


def stated_role(party: str) -> str | None:
    m = _ROLE_RE.search(party or "")
    return m.group(1).lower() if m else None


def validate(party: str, source_document: str, court: str = "") -> dict:
    """Resolve + validate one event's actor/role/forum."""
    actor_id, canonical = resolve_party(party)
    forum_id = classify_forum(source_document, court)
    forum = _forum(forum_id)
    role = stated_role(party)
    is_court = bool(_COURT_RE.search(party or "")) and actor_id is None

    forum_role = None
    if forum and actor_id:
        if forum.get("petitioner") == actor_id:
            forum_role = "petitioner"
        elif forum.get("respondent") == actor_id:
            forum_role = "respondent"

    if actor_id is None:
        status = "court" if is_court else "unresolved"
    elif role is None:
        status = "na"
    elif forum_role is None:
        status = "na"
    elif role == forum_role:
        status = "ok"
    else:
        status = "conflict"

    canonical_party = _canonical_party(
        party, actor_id, canonical, forum, forum_role, is_court
    )

    return {
        "actor_id": actor_id,
        "actor_canonical": canonical,
        "forum_id": forum_id,
        "forum_role": forum_role,
        "stated_role": role,
        "role_status": status,
        "canonical_party": canonical_party,
    }


def _canonical_party(party, actor_id, canonical, forum, forum_role, is_court) -> str:
    if actor_id and forum_role and forum:
        return f"{canonical} ({actor_id}) — {forum_role.capitalize()} in {forum['id']}"
    if actor_id:
        return f"{canonical} ({actor_id})"
    if is_court and forum:
        return f"the Court — {forum['name']}, {forum['id']}"
    return party or "undetermined"


def enrich_event(event: dict) -> dict:
    return validate(
        event.get("party", "") or "",
        event.get("source_documents") or event.get("source_document", "") or "",
        event.get("court", "") or "",
    )


def resolve_cast(party: str, other_actors: str, source_document: str,
                 court: str = "") -> list[dict]:
    """Resolve the full participant cast of an event (primary + secondary)."""
    cast: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, is_primary: bool) -> None:
        key = normalize_name(name)
        if not name or key in seen:
            return
        seen.add(key)
        g = validate(name, source_document, court)
        cast.append({**g, "name": name, "is_primary": 1 if is_primary else 0})

    if party and party != "undetermined":
        _add(party, True)
    for nm in re.split(r"[;]", other_actors or ""):
        nm = nm.strip()
        if nm and is_person_name(nm):
            _add(nm, False)
    return cast
