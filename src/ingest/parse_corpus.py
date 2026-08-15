"""
Parses the 5 customer-corpus markdown files into a single, citation-ready
text block that gets embedded directly into the LLM's system prompt.

No chunking / embeddings / vector DB — the corpus is small (~1,800 records,
well under a modern LLM's context window), so we hand the model the whole
thing and let it cite record IDs directly. Re-run this any time the source
.md files change; there is no separate "index" to rebuild.
"""

import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _parse_pipe_table(md_text: str) -> list[dict]:
    """Parses a standard markdown pipe table into a list of row dicts."""
    lines = [l.strip() for l in md_text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:  # skip header + separator row
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def load_accounts() -> list[dict]:
    text = (DATA_DIR / "accounts.md").read_text()
    return _parse_pipe_table(text)


def load_issues() -> list[dict]:
    text = (DATA_DIR / "issues.md").read_text()
    return _parse_pipe_table(text)


def load_feature_requests() -> list[dict]:
    text = (DATA_DIR / "feature_requests.md").read_text()
    return _parse_pipe_table(text)


def load_tasks() -> list[dict]:
    text = (DATA_DIR / "tasks.md").read_text()
    return _parse_pipe_table(text)


def load_meeting_notes() -> list[dict]:
    """meeting_notes.md uses ## headers + **bold** fields, not a pipe table."""
    text = (DATA_DIR / "meeting_notes.md").read_text()
    blocks = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    notes = []
    for block in blocks:
        m = re.match(r"^## (MTG-\d+):\s*(.+)", block)
        if not m:
            continue
        note_id, account = m.group(1), m.group(2).strip()
        topic = re.search(r"\*\*Topic:\*\*\s*(.+)", block)
        attendees = re.search(r"\*\*Attendees:\*\*\s*(.+)", block)
        date = re.search(r"\*\*Date:\*\*\s*(.+)", block)
        action_items = re.findall(r"^- (.+)$", block, flags=re.MULTILINE)
        notes.append({
            "ID": note_id,
            "Account": account,
            "Topic": topic.group(1).strip() if topic else "",
            "Attendees": attendees.group(1).strip() if attendees else "",
            "Date": date.group(1).strip() if date else "",
            "Action Items": "; ".join(action_items),
        })
    return notes


def _rows_to_text(rows: list[dict], id_key: str) -> str:
    """One line per record, e.g. '[acct-001] Name=Meridian AgriTech | Industry=agriculture | ...'"""
    lines = []
    for r in rows:
        rid = r.get(id_key, "?")
        fields = " | ".join(f"{k}={v}" for k, v in r.items() if k != id_key)
        lines.append(f"[{rid}] {fields}")
    return "\n".join(lines)


def build_corpus_text() -> str:
    """Builds the full citation-ready corpus text, grouped by record type."""
    accounts = load_accounts()
    issues = load_issues()
    features = load_feature_requests()
    tasks = load_tasks()
    notes = load_meeting_notes()

    # feature_requests.md has no natural ID column - synthesize one
    for i, f in enumerate(features, start=1):
        f["ID"] = f"FEAT-{i:04d}"

    sections = [
        "=== ACCOUNTS ===",
        _rows_to_text(accounts, "ID"),
        "\n=== ISSUES ===",
        _rows_to_text(issues, "ID"),
        "\n=== FEATURE REQUESTS ===",
        _rows_to_text(features, "ID"),
        "\n=== TASKS ===",
        _rows_to_text(tasks, "ID"),
        "\n=== MEETING NOTES ===",
        _rows_to_text(notes, "ID"),
    ]
    return "\n".join(sections)


def load_all() -> dict:
    """Structured access to all record types, keyed by type name."""
    features = load_feature_requests()
    for i, f in enumerate(features, start=1):
        f["ID"] = f"FEAT-{i:04d}"
    return {
        "accounts": load_accounts(),
        "issues": load_issues(),
        "feature_requests": features,
        "tasks": load_tasks(),
        "meeting_notes": load_meeting_notes(),
    }


if __name__ == "__main__":
    data = load_all()
    for k, v in data.items():
        print(f"{k}: {len(v)} records")
    corpus_text = build_corpus_text()
    print(f"\nTotal corpus text: {len(corpus_text):,} chars (~{len(corpus_text)//4:,} tokens)")
