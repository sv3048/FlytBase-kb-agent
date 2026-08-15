"""
Lightweight keyword filtering over the customer corpus.

No embeddings / vector DB. For each query, scores every record by simple
substring/keyword overlap against its text, and returns only the top-N
matches. This keeps per-query prompts small (fast on Groq's free tier)
while still covering the "combine records across categories" case, since
filtering runs independently over each record type.

Because this reads data/raw/*.md fresh via parse_corpus on every call,
updating the source files (add/change/remove records) is picked up
immediately - no rebuild step.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.parse_corpus import load_all

MAX_RECORDS_PER_TYPE = 25  # cap so a single query never blows the token budget

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "which", "what", "who",
    "on", "in", "of", "to", "for", "and", "or", "with", "that", "this",
    "does", "do", "did", "has", "have", "had", "any", "all", "we", "our",
    "their", "its", "be", "as", "at", "by", "from", "how", "many", "most",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _record_to_text(record: dict) -> str:
    return " ".join(str(v) for v in record.values())


def _score(query_tokens: set[str], record: dict) -> int:
    record_tokens = _tokenize(_record_to_text(record))
    return len(query_tokens & record_tokens)


def _top_matches(query: str, records: list[dict], id_key: str, limit: int) -> list[dict]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return records[:limit]
    scored = [(_score(query_tokens, r), r) for r in records]
    scored = [(s, r) for s, r in scored if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]


def _rows_to_text(rows: list[dict], id_key: str) -> str:
    lines = []
    for r in rows:
        rid = r.get(id_key, "?")
        fields = " | ".join(f"{k}={v}" for k, v in r.items() if k != id_key)
        lines.append(f"[{rid}] {fields}")
    return "\n".join(lines) if lines else "(no matching records)"


def retrieve_customer_context(query: str, limit_per_type: int = MAX_RECORDS_PER_TYPE) -> str:
    """
    Given a natural-language query, returns a citation-ready text block
    containing only the most relevant records across all 5 record types.
    """
    data = load_all()
    sections = []
    for type_name, id_key in [
        ("accounts", "ID"),
        ("issues", "ID"),
        ("feature_requests", "ID"),
        ("tasks", "ID"),
        ("meeting_notes", "ID"),
    ]:
        records = data[type_name]
        matches = _top_matches(query, records, id_key, limit_per_type)
        if matches:
            sections.append(f"=== {type_name.upper().replace('_', ' ')} ===")
            sections.append(_rows_to_text(matches, id_key))
    return "\n".join(sections) if sections else "(no relevant customer records found)"


def aggregate_features_by_industry(industry: str, top_n: int = 10) -> str:
    """
    Real computation (not LLM guessing): finds every account in the given
    industry, then counts how many of those accounts requested each
    feature, and returns a ranked table. Used for "most requested
    features among accounts in X industry" style questions.
    """
    data = load_all()
    industry_lower = industry.strip().lower()

    matching_accounts = {
        a["Name"] for a in data["accounts"]
        if a.get("Industry", "").strip().lower() == industry_lower
    }
    if not matching_accounts:
        return f"(no accounts found with industry='{industry}')"

    counts = []
    for f in data["feature_requests"]:
        requesting = [n.strip() for n in f.get("Accounts Requesting", "").split(",")]
        overlap = [n for n in requesting if n in matching_accounts]
        if overlap:
            counts.append((len(overlap), f["ID"], f["Title"], f.get("Status", ""), overlap))

    counts.sort(key=lambda x: x[0], reverse=True)
    top = counts[:top_n]

    lines = [f"Accounts in '{industry}' industry: {', '.join(sorted(matching_accounts))}", ""]
    lines.append(f"Feature requests ranked by number of requesting {industry} accounts:")
    for rank_count, fid, title, status, accounts in top:
        lines.append(f"[{fid}] \"{title}\" (status={status}) - requested by {rank_count} {industry} account(s): {', '.join(accounts)}")
    return "\n".join(lines)


def get_known_industries() -> list[str]:
    data = load_all()
    return sorted({a.get("Industry", "").strip() for a in data["accounts"] if a.get("Industry")})


if __name__ == "__main__":
    test_query = "accounts that requested offline mission caching"
    result = retrieve_customer_context(test_query)
    print(result[:2000])
    print(f"\n...\nTotal retrieved context: {len(result):,} chars (~{len(result)//4:,} tokens)")

    print("\n\n=== Aggregation test: agriculture ===")
    print(aggregate_features_by_industry("agriculture"))
