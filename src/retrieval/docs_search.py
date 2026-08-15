"""
Live retrieval over docs.flytbase.com and releases.flytbase.com.

Both sites are GitBook-hosted and expose:
  - /llms.txt          -> a full page index: "- [Title](url.md): description"
  - <page-url>.md       -> that page's content as raw markdown

Strategy: fetch the (small) llms.txt index, keyword-match the query against
each entry's title+description to shortlist candidate pages, then fetch the
full .md content of only the top matches. Nothing is stored on disk - the
index is cached in memory for a short TTL purely to avoid re-fetching it on
every keystroke in one session, and every answer still reflects a fetch
made in that session, not a static scrape.
"""

import re
import time
import requests

DOCS_BASE = "https://docs.flytbase.com"
RELEASES_BASE = "https://releases.flytbase.com"

_INDEX_CACHE: dict[str, tuple[float, list[dict]]] = {}
_INDEX_TTL_SECONDS = 300  # re-fetch the page index at most every 5 minutes

_LINE_RE = re.compile(r"^- \[(?P<title>[^\]]+)\]\((?P<url>[^)]+)\)(?::\s*(?P<desc>.*))?$")


def _fetch_index(base_url: str) -> list[dict]:
    """Fetches and parses llms.txt into a list of {title, url, desc}."""
    now = time.time()
    cached = _INDEX_CACHE.get(base_url)
    if cached and (now - cached[0]) < _INDEX_TTL_SECONDS:
        return cached[1]

    resp = requests.get(f"{base_url}/llms.txt", timeout=15)
    resp.raise_for_status()
    entries = []
    for line in resp.text.splitlines():
        m = _LINE_RE.match(line.strip())
        if m:
            entries.append({
                "title": m.group("title").strip(),
                "url": m.group("url").strip(),
                "desc": (m.group("desc") or "").strip(),
            })
    _INDEX_CACHE[base_url] = (now, entries)
    return entries


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 2}


def _score_entry(query_tokens: set[str], entry: dict) -> int:
    entry_tokens = _tokenize(entry["title"] + " " + entry["desc"])
    return len(query_tokens & entry_tokens)


def _shortlist(query: str, entries: list[dict], top_k: int) -> list[dict]:
    query_tokens = _tokenize(query)
    scored = [(_score_entry(query_tokens, e), e) for e in entries]
    scored = [(s, e) for s, e in scored if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]


def _fetch_page_md(url: str) -> str:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.text


def retrieve_docs_context(query: str, top_k_per_site: int = 2) -> str:
    """
    Live-fetches the most relevant pages from docs.flytbase.com and
    releases.flytbase.com for the given query. Returns a citation-ready
    text block with each page's source URL.
    """
    sections = []
    for label, base in [("PRODUCT DOCS", DOCS_BASE), ("RELEASE NOTES", RELEASES_BASE)]:
        try:
            index = _fetch_index(base)
        except requests.RequestException as e:
            sections.append(f"=== {label} (unavailable: {e}) ===")
            continue

        matches = _shortlist(query, index, top_k_per_site)
        if not matches:
            continue

        sections.append(f"=== {label} ===")
        for entry in matches:
            try:
                content = _fetch_page_md(entry["url"])
            except requests.RequestException:
                continue
            # Trim very long pages to keep the prompt lean
            trimmed = content[:1500]
            sections.append(f"[SOURCE: {entry['url']}]\nTitle: {entry['title']}\n{trimmed}")

    return "\n\n".join(sections) if sections else "(no relevant documentation found)"


if __name__ == "__main__":
    result = retrieve_docs_context("mission scheduling recurring missions")
    print(result[:3000])
    print(f"\n...\nTotal retrieved docs context: {len(result):,} chars")
