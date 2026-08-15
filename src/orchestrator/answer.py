"""
Orchestrator: decides which source(s) a question needs, retrieves relevant
context from each, and generates a grounded, citation-enforced answer.

Routing is done with a cheap keyword heuristic rather than a separate LLM
call, to keep latency low for the live demo. This is intentionally simple -
swap in an LLM-based classifier later if time allows.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import re
from retrieval.customer_search import (
    retrieve_customer_context,
    aggregate_features_by_industry,
    get_known_industries,
)
from retrieval.docs_search import retrieve_docs_context
from llm_client import chat_completion

# "tier" removed - it's ambiguous (account tier vs. plan/pricing tier) and
# was causing customer-only questions to also trigger a needless docs fetch.
# "feature" removed too - it's in the customer data's own vocabulary
# (feature_requests), so it was making customer-only questions needlessly
# fetch docs as well.
_DOCS_KEYWORDS = {
    "docs", "documentation", "release", "supported", "does flytbase",
    "how do i", "how to", "changelog", "shipped", "available",
    "capability", "capable",
}

_AGGREGATE_TRIGGER_WORDS = {"most requested", "most common", "top feature", "most popular"}

SYSTEM_PROMPT = """You are a knowledge base agent for a FlytBase Solutions Engineer.
You answer questions using ONLY the CUSTOMER DATA and PRODUCT DOCS/RELEASE NOTES
context provided below - never from general knowledge or assumption.

Rules you must follow:
1. Every factual claim must be followed by a citation in brackets: either a
   record ID from the customer data (e.g. [ISS-0042], [acct-007], [FEAT-0012])
   or a doc source URL (e.g. [SOURCE: https://docs.flytbase.com/...]).
2. If the provided context does not contain enough information to answer,
   say so explicitly: "I don't have enough information in the available
   customer data / documentation to answer that." Do not guess.
3. If a question needs both customer data and product docs, combine them
   explicitly in your answer and cite both.
4. If you notice a contradiction between customer data and docs/release
   notes (e.g. a feature marked as still-requested in customer data but
   already shipped per the release notes), point it out clearly.
5. Be concise and direct. No filler.
6. Cite each record only once, inline, right after the claim it supports -
   never repeat a bracketed ID again later in the same sentence or as a
   trailing list at the end of a paragraph.
   WRONG: "Eastbrook Energy [acct-029], Ashgrove Rail Services [acct-031]
   are at risk [acct-029], [acct-031]."
   RIGHT: "Eastbrook Energy [acct-029] and Ashgrove Rail Services [acct-031]
   are at risk."
"""


def _route(query: str) -> tuple[bool, bool]:
    """Returns (needs_customer_data, needs_docs). Defaults to both if unsure."""
    q_lower = query.lower()
    needs_docs = any(kw in q_lower for kw in _DOCS_KEYWORDS)
    needs_customer = any(
        kw in q_lower for kw in
        ["account", "customer", "issue", "ticket", "task", "meeting", "arr", "industry", "tier"]
    )
    if not needs_docs and not needs_customer:
        # ambiguous - fetch both, cheap insurance for combined questions
        return True, True
    return needs_customer or not needs_docs, needs_docs or not needs_customer


def _detect_industry_aggregation(query: str) -> str | None:
    """
    If the question asks for "most requested features" (or similar) within
    a specific industry, returns that industry name so we can run a real
    computed aggregation instead of letting the LLM eyeball a ranking from
    a flat list of records.
    """
    q_lower = query.lower()
    if not any(trigger in q_lower for trigger in _AGGREGATE_TRIGGER_WORDS):
        return None
    for industry in get_known_industries():
        if industry.lower().replace("_", " ") in q_lower or industry.lower() in q_lower:
            return industry
    return None


def answer_question(query: str) -> dict:
    needs_customer, needs_docs = _route(query)

    context_parts = []
    sources_used = []

    industry = _detect_industry_aggregation(query)
    if industry:
        agg_result = aggregate_features_by_industry(industry)
        context_parts.append(
            "PRECOMPUTED AGGREGATION (already correctly ranked and counted - "
            "report these numbers directly, do not recount or re-rank):\n" + agg_result
        )
        sources_used.append("customer_data (aggregated)")
        needs_customer = False  # avoid also dumping the raw unranked records
        needs_docs = False  # aggregation questions are customer-only by default

    if needs_customer:
        customer_ctx = retrieve_customer_context(query)
        context_parts.append("CUSTOMER DATA:\n" + customer_ctx)
        sources_used.append("customer_data")

    if needs_docs:
        docs_ctx = retrieve_docs_context(query)
        context_parts.append("PRODUCT DOCS / RELEASE NOTES:\n" + docs_ctx)
        sources_used.append("live_docs")

    full_context = "\n\n".join(context_parts)
    user_message = f"CONTEXT:\n{full_context}\n\nQUESTION: {query}"

    answer_text = chat_completion(SYSTEM_PROMPT, user_message)

    return {
        "answer": answer_text,
        "sources_used": sources_used,
    }


if __name__ == "__main__":
    result = answer_question(
        "Which accounts requested offline mission caching, and does the "
        "current product already support it according to the docs?"
    )
    print(result["answer"])
    print("\nSources used:", result["sources_used"])
