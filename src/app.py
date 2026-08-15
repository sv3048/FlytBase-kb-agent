"""
Streamlit chat UI for the FlytBase knowledge base agent.

Run locally:   streamlit run src/app.py
Deploy:        push to GitHub, connect the repo on Streamlit Community Cloud,
                add GROQ_API_KEY under App Settings > Secrets.
"""

import os
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orchestrator.answer import answer_question

# Support both a local .env (via python-dotenv) and Streamlit Cloud secrets
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if "GROQ_API_KEY" not in os.environ:
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except FileNotFoundError:
        pass  # no secrets.toml locally - fine, we're using .env instead

st.set_page_config(page_title="FlytBase Knowledge Base Agent", page_icon="◈", layout="centered")

# ---------------------------------------------------------------------------
# Design system
#
# Palette is drawn from flight-instrumentation displays (amber caution
# lighting + teal telemetry/status readouts on near-black), because the
# product itself is a drone-ops knowledge tool - not a generic SaaS accent.
# Monospace is reserved specifically for citations and record IDs, so a
# grounded fact is visually distinguishable from prose at a glance, which
# doubles as reinforcement of the "always cite, never guess" requirement.
# ---------------------------------------------------------------------------
_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
--bg-deep:#0A0E14; --bg-panel:#121822; --bg-panel-2:#161E2A;
--border:#232E3D; --amber:#FFB020; --teal:#2DD4C8;
--text:#E8EDF4; --muted:#8494A8;
}
.stApp{
background:
radial-gradient(ellipse 900px 500px at 15% -10%, rgba(255,176,32,0.07), transparent 60%),
radial-gradient(ellipse 900px 500px at 100% 0%, rgba(45,212,200,0.06), transparent 60%),
var(--bg-deep);
}
html, body, [class*="css"]{ font-family:'Inter', sans-serif; color:var(--text); }
.kb-kicker{
display:flex; align-items:center; gap:.5rem;
font-family:'JetBrains Mono', monospace; font-size:.72rem; letter-spacing:.08em;
color:var(--teal); text-transform:uppercase; margin-bottom:.9rem;
}
.kb-dot{
width:7px; height:7px; border-radius:50%; background:var(--teal);
box-shadow:0 0 0 0 rgba(45,212,200,.6); animation:kb-pulse 1.8s infinite;
}
@keyframes kb-pulse{
0%{ box-shadow:0 0 0 0 rgba(45,212,200,.55); }
70%{ box-shadow:0 0 0 7px rgba(45,212,200,0); }
100%{ box-shadow:0 0 0 0 rgba(45,212,200,0); }
}
h1{
font-family:'Space Grotesk', sans-serif !important; font-weight:700 !important;
letter-spacing:-.01em; color:var(--text) !important; margin-bottom:.3rem !important;
}
[data-testid="stCaptionContainer"] p{ color:var(--muted) !important; font-size:.95rem; }
.kb-section-label{
font-family:'Space Grotesk', sans-serif; font-weight:600; font-size:.95rem;
color:var(--text); margin:1.6rem 0 .1rem 0;
}
div[data-testid="stButton"] button{
background:var(--bg-panel) !important; border:1px solid var(--border) !important;
border-radius:10px !important; color:var(--text) !important;
font-family:'Inter', sans-serif !important; font-weight:500 !important;
padding:.85rem .9rem !important; min-height:5.4rem; white-space:normal !important;
text-align:left !important; line-height:1.35 !important;
transition:border-color .15s ease, background .15s ease;
}
div[data-testid="stButton"] button:hover{
border-color:var(--amber) !important; background:var(--bg-panel-2) !important; color:var(--amber) !important;
}
div[data-testid="stButton"] button:active,
div[data-testid="stButton"] button:focus:not(:active){
color:var(--text) !important; box-shadow:none !important;
}
.kb-tag{
font-family:'JetBrains Mono', monospace; font-size:.68rem; font-weight:500; color:var(--amber);
background:rgba(255,176,32,0.08); border:1px solid rgba(255,176,32,0.25);
border-radius:5px; padding:.14rem .5rem; display:inline-block; margin-bottom:.5rem;
text-transform:uppercase; letter-spacing:.06em;
}
.kb-tag.teal{ color:var(--teal); background:rgba(45,212,200,0.08); border-color:rgba(45,212,200,0.25); }
[data-testid="stChatMessage"]{
background:var(--bg-panel) !important; border:1px solid var(--border) !important;
border-radius:12px !important;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li{
font-family:'Inter', sans-serif; font-size:.96rem; line-height:1.55; color:var(--text);
}
code{
font-family:'JetBrains Mono', monospace !important; font-size:.82em !important;
background:rgba(255,176,32,0.10) !important; color:var(--amber) !important;
border:1px solid rgba(255,176,32,0.25) !important; border-radius:5px !important;
padding:.08rem .35rem !important;
}
[data-testid="stChatInput"]{
background:var(--bg-panel) !important; border:1px solid var(--border) !important;
border-radius:12px !important;
}
[data-testid="stChatInput"] *{
background-color:var(--bg-panel) !important;
}
[data-testid="stChatInput"] textarea{
color:var(--text) !important; font-family:'Inter', sans-serif !important;
}
[data-testid="stChatInput"] textarea::placeholder{ color:var(--muted) !important; opacity:1 !important; }
[data-testid="stChatInput"] button svg{ fill:var(--amber) !important; }
[data-testid="stAlert"]{
background:var(--bg-panel) !important; border:1px solid rgba(255,176,32,0.35) !important;
border-radius:10px !important;
}
</style>
"""
st.markdown("\n".join(line.strip() for line in _CSS.splitlines()), unsafe_allow_html=True)

st.markdown(
    '<div class="kb-kicker"><span class="kb-dot"></span>'
    'LIVE &middot; customer_data + docs.flytbase.com + releases.flytbase.com</div>',
    unsafe_allow_html=True,
)
st.title("FlytBase knowledge base agent")
st.caption(
    "Ask about customer accounts, issues, feature requests, tasks, and meeting "
    "notes - or about live FlytBase product docs and release notes. Combined "
    "questions are supported."
)

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY is not set. Add it to a local `.env` file, or under "
        "**App settings > Secrets** if this is running on Streamlit Community Cloud."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None

EXAMPLE_QUESTIONS = [
    (
        "Customer data",
        "Which accounts are on the enterprise tier and marked at_risk?",
    ),
    (
        "Live docs",
        "Does FlytBase support scheduled recurring missions?",
    ),
    (
        "Customer data + live docs",
        "Which accounts requested offline mission caching, and does FlytBase "
        "already support it according to the docs?",
    ),
]

if not st.session_state.messages:
    st.markdown('<div class="kb-section-label">Try one of the three demo question types</div>', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, (tag, question) in zip(cols, EXAMPLE_QUESTIONS):
        with col:
            tag_class = "kb-tag" if "+" not in tag else "kb-tag teal"
            st.markdown(f'<span class="{tag_class}">{tag}</span>', unsafe_allow_html=True)
            if st.button(question, key=f"example_{tag}", use_container_width=True):
                st.session_state.queued_prompt = question

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

typed_prompt = st.chat_input("Ask a question...")
prompt = typed_prompt or st.session_state.queued_prompt
st.session_state.queued_prompt = None

_CITATION_RE = re.compile(r"\[(?!\s*\])((?:SOURCE:\s*)?[^\[\]]+)\](?!\()")

_SOURCE_LABELS = {
    "customer_data": "Customer data",
    "live_docs": "Live docs",
}


def _display_source(key: str) -> str:
    aggregated = "(aggregated)" in key
    base = key.replace("(aggregated)", "").strip()
    label = _SOURCE_LABELS.get(base, base.replace("_", " ").capitalize())
    return f"{label} · aggregated" if aggregated else label


def _style_citations(text: str) -> str:
    return _CITATION_RE.sub(lambda m: f"`[{m.group(1)}]`", text)


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering..."):
            try:
                result = answer_question(prompt)
                answer = _style_citations(result["answer"])
                sources = " ".join(f"`{_display_source(s)}`" for s in result["sources_used"])
                sources_note = f"\n\n*Sources consulted:* {sources}"
                st.markdown(answer + sources_note)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer + sources_note}
                )
            except Exception as e:
                print(f"[kb-agent] answer_question failed: {e!r}")
                error_msg = (
                    "Something went wrong generating that answer. If this is a "
                    "rate limit, it should resolve shortly - try again in a moment."
                )
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

    if not typed_prompt:
        st.rerun()
