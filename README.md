# FlytBase knowledge base agent

A conversational knowledge base agent that answers questions over the
FlytBase synthetic customer-data corpus (accounts, issues, feature requests,
tasks, meeting notes) and live FlytBase product documentation and release
notes (docs.flytbase.com, releases.flytbase.com) - combining both when a
question needs it, and citing a specific record or doc page for every claim.

## How it works

- **Customer data**: the 5 corpus markdown files are parsed fresh on every
  query (`src/ingest/parse_corpus.py`). A lightweight keyword filter
  (`src/retrieval/customer_search.py`) narrows ~1,800 records down to the
  ones relevant to the question, so no vector database is needed and no
  rebuild step is required when the source files change - just edit them.
- **Live docs**: `src/retrieval/docs_search.py` fetches `llms.txt` from
  docs.flytbase.com and releases.flytbase.com (a GitBook page index), keyword-
  matches the query against it, and live-fetches the matching pages' markdown
  content. Nothing is scraped or stored - every answer reflects a fetch made
  in that session.
- **Orchestration**: `src/orchestrator/answer.py` routes each question to the
  source(s) it needs, builds a combined context block, and prompts the LLM
  (Groq / Llama 3.3 70B) with strict citation-enforcement rules: every claim
  must cite a record ID or doc URL, and the model must say so explicitly when
  it doesn't have enough information.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your Groq API key
streamlit run src/app.py
```

Get a free Groq API key at https://console.groq.com.

## Deployment

Push to GitHub, connect the repo on
[Streamlit Community Cloud](https://streamlit.io/cloud), and add
`GROQ_API_KEY` under App Settings > Secrets.

## Project structure

```
kb-agent/
├── data/raw/              # the 5 customer-corpus markdown files
├── src/
│   ├── ingest/parse_corpus.py     # parses the .md files into records
│   ├── retrieval/
│   │   ├── customer_search.py     # keyword filter over customer records
│   │   └── docs_search.py         # live fetch of docs/release notes
│   ├── orchestrator/answer.py     # routing + citation-enforced answers
│   ├── llm_client.py              # Groq API wrapper
│   └── app.py                     # Streamlit chat UI
└── requirements.txt
```
