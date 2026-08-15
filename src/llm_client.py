"""
Thin LLM client wrapper. Swap providers by editing only this file.

Groq enforces a separate daily token quota PER MODEL, not one shared pool.
llama-3.1-8b-instant is the primary model precisely because it is cheap
enough to comfortably survive a full demo/rehearsal cycle on the free tier,
and - critically - it draws from a completely separate quota than
llama-3.3-70b-versatile, so exhausting one model's daily limit does not
block the other. If the primary model's quota is ever hit anyway, we
automatically retry once against the fallback model before giving up.
"""

import os
from groq import Groq, RateLimitError

_PRIMARY_MODEL = "llama-3.1-8b-instant"
_FALLBACK_MODEL = "llama-3.3-70b-versatile"


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env or Streamlit secrets.")
    return Groq(api_key=api_key)


def _call(client: Groq, model: str, system_prompt: str, user_message: str, temperature: float) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def chat_completion(system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
    client = get_client()
    try:
        return _call(client, _PRIMARY_MODEL, system_prompt, user_message, temperature)
    except RateLimitError:
        try:
            return _call(client, _FALLBACK_MODEL, system_prompt, user_message, temperature)
        except RateLimitError:
            raise RuntimeError(
                "Both Groq models are at their daily free-tier token limit right now. "
                "Wait for the quota to reset, or add billing at "
                "console.groq.com/settings/billing to raise it."
            )
