"""
Thin LLM client wrapper. Swap providers by editing only this file.
"""

import os
from groq import Groq

_MODEL = "llama-3.3-70b-versatile"


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env or Streamlit secrets.")
    return Groq(api_key=api_key)


def chat_completion(system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content
