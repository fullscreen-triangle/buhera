"""
Intent -> vaHera translator.

Three backends:
  - OpenAIBackend: uses the OpenAI Chat Completions API
  - OllamaBackend: uses a local Ollama server
  - StubTranslator: hand-written pattern matcher for offline smoke tests
"""
from __future__ import annotations

import os
import re
import json
from typing import Protocol


SYSTEM_PROMPT = """\
You translate natural-language scientific queries into vaHera — the internal
declarative language of the Buhera research operating system.

vaHera grammar (one statement per line). Two equivalent notations are
available and may be mixed freely — prefer the scientific-statement forms,
since they read as the sentences a scientist would write in a methods
section:

  Primitive form                          Scientific-statement form
  ---------------------------------------  ---------------------------------------
  describe <name> with "<text>"           observed <name> as "<text>"
  describe <name> with "<text>" + resolve <name>  hypothesize <name>: "<text>"
  spawn <program> from <name>             run <program> on <name>
  navigate to penultimate + complete trajectory   to completion
  memory find nearest "<text>" k=<n>      compare <name> to "<text>" [k=<n>]
  memory store "<name>" = "<text>"        record "<name>" = "<text>"
  controller verify                       check consistency
  demon sort                              rank by category

Rules:
  - Emit ONLY vaHera statements, one per line. No prose, no explanation.
  - Prefer the scientific-statement form when the query reads like a
    scientific claim, observation, procedure, or comparison.
  - Use observed/describe before hypothesize/resolve/run/spawn for novel
    entities.
  - For lookup queries, emit: observed .../hypothesize .../run .../to completion.
  - For retrieval queries, emit: compare <name> to "<query>" k=<n>.
  - Keep programs short (3-6 lines typical).

Example:
  User: "What is the boiling point of ethanol?"
  vaHera:
    hypothesize ethanol_bp: "boiling point of ethanol, C2H5OH, small alcohol"
    run query on ethanol_bp
    to completion
"""


class TranslatorBackend(Protocol):
    def translate(self, intent: str) -> str: ...


# ─── OpenAI ─────────────────────────────────────────────────────────

class OpenAIBackend:
    def __init__(self, api_key: str | None = None,
                 model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=self.api_key)
        except ImportError as e:
            raise RuntimeError("pip install openai") from e

    def translate(self, intent: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": intent},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()


# ─── Ollama ─────────────────────────────────────────────────────────

class OllamaBackend:
    def __init__(self, model: str = "llama3.2",
                 host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")
        try:
            import requests  # type: ignore
            self._requests = requests
        except ImportError as e:
            raise RuntimeError("pip install requests") from e

    def translate(self, intent: str) -> str:
        url = f"{self.host}/api/chat"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": intent},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        r = self._requests.post(url, json=body, timeout=60)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


# ─── Stub (offline) ─────────────────────────────────────────────────

class StubTranslator:
    """Hand-written pattern matcher for offline testing."""

    PATTERNS = [
        (re.compile(r"boiling point of (\w+)", re.I),
         lambda m: f'hypothesize {m.group(1)}_bp: "boiling point of {m.group(1)}"\n'
                   f'run query on {m.group(1)}_bp\n'
                   f'to completion\n'
                   f'check consistency'),
        (re.compile(r"what is (\w+)", re.I),
         lambda m: f'hypothesize {m.group(1)}: "{m.group(1)}"\n'
                   f'run query on {m.group(1)}\n'
                   f'to completion'),
        (re.compile(r"compare (.+?) to (.+)", re.I),
         lambda m: f'compare {m.group(1).strip()} to "{m.group(2).strip()}" k=5'),
        (re.compile(r"find (.+)", re.I),
         lambda m: f'compare query to "{m.group(1)}" k=5'),
        (re.compile(r"store (.+)", re.I),
         lambda m: f'record "note_{abs(hash(m.group(1)))%10000}" = "{m.group(1)}"'),
    ]

    def translate(self, intent: str) -> str:
        for pat, tmpl in self.PATTERNS:
            m = pat.search(intent.strip())
            if m:
                return tmpl(m)
        # default: treat as a general query
        safe = intent.replace('"', "'")
        return (f'hypothesize query: "{safe}"\n'
                f'run q on query\n'
                f'to completion')


# ─── front end ──────────────────────────────────────────────────────

class IntentTranslator:
    """Unified translator with automatic backend selection."""

    def __init__(self, backend: TranslatorBackend | None = None):
        self.backend = backend or self._auto_select()

    @staticmethod
    def _auto_select() -> TranslatorBackend:
        # Prefer OpenAI if key is set
        if os.environ.get("OPENAI_API_KEY"):
            try:
                return OpenAIBackend()
            except Exception:
                pass
        # Try Ollama
        try:
            backend = OllamaBackend()
            # probe for availability
            import requests  # type: ignore
            r = requests.get(f"{backend.host}/api/tags", timeout=2)
            if r.status_code == 200:
                return backend
        except Exception:
            pass
        # Fall back to stub
        return StubTranslator()

    def translate(self, intent: str) -> str:
        raw = self.backend.translate(intent)
        # strip any accidental fencing the model may have added
        lines = [ln for ln in raw.split("\n")
                 if ln.strip() and not ln.strip().startswith("```")]
        return "\n".join(lines)
