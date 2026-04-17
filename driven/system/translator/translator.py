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

vaHera grammar (one statement per line):
  describe <name> with "<text>"
  resolve <name>
  spawn <program-name> from <name>
  navigate to penultimate
  complete trajectory
  memory store "<name>" = "<text>"
  memory find nearest "<text>" k=<n>
  demon sort
  controller verify

Rules:
  - Emit ONLY vaHera statements, one per line. No prose, no explanation.
  - Use describe before resolve/spawn for novel entities.
  - For lookup queries, emit: describe/resolve/spawn/navigate/complete.
  - For retrieval queries, emit: memory find nearest "<query>" k=<n>.
  - Keep programs short (3-6 lines typical).

Example:
  User: "What is the boiling point of ethanol?"
  vaHera:
    describe ethanol_bp with "boiling point of ethanol, C2H5OH, small alcohol"
    resolve ethanol_bp
    spawn query from ethanol_bp
    navigate to penultimate
    complete trajectory
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
         lambda m: f'describe {m.group(1)}_bp with "boiling point of {m.group(1)}"\n'
                   f'resolve {m.group(1)}_bp\n'
                   f'spawn query from {m.group(1)}_bp\n'
                   f'navigate to penultimate\n'
                   f'complete trajectory\n'
                   f'controller verify'),
        (re.compile(r"what is (\w+)", re.I),
         lambda m: f'describe {m.group(1)} with "{m.group(1)}"\n'
                   f'resolve {m.group(1)}\n'
                   f'spawn query from {m.group(1)}\n'
                   f'navigate to penultimate\n'
                   f'complete trajectory'),
        (re.compile(r"find (.+)", re.I),
         lambda m: f'memory find nearest "{m.group(1)}" k=5'),
        (re.compile(r"store (.+)", re.I),
         lambda m: f'memory store "note_{abs(hash(m.group(1)))%10000}" = "{m.group(1)}"'),
    ]

    def translate(self, intent: str) -> str:
        for pat, tmpl in self.PATTERNS:
            m = pat.search(intent.strip())
            if m:
                return tmpl(m)
        # default: treat as a general query
        safe = intent.replace('"', "'")
        return (f'describe query with "{safe}"\n'
                f'resolve query\n'
                f'spawn q from query\n'
                f'navigate to penultimate\n'
                f'complete trajectory')


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
