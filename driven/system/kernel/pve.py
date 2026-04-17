"""
Proof Validation Engine (lightweight).

For the working kernel we perform type/sanity checks on vaHera
statements, not full Lean 4 proofs. The structure is in place so a
real Lean backend can plug in later.
"""
from __future__ import annotations

from typing import Any


class PVEError(Exception):
    pass


class PVE:
    def __init__(self):
        self._verified = 0
        self._rejected = 0
        self._events: list[str] = []

    def verify(self, statement_type: str, payload: dict) -> bool:
        """
        Check a vaHera statement against structural constraints.
        Raises PVEError if invalid; returns True otherwise.
        """
        ok = True
        reason = ""

        if statement_type == "resolve":
            if "target" not in payload or not payload["target"]:
                ok, reason = False, "resolve requires target"
        elif statement_type == "navigate":
            if payload.get("mode") not in ("penultimate", "explicit"):
                ok, reason = False, "navigate requires mode"
        elif statement_type == "complete":
            if "s_penultimate" not in payload:
                ok, reason = False, "complete requires penultimate"
        elif statement_type == "memory_create":
            if "coord" not in payload:
                ok, reason = False, "memory create requires coord"
        elif statement_type in ("memory_read", "memory_write"):
            pass  # anything goes at this layer
        elif statement_type == "demon_sort":
            pass
        elif statement_type == "controller_verify":
            pass
        else:
            ok, reason = False, f"unknown statement {statement_type}"

        if ok:
            self._verified += 1
            self._events.append(f"PVE.verify {statement_type} OK")
            return True
        else:
            self._rejected += 1
            self._events.append(f"PVE.verify {statement_type} REJECTED: {reason}")
            raise PVEError(f"{statement_type}: {reason}")

    def stats(self) -> dict:
        return {"verified": self._verified, "rejected": self._rejected}

    def events(self) -> list[str]:
        return list(self._events)
