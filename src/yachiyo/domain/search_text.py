"""Shared deterministic lexical normalization for FTS indexing and queries."""

from __future__ import annotations

import re

_LATIN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]+")
_CJK = re.compile(r"[\u3400-\u9fff]+")


def search_terms(text: str, *, limit: int = 64) -> tuple[str, ...]:
    terms = [token.casefold() for token in _LATIN.findall(text)]
    for sequence in _CJK.findall(text):
        terms.extend(sequence)
        if len(sequence) > 1:
            terms.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tuple(dict.fromkeys(terms))[:limit]


def fts_document(text: str) -> str:
    terms = search_terms(text, limit=max(64, min(len(text) * 2, 100_000)))
    return f"{text}\n{' '.join(terms)}" if terms else text
