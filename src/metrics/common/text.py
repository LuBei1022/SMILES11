"""Deterministic English and Russian text helpers."""

from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"[\w\u0400-\u04ff]+", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|[\r\n]+")

STOP_WORDS = {
    "en": {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "for",
        "from", "how", "i", "in", "is", "it", "my", "of", "on", "the",
        "to", "with", "you", "your",
    },
    "ru": {
        "а", "без", "в", "вы", "для", "и", "из", "как", "к", "ли", "мой",
        "на", "не", "о", "по", "с", "у", "что", "это", "я",
    },
}


def tokenize(text: str, language: str | None = None) -> list[str]:
    if not isinstance(text, str):
        return []
    stop_words = STOP_WORDS.get(language or "", set())
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in stop_words]


def split_sentences(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(text) if sentence.strip()]
