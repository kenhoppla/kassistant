"""Normalise sentences before comparing them.

Deliberately restrained: accents and words are left alone, because the embedding
model handles them better than a stripped-down version. We only remove what is
pure noise -- punctuation, capitalisation and repeated whitespace.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Bring a sentence into a comparable form."""
    text = unicodedata.normalize("NFC", text)
    text = text.casefold()
    text = _PUNCTUATION.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()
