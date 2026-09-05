"""Normalisation: strip the noise, keep the meaning."""

from __future__ import annotations

import pytest
from module_loader import load

normalize = load("text").normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Turn the light on!", "turn the light on"),
        ("  double   spaces  ", "double spaces"),
        ("Living room, please.", "living room please"),
        ("SHOUTING", "shouting"),
        ("", ""),
        ("...", ""),
    ],
)
def test_strips_noise(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_accents_are_preserved() -> None:
    """The embedding model copes better with accents than with ASCII folding."""
    assert normalize("Küche wärmer") == "küche wärmer"


def test_same_meaning_same_form() -> None:
    assert normalize("Light on!") == normalize("light on")
