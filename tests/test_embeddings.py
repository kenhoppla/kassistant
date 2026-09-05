"""Vector normalisation.

The whole matching scheme rests on this: rows scaled to unit length make the
dot product equal to the cosine similarity. If normalisation is wrong, every
similarity score silently becomes meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest
from module_loader import load

normalize_rows = load("embeddings")._normalize_rows


def test_rows_become_unit_length() -> None:
    matrix = np.array([[3.0, 4.0], [1.0, 0.0], [5.0, 12.0]], dtype=np.float32)

    result = normalize_rows(matrix)

    assert np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)


def test_direction_is_preserved() -> None:
    matrix = np.array([[3.0, 4.0]], dtype=np.float32)

    result = normalize_rows(matrix)

    assert result[0] == pytest.approx([0.6, 0.8], abs=1e-6)


def test_zero_rows_stay_zero() -> None:
    """A division by zero here would poison the index with NaN."""
    matrix = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    result = normalize_rows(matrix)

    assert not np.isnan(result).any()
    assert result[0] == pytest.approx([0.0, 0.0])


def test_dot_product_of_identical_vectors_is_one() -> None:
    """This is the property store.search() relies on for its score."""
    matrix = normalize_rows(np.array([[2.0, 7.0, 1.0]], dtype=np.float32))

    assert float(matrix[0] @ matrix[0]) == pytest.approx(1.0, abs=1e-6)
