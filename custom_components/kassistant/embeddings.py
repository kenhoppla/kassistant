"""Turn sentences into vectors.

We do not compute anything ourselves. The sentence goes to an Ollama instance,
which translates it into a row of numbers. That keeps this integration free of
heavyweight dependencies -- which matters, because Home Assistant OS runs on
Alpine Linux, where no suitable wheels exist for onnxruntime or torch.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from aiohttp import ClientError, ClientSession, ClientTimeout

_LOGGER = logging.getLogger(__name__)

_TIMEOUT = ClientTimeout(total=30)


class EmbeddingError(Exception):
    """The embedding service was unreachable or returned something unusable."""


class OllamaEmbeddings:
    """Thin client for Ollama's /api/embed endpoint."""

    def __init__(self, session: ClientSession, base_url: str, model: str) -> None:
        self._session = session
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._model = model
        self._dimension: int | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int | None:
        """Length of the vectors. Only known after the first call."""
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Translate sentences into a matrix of shape (count, dimension).

        Rows are scaled to unit length. That makes the dot product of two rows
        the cosine similarity already -- saving a division on every later search.
        """
        if not texts:
            raise ValueError("embed() called without texts")

        payload = {"model": self._model, "input": list(texts)}

        try:
            async with self._session.post(
                self._url, json=payload, timeout=_TIMEOUT
            ) as response:
                if response.status != 200:
                    body = (await response.text())[:200]
                    raise EmbeddingError(
                        f"{self._url} returned HTTP {response.status}: {body}"
                    )
                data = await response.json()
        except ClientError as err:
            raise EmbeddingError(f"{self._url} is unreachable: {err}") from err
        except TimeoutError as err:
            raise EmbeddingError(f"{self._url} did not answer in time") from err
        except ValueError as err:
            # A 200 response whose body is not JSON. Usually something other
            # than Ollama listening on that address.
            raise EmbeddingError(f"{self._url} returned no valid JSON: {err}") from err

        raw = data.get("embeddings") if isinstance(data, dict) else None
        if not raw:
            raise EmbeddingError(
                f"Response contained no vectors. Does the model {self._model!r} exist?"
            )

        # Anything malformed past this point must surface as an EmbeddingError
        # too -- callers fall back to the slow path on that, but a raw ValueError
        # would tear down the whole conversation.
        try:
            matrix = np.asarray(raw, dtype=np.float32)
        except (ValueError, TypeError) as err:
            raise EmbeddingError(f"Vectors are not numeric: {err}") from err

        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise EmbeddingError(
                f"Unexpected response shape {matrix.shape} for {len(texts)} texts"
            )

        matrix = _normalize_rows(matrix)

        if self._dimension is None:
            self._dimension = int(matrix.shape[1])
            _LOGGER.debug(
                "Embedding model %s returns %d dimensions",
                self._model,
                self._dimension,
            )
        elif matrix.shape[1] != self._dimension:
            raise EmbeddingError(
                f"Model {self._model} returned {matrix.shape[1]} instead of "
                f"{self._dimension} dimensions"
            )

        return matrix

    async def embed_one(self, text: str) -> np.ndarray:
        """Convenience for the common case: exactly one sentence."""
        return (await self.embed([text]))[0]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit length; zero rows stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix
