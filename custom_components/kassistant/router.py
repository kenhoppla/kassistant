"""The decision: do we know this sentence well enough to handle it ourselves?

The router may abstain at any point. That is by design. A wrong but confident
decision annoys users far more than a slow one -- the fallback agent catches
everything that falls through here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from homeassistant.core import HomeAssistant

from .embeddings import EmbeddingError, OllamaEmbeddings
from .store import Match, Store
from .text import normalize

_LOGGER = logging.getLogger(__name__)

# Confident hit in the card box -- kassistant may act on its own.
TIER_FASTPATH = "fastpath"
# No sufficiently similar sentence known -- pass it on.
TIER_ABSTAIN = "abstain"
# The embedding service was unreachable -- pass it on.
TIER_UNAVAILABLE = "unavailable"
# Mode "observe": we did not even look.
TIER_OBSERVE = "observe"


@dataclass(slots=True, frozen=True)
class Decision:
    """What the router made of a sentence."""

    tier: str
    score: float | None = None
    match: Match | None = None
    latency_ms: int | None = None

    @property
    def is_fastpath(self) -> bool:
        return self.tier == TIER_FASTPATH

    @property
    def action(self) -> dict[str, Any] | None:
        return self.match.action if self.match is not None else None


class Router:
    """Looks a sentence up in the card box."""

    def __init__(
        self,
        hass: HomeAssistant,
        store: Store,
        embeddings: OllamaEmbeddings,
        threshold: float,
    ) -> None:
        self._hass = hass
        self._store = store
        self._embeddings = embeddings
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def set_threshold(self, threshold: float) -> None:
        self._threshold = threshold

    async def decide(self, text: str, language: str | None) -> Decision:
        """Decide whether we can answer this sentence ourselves."""
        started = time.monotonic()
        norm = normalize(text)
        if not norm:
            return Decision(tier=TIER_ABSTAIN, latency_ms=0)

        try:
            vector = await self._embeddings.embed_one(norm)
        except EmbeddingError as err:
            _LOGGER.debug("Embedding failed, passing on: %s", err)
            return Decision(tier=TIER_UNAVAILABLE, latency_ms=_elapsed(started))
        except ValueError:
            return Decision(tier=TIER_ABSTAIN, latency_ms=_elapsed(started))

        match = await self._hass.async_add_executor_job(
            self._store.search, vector, language
        )
        latency = _elapsed(started)

        if match is None:
            return Decision(tier=TIER_ABSTAIN, latency_ms=latency)

        if match.score >= self._threshold:
            return Decision(
                tier=TIER_FASTPATH,
                score=match.score,
                match=match,
                latency_ms=latency,
            )

        return Decision(
            tier=TIER_ABSTAIN, score=match.score, match=match, latency_ms=latency
        )

    async def remember(
        self,
        *,
        utterance: str,
        language: str,
        action: dict[str, Any],
        source: str,
    ) -> int | None:
        """Store a sentence together with its action as a card."""
        norm = normalize(utterance)
        if not norm:
            return None

        embedding = None
        try:
            embedding = await self._embeddings.embed_one(norm)
        except (EmbeddingError, ValueError) as err:
            # Store it anyway -- the vector can be filled in later.
            _LOGGER.debug("Card stored without a vector: %s", err)

        return await self._hass.async_add_executor_job(
            _add_example,
            self._store,
            utterance,
            norm,
            language,
            action,
            source,
            embedding,
        )


def _add_example(store, utterance, norm, language, action, source, embedding):
    """Small detour, because add_example only takes keyword arguments."""
    return store.add_example(
        utterance=utterance,
        norm=norm,
        language=language,
        action=action,
        source=source,
        embedding=embedding,
    )


def _elapsed(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
