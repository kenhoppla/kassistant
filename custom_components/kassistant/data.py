"""Runtime objects attached to a config entry."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .embeddings import OllamaEmbeddings
from .learn import ActionRecorder
from .router import Router
from .store import Store


@dataclass(slots=True)
class KassistantData:
    """Objects shared across one configured kassistant instance."""

    store: Store
    embeddings: OllamaEmbeddings
    router: Router
    recorder: ActionRecorder
    fallback_agent: str
    # Seeding reads every exposed entity and talks to Ollama in batches. Running
    # two of those at once would only duplicate work and hammer the service.
    seeding: asyncio.Lock = field(default_factory=asyncio.Lock)
