"""Runtime objects attached to a config entry."""

from __future__ import annotations

from dataclasses import dataclass

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
