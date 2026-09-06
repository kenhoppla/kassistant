"""What the embedding request actually looks like on the wire.

Needs a real Home Assistant only for its aiohttp session; the property under
test is the request body Ollama receives.
"""

from __future__ import annotations

import json

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    AiohttpClientMockResponse,
)


async def test_keep_alive_is_sent(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """The request Ollama receives must carry keep_alive."""
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path.cwd() / "custom_components"))
    from homeassistant.helpers.aiohttp_client import async_get_clientsession
    from kassistant.embeddings import KEEP_ALIVE, OllamaEmbeddings

    seen = {}

    async def respond(method, url, data):
        seen.update(json.loads(data) if isinstance(data, str | bytes) else data)
        return AiohttpClientMockResponse(method, url, json={"embeddings": [[1.0, 0.0]]})

    aioclient_mock.post("http://x:11434/api/embed", side_effect=respond)
    await OllamaEmbeddings(
        async_get_clientsession(hass), "http://x:11434", "m"
    ).embed_one("hi")

    assert seen["keep_alive"] == KEEP_ALIVE
