"""Seeding the card box from Home Assistant's own example sentences.

This is the part that makes kassistant useful on day one instead of after a
fortnight of talking to it, so it is worth proving against a real Home
Assistant: real intent data, a real entity registry, real exposure rules.
"""

from __future__ import annotations

import json

import pytest
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    AiohttpClientMockResponse,
)

DIMENSION = 8
LIGHT_NAME = "Ceiling Light"
LIGHT_ALIAS = "big light"


@pytest.fixture
def embed_mock(aioclient_mock: AiohttpClientMocker) -> AiohttpClientMocker:
    """Return one vector per input, the way Ollama does.

    Seeding sends batches, so a fixed single-vector answer would not do.
    """

    async def respond(method, url, data):
        payload = json.loads(data) if isinstance(data, str | bytes) else data
        count = len(payload["input"])
        return AiohttpClientMockResponse(
            method,
            url,
            json={
                "embeddings": [[1.0] + [0.0] * (DIMENSION - 1) for _ in range(count)]
            },
        )

    aioclient_mock.post("http://localhost:11434/api/embed", side_effect=respond)
    return aioclient_mock


@pytest.fixture
async def seeded_hass(
    hass: HomeAssistant, custom_integration, embed_mock
) -> MockConfigEntry:
    """A kassistant with one exposed light, set up but not yet seeded."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})

    registry = er.async_get(hass)
    entry = registry.async_get_or_create(
        "light", "demo", "ceiling", suggested_object_id="ceiling"
    )
    registry.async_update_entity(
        entry.entity_id, name=LIGHT_NAME, aliases={LIGHT_ALIAS}
    )
    hass.states.async_set(entry.entity_id, "off", {"friendly_name": LIGHT_NAME})
    async_expose_entity(hass, conversation.DOMAIN, entry.entity_id, True)

    config_entry = MockConfigEntry(
        domain="kassistant",
        data={
            "embed_url": "http://localhost:11434",
            "embed_model": "test-model",
            "fallback_agent": conversation.HOME_ASSISTANT_AGENT,
        },
        options={
            "mode": "observe",
            "threshold": 0.92,
            "learn": False,
            "learn_delay": 0,
        },
    )
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    return config_entry


async def test_exposed_names_include_aliases(
    hass: HomeAssistant, seeded_hass: MockConfigEntry
) -> None:
    """An alias exists because someone says that instead of the real name."""
    from kassistant.seed import async_exposed_names

    names = async_exposed_names(hass)

    assert LIGHT_NAME in names
    assert LIGHT_ALIAS in names


async def test_seeding_fills_the_card_box(
    hass: HomeAssistant, seeded_hass: MockConfigEntry
) -> None:
    result = await hass.services.async_call(
        "kassistant",
        "seed",
        {"language": "en", "max_per_intent": 4},
        blocking=True,
        return_response=True,
    )

    assert result["language"] == "en"
    assert result["stored"] > 0
    assert result["names"] >= 2

    stats = await hass.async_add_executor_job(seeded_hass.runtime_data.store.stats)
    assert stats["examples"] == result["stored"]
    # Everything stored must be searchable, or the whole point is lost.
    assert stats["indexed"] == stats["examples"]


async def test_seeded_cards_carry_a_replayable_intent(
    hass: HomeAssistant, seeded_hass: MockConfigEntry
) -> None:
    """A card is only useful if the agent can actually execute it later."""
    await hass.services.async_call(
        "kassistant",
        "seed",
        {"language": "en", "max_per_intent": 4},
        blocking=True,
        return_response=True,
    )

    store = seeded_hass.runtime_data.store
    row = await hass.async_add_executor_job(store.get_example, 1)
    action = json.loads(row["action"])

    assert action["type"] == "intent"
    assert action["intent"].startswith("Hass")
    # Slots must be in the shape intent.async_handle expects.
    for value in action["slots"].values():
        assert "value" in value


async def test_seeding_twice_does_not_duplicate(
    hass: HomeAssistant, seeded_hass: MockConfigEntry
) -> None:
    call = {"language": "en", "max_per_intent": 4}
    first = await hass.services.async_call(
        "kassistant", "seed", call, blocking=True, return_response=True
    )
    await hass.services.async_call(
        "kassistant", "seed", call, blocking=True, return_response=True
    )

    stats = await hass.async_add_executor_job(seeded_hass.runtime_data.store.stats)
    assert stats["examples"] == first["stored"]
