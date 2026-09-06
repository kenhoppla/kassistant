"""The diagnostic sensors.

They exist so the one judgement call the user has to make -- when to switch the
mode to active -- rests on numbers rather than a hunch.
"""

from __future__ import annotations

import json

import pytest
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    AiohttpClientMockResponse,
)

DIMENSION = 8


@pytest.fixture
def embed_mock(aioclient_mock: AiohttpClientMocker) -> AiohttpClientMocker:
    async def respond(method, url, data):
        payload = json.loads(data) if isinstance(data, str | bytes) else data
        return AiohttpClientMockResponse(
            method,
            url,
            json={
                "embeddings": [
                    [1.0] + [0.0] * (DIMENSION - 1) for _ in payload["input"]
                ]
            },
        )

    aioclient_mock.post("http://localhost:11434/api/embed", side_effect=respond)
    return aioclient_mock


@pytest.fixture
async def entry(hass: HomeAssistant, custom_integration, embed_mock) -> MockConfigEntry:
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})

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
    await hass.async_block_till_done(wait_background_tasks=True)
    return config_entry


async def test_both_sensors_exist(hass: HomeAssistant, entry) -> None:
    assert hass.states.get("sensor.kassistant_cards") is not None
    assert hass.states.get("sensor.kassistant_recognised") is not None


async def test_the_card_sensor_separates_seeded_from_learned(
    hass: HomeAssistant, entry
) -> None:
    """Two very different things: what came in the box, and what you taught it.

    Also proves the sensor does not sit at zero after setup: seeding runs after
    the platforms are up, so it has to push a refresh when it is done.
    """
    state = hass.states.get("sensor.kassistant_cards")

    assert int(state.state) > 0
    assert state.attributes["seeded"] > 0
    assert state.attributes["learned"] == 0
    # Nothing may sit in the box unsearchable after a healthy seeding run.
    assert state.attributes["searchable"] == int(state.state)


async def test_the_recognition_sensor_starts_empty(hass: HomeAssistant, entry) -> None:
    """No requests yet means no share to report -- not zero percent."""
    state = hass.states.get("sensor.kassistant_recognised")

    assert state.attributes["sampled"] == 0
    assert state.attributes["recognised"] == 0


async def test_the_recognition_sensor_counts_decisions(
    hass: HomeAssistant, entry
) -> None:
    """It must count decisions, not executions.

    Otherwise the number would read zero in observe and shadow mode, which is
    exactly when the user needs it to decide whether to arm the fast path.
    """
    from homeassistant.core import Context

    await conversation.async_converse(
        hass,
        text="something nobody has said before",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id="conversation.kassistant",
    )
    await hass.async_block_till_done()
    await entry.runtime_data.coordinator.async_refresh()

    state = hass.states.get("sensor.kassistant_recognised")
    assert state.attributes["sampled"] >= 1
    assert state.attributes["by_outcome"]


async def test_sensors_are_diagnostic(hass: HomeAssistant, entry) -> None:
    """They belong under diagnostics, not in the user's main dashboard."""
    from homeassistant.helpers import entity_registry as er

    registry = er.async_get(hass)
    for entity_id in ("sensor.kassistant_cards", "sensor.kassistant_recognised"):
        assert registry.async_get(entity_id).entity_category == "diagnostic"


async def test_everything_hangs_on_one_device(hass: HomeAssistant, entry) -> None:
    """Agent and sensors belong together in the interface.

    They also take their names from the device, which is what allows the German
    translation to apply instead of an English string baked into the code.
    """
    from homeassistant.helpers import device_registry as dr
    from homeassistant.helpers import entity_registry as er

    devices = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)
    assert len(devices) == 1
    assert devices[0].name == "kassistant"

    registry = er.async_get(hass)
    for entity_id in (
        "conversation.kassistant",
        "sensor.kassistant_cards",
        "sensor.kassistant_recognised",
    ):
        assert registry.async_get(entity_id).device_id == devices[0].id


async def test_the_sensor_names_come_from_translations(
    hass: HomeAssistant, entry
) -> None:
    """A hardcoded name would silently ignore the translation files."""
    from homeassistant.helpers import entity_registry as er

    registry = er.async_get(hass)
    assert registry.async_get("sensor.kassistant_cards").translation_key == "cards"
    assert (
        registry.async_get("sensor.kassistant_recognised").translation_key == "handled"
    )
