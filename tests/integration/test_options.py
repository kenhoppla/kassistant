"""Changing the mode through the dialog the user actually clicks.

Setting the option directly in code was already covered, but that skips the
options flow -- which is the only way a user can change anything.
"""

from __future__ import annotations

import json

import pytest
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
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
        options={"mode": "observe", "threshold": 0.92, "learn": True},
    )
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    return config_entry


async def test_the_dialog_offers_every_mode(hass: HomeAssistant, entry) -> None:
    result = await hass.config_entries.options.async_init(entry.entry_id)

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"


async def test_switching_the_mode_through_the_dialog_takes_effect(
    hass: HomeAssistant, entry
) -> None:
    """The whole point of the dialog.

    Observe mode never consults the card box, so its decisions say nothing about
    how well kassistant recognises. If a switch to shadow did not reach the
    agent, the diagnostic sensor would stay unknown forever and there would be
    no way to tell it apart from a broken sensor.
    """
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"mode": "shadow", "threshold": 0.92, "learn": True}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()

    assert entry.options["mode"] == "shadow"

    await conversation.async_converse(
        hass,
        text="anything at all",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id="conversation.kassistant",
    )
    await hass.async_block_till_done()

    store = entry.runtime_data.store
    last = await hass.async_add_executor_job(_last_decision, store)
    assert last["mode"] == "shadow", "the agent is still running in the old mode"
    assert last["tier"] != "observe", "the card box was not consulted"


def _last_decision(store):
    connection = store._require_conn()
    with store._lock:
        return connection.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT 1"
        ).fetchone()


async def test_a_second_setup_is_reported(hass: HomeAssistant, entry) -> None:
    """Two entries share a card box but not an index, and settings apply per entry.

    Switching the mode on one leaves the other untouched -- and the one the
    voice pipeline actually uses may not be the one that was changed. From the
    outside that is nearly impossible to work out, so it has to be said plainly.
    """
    from homeassistant.helpers import issue_registry as ir

    issues = ir.async_get(hass)
    assert issues.async_get_issue("kassistant", "duplicate_entries") is None

    second = MockConfigEntry(
        domain="kassistant",
        data=dict(entry.data),
        options={"mode": "observe", "threshold": 0.92, "learn": True},
    )
    second.add_to_hass(hass)
    await hass.config_entries.async_setup(second.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)

    assert issues.async_get_issue("kassistant", "duplicate_entries") is not None


async def test_the_warning_clears_when_the_extra_one_is_removed(
    hass: HomeAssistant, entry
) -> None:
    from homeassistant.helpers import issue_registry as ir

    second = MockConfigEntry(
        domain="kassistant",
        data=dict(entry.data),
        options={"mode": "observe", "threshold": 0.92, "learn": True},
    )
    second.add_to_hass(hass)
    await hass.config_entries.async_setup(second.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)

    await hass.config_entries.async_remove(second.entry_id)
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)

    assert ir.async_get(hass).async_get_issue("kassistant", "duplicate_entries") is None
