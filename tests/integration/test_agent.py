"""End-to-end tests against a real Home Assistant instance.

The important one is delegation. kassistant answers from inside its own
conversation turn and then calls another agent for the same conversation id --
whether Home Assistant tolerates that was the single largest open risk in the
design, and this is where it gets settled.
"""

from __future__ import annotations

import pytest
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

EMBED_URL = "http://localhost:11434/api/embed"
DIMENSION = 8


@pytest.fixture
def embed_mock(aioclient_mock: AiohttpClientMocker) -> AiohttpClientMocker:
    """Answer every embedding request with a fixed vector."""
    aioclient_mock.post(
        EMBED_URL, json={"embeddings": [[1.0] + [0.0] * (DIMENSION - 1)]}
    )
    return aioclient_mock


@pytest.fixture
async def entry(hass: HomeAssistant, custom_integration, embed_mock) -> MockConfigEntry:
    """A configured kassistant that falls back to Home Assistant's own agent."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})

    entry = MockConfigEntry(
        domain="kassistant",
        title="kassistant",
        data={
            "embed_url": "http://localhost:11434",
            "embed_model": "test-model",
            "fallback_agent": conversation.HOME_ASSISTANT_AGENT,
        },
        options={
            "mode": "observe",
            "threshold": 0.92,
            "learn": True,
            "learn_delay": 0,
        },
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def test_setup_creates_the_conversation_entity(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    state = hass.states.get("conversation.kassistant")
    assert state is not None


async def test_request_is_delegated_to_the_fallback_agent(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    """The core risk: calling another agent from inside our own turn.

    If Home Assistant could not cope with two agents sharing one conversation
    id, this would deadlock or raise instead of returning a result.
    """
    result = await conversation.async_converse(
        hass,
        text="do something nobody has ever asked for",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id="conversation.kassistant",
    )

    assert result.response is not None
    # The default agent does not understand the sentence -- that is fine. What
    # matters is that its answer came back through kassistant at all.
    assert result.response.speech or result.response.error_code


async def test_decisions_are_written_to_the_card_box(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    """Observe mode does nothing visible, but it must record what it saw."""
    store = entry.runtime_data.store

    await conversation.async_converse(
        hass,
        text="another sentence",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id="conversation.kassistant",
    )
    await hass.async_block_till_done()

    stats = await hass.async_add_executor_job(store.stats)
    assert stats["decisions"] >= 1


async def test_unload_releases_everything(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    """Unloading must succeed and take the agent out of service.

    Home Assistant keeps a restored placeholder state around after an unload,
    so the entity does not vanish -- it goes unavailable, which is what stops
    the voice pipeline from routing to it.
    """
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    state = hass.states.get("conversation.kassistant")
    assert state is None or state.state == "unavailable"
