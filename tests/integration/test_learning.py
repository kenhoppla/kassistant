"""The promise of the project, end to end.

kassistant claims three things: it watches what your fallback agent does and
remembers it, it answers a familiar sentence itself, and it refuses to guess
when it is unsure. Everything else is plumbing around those three.
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta

import pytest
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.core import Context, HomeAssistant, ServiceCall
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    AiohttpClientMockResponse,
)

DIMENSION = 8
LIGHT = "light.ceiling"
LIGHT_NAME = "Ceiling Light"


@pytest.fixture
def calls(hass: HomeAssistant) -> list[ServiceCall]:
    """A stand-in light, so acting on it can be observed."""
    recorded: list[ServiceCall] = []

    async def record(call: ServiceCall) -> None:
        recorded.append(call)

    hass.services.async_register("light", "turn_on", record)
    hass.services.async_register("light", "turn_off", record)
    return recorded


@pytest.fixture
def embed_mock(aioclient_mock: AiohttpClientMocker) -> AiohttpClientMocker:
    """A deterministic stand-in for a real embedding model.

    Identical text gives an identical vector, and different text gives a
    near-orthogonal one. That is exactly the property the router depends on:
    recognise what was said before, abstain on anything else.
    """

    async def respond(method, url, data):
        payload = json.loads(data) if isinstance(data, str | bytes) else data
        vectors = []
        for text in payload["input"]:
            digest = hashlib.sha256(text.encode()).digest()
            # Centred on zero, so unrelated text comes out near-orthogonal.
            # All-positive components would make everything look 0.6+ similar
            # and the router would appear to match things it never would.
            vectors.append([digest[i] / 127.5 - 1.0 for i in range(DIMENSION)])
        return AiohttpClientMockResponse(method, url, json={"embeddings": vectors})

    aioclient_mock.post("http://localhost:11434/api/embed", side_effect=respond)
    return aioclient_mock


async def setup_kassistant(hass: HomeAssistant, **options: object) -> MockConfigEntry:
    """A kassistant in front of Home Assistant's own agent, with one light."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})

    registry = er.async_get(hass)
    entry = registry.async_get_or_create(
        "light", "demo", "ceiling", suggested_object_id="ceiling"
    )
    registry.async_update_entity(entry.entity_id, name=LIGHT_NAME)
    hass.states.async_set(LIGHT, "off", {"friendly_name": LIGHT_NAME})
    async_expose_entity(hass, conversation.DOMAIN, LIGHT, True)

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
            "learn": True,
            "learn_delay": 0,
            **options,
        },
    )
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    return config_entry


async def say(hass: HomeAssistant, text: str) -> conversation.ConversationResult:
    """Speak to kassistant, then let the deferred learning job run.

    Learning waits a moment for the user to object, so the test has to move the
    clock past that before checking what was remembered.
    """
    result = await conversation.async_converse(
        hass,
        text=text,
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id="conversation.kassistant",
    )
    await hass.async_block_till_done()
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=60))
    await hass.async_block_till_done()
    return result


# -- Watching the fallback agent ----------------------------------------------


async def test_an_action_by_the_fallback_agent_becomes_a_card(
    hass: HomeAssistant, custom_integration, embed_mock, calls
) -> None:
    """This is the loop the whole project rests on.

    Home Assistant tags every service call with the context of the conversation
    that caused it. That is what lets us learn from an agent that knows nothing
    about us.
    """
    entry = await setup_kassistant(hass)
    store = entry.runtime_data.store
    before = await hass.async_add_executor_job(store.stats)

    await say(hass, f"turn on {LIGHT_NAME}")

    assert calls, "the fallback agent did not actually act"
    after = await hass.async_add_executor_job(store.stats)
    assert after["examples"] > before["examples"]


async def test_the_learned_card_holds_the_action_that_happened(
    hass: HomeAssistant, custom_integration, embed_mock, calls
) -> None:
    entry = await setup_kassistant(hass)
    store = entry.runtime_data.store

    await say(hass, f"turn on {LIGHT_NAME}")

    learned = await hass.async_add_executor_job(_learned_cards, store)
    assert learned, "nothing was learned"
    action = json.loads(learned[0]["action"])
    assert action["type"] == "actions"
    assert action["actions"], "the card would replay nothing"


async def test_learning_can_be_switched_off(
    hass: HomeAssistant, custom_integration, embed_mock, calls
) -> None:
    entry = await setup_kassistant(hass, learn=False)
    store = entry.runtime_data.store

    await say(hass, f"turn on {LIGHT_NAME}")

    assert calls, "the fallback agent did not act, so the test proves nothing"
    assert not await hass.async_add_executor_job(_learned_cards, store)


def _learned_cards(store) -> list:
    """Cards that came from watching, not from seeding."""
    connection = store._require_conn()
    with store._lock:
        return connection.execute(
            "SELECT * FROM examples WHERE source = 'learned'"
        ).fetchall()


# -- Answering from the card box ----------------------------------------------


async def test_a_learned_sentence_is_answered_without_the_fallback(
    hass: HomeAssistant, custom_integration, embed_mock, calls
) -> None:
    """The payoff: the second time, kassistant does it itself.

    Proven through the decision log rather than by timing, so it cannot pass by
    accident on a fast machine.
    """
    entry = await setup_kassistant(hass)
    sentence = f"turn on {LIGHT_NAME}"

    await say(hass, sentence)  # learned via the fallback
    assert await hass.async_add_executor_job(_learned_cards, entry.runtime_data.store)

    hass.config_entries.async_update_entry(
        entry, options={**entry.options, "mode": "active"}
    )
    await hass.async_block_till_done()
    calls.clear()

    await say(hass, sentence)

    assert calls, "the card was not actually executed"
    last = await hass.async_add_executor_job(_last_decision, entry.runtime_data.store)
    assert last["tier"] == "fastpath"
    assert last["executed"] == 1


async def test_an_unknown_sentence_is_passed_on_rather_than_guessed(
    hass: HomeAssistant, custom_integration, embed_mock, calls
) -> None:
    """Abstaining is the product. A confident wrong answer is the worst outcome."""
    entry = await setup_kassistant(hass, mode="active")

    await say(hass, "something entirely unlike any command")

    last = await hass.async_add_executor_job(_last_decision, entry.runtime_data.store)
    assert last["tier"] in {"abstain", "unavailable"}
    assert last["executed"] == 0


async def test_a_follow_up_cancels_learning(
    hass: HomeAssistant, custom_integration, embed_mock, calls
) -> None:
    """Correcting yourself means the first answer was wrong; learn nothing.

    A missing card only costs time, a wrong one costs trust.
    """
    entry = await setup_kassistant(hass, learn_delay=60)
    store = entry.runtime_data.store

    result = await conversation.async_converse(
        hass,
        text=f"turn on {LIGHT_NAME}",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id="conversation.kassistant",
    )
    await hass.async_block_till_done()

    # Same conversation, immediately after: the user is correcting themselves.
    await conversation.async_converse(
        hass,
        text=f"no, turn off {LIGHT_NAME}",
        conversation_id=result.conversation_id,
        context=Context(),
        language="en",
        agent_id="conversation.kassistant",
    )
    await hass.async_block_till_done()
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=120))
    await hass.async_block_till_done()

    learned = await hass.async_add_executor_job(_learned_cards, store)
    utterances = {row["utterance"] for row in learned}
    assert f"turn on {LIGHT_NAME}" not in utterances


def _last_decision(store):
    connection = store._require_conn()
    with store._lock:
        return connection.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT 1"
        ).fetchone()
