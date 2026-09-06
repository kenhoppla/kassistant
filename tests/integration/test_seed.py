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
    """A kassistant with one exposed light, seeded on start."""
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
    # Seeding runs as a background task so it never delays startup; tests have
    # to say explicitly that they want to wait for it.
    await hass.async_block_till_done(wait_background_tasks=True)
    return config_entry


async def test_exposed_names_include_aliases(
    hass: HomeAssistant, seeded_hass: MockConfigEntry
) -> None:
    """An alias exists because someone says that instead of the real name."""
    from kassistant.seed import async_exposed_names

    names = async_exposed_names(hass)

    assert LIGHT_NAME in names
    assert LIGHT_ALIAS in names


async def test_seeding_runs_on_start_without_being_asked(
    hass: HomeAssistant, seeded_hass: MockConfigEntry
) -> None:
    """The promise is a card box that is useful on day one.

    Seeding only writes cards, it never acts on them -- the agent still obeys
    its configured mode -- so there is nothing to hold back for.
    """
    stats = await hass.async_add_executor_job(seeded_hass.runtime_data.store.stats)

    assert stats["examples"] > 0
    # Everything stored must be searchable, or the whole point is lost.
    assert stats["indexed"] == stats["examples"]


async def test_seeded_cards_carry_a_replayable_intent(
    hass: HomeAssistant, seeded_hass: MockConfigEntry
) -> None:
    """A card is only useful if the agent can actually execute it later."""
    store = seeded_hass.runtime_data.store
    row = await hass.async_add_executor_job(store.get_example, 1)
    action = json.loads(row["action"])

    assert action["type"] == "intent"
    assert action["intent"].startswith("Hass")
    # Slots must be in the shape intent.async_handle expects.
    for value in action["slots"].values():
        assert "value" in value


async def test_reseeding_stores_nothing_new(
    hass: HomeAssistant, seeded_hass: MockConfigEntry
) -> None:
    """Repeat runs have to be cheap, or automatic reseeding is not defensible.

    Sentences already in the box are dropped before anything reaches the
    embedding service, so a second pass over the same devices is nearly free.
    """
    before = await hass.async_add_executor_job(seeded_hass.runtime_data.store.stats)

    result = await hass.services.async_call(
        "kassistant", "seed", {"language": "en"}, blocking=True, return_response=True
    )
    after = await hass.async_add_executor_job(seeded_hass.runtime_data.store.stats)

    assert result["stored"] == 0
    assert after["examples"] == before["examples"]


async def test_a_wider_sample_adds_more_cards(
    hass: HomeAssistant, seeded_hass: MockConfigEntry
) -> None:
    """Asking for more phrasings must genuinely widen the box."""
    before = await hass.async_add_executor_job(seeded_hass.runtime_data.store.stats)

    result = await hass.services.async_call(
        "kassistant",
        "seed",
        {"language": "en", "max_per_intent": 30},
        blocking=True,
        return_response=True,
    )
    after = await hass.async_add_executor_job(seeded_hass.runtime_data.store.stats)

    assert result["stored"] > 0
    assert after["examples"] == before["examples"] + result["stored"]


async def test_no_repair_issue_while_seeding_works(
    hass: HomeAssistant, seeded_hass: MockConfigEntry
) -> None:
    from homeassistant.helpers import issue_registry as ir

    assert ir.async_get(hass).async_get_issue("kassistant", "seed_failed") is None


async def test_a_failed_seeding_run_tells_the_user(
    hass: HomeAssistant, custom_integration, aioclient_mock: AiohttpClientMocker
) -> None:
    """An empty card box is invisible otherwise -- kassistant would just stay slow."""
    from homeassistant.helpers import issue_registry as ir

    aioclient_mock.post("http://localhost:11434/api/embed", status=500)

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

    assert ir.async_get(hass).async_get_issue("kassistant", "seed_failed") is not None


async def test_vectors_are_filled_in_after_the_service_recovers(
    hass: HomeAssistant, custom_integration, aioclient_mock: AiohttpClientMocker
) -> None:
    """Cards stored during an outage must not stay unsearchable forever.

    Seeding keeps the sentences even when embedding fails, and a later run skips
    them as already known -- so something has to come back for the vectors.
    """
    from homeassistant.helpers import issue_registry as ir

    aioclient_mock.post("http://localhost:11434/api/embed", status=500)

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

    store = config_entry.runtime_data.store
    broken = await hass.async_add_executor_job(store.stats)
    assert broken["examples"] > 0
    assert broken["indexed"] == 0
    assert ir.async_get(hass).async_get_issue("kassistant", "seed_failed") is not None

    # Ollama comes back.
    aioclient_mock.clear_requests()

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

    await hass.services.async_call(
        "kassistant", "seed", {"language": "en"}, blocking=True, return_response=True
    )
    await hass.async_block_till_done(wait_background_tasks=True)

    healed = await hass.async_add_executor_job(store.stats)
    assert healed["indexed"] == healed["examples"]


def test_the_sentence_ceiling_costs_phrasings_not_devices() -> None:
    """Every device must get some cards, even when the ceiling cuts the run.

    Truncating device by device would leave a house past the limit with some
    lamps answering instantly and others never getting a card at all.
    """
    from kassistant.seed import PLACEHOLDER, _expand
    from kassistant.seed import _Template as Template

    templates = [
        Template(intent="HassTurnOn", sentence=f"turn on {PLACEHOLDER} #{i}", slots={})
        for i in range(10)
    ]
    names = [f"device {i}" for i in range(20)]

    pairs = _expand(templates, [], names, max_sentences=50)

    covered = {name for name in names if any(name in sentence for sentence, _ in pairs)}
    assert covered == set(names)


async def test_a_change_during_a_run_is_not_lost(
    hass: HomeAssistant, seeded_hass: MockConfigEntry
) -> None:
    """Expose a device while seeding is busy and it must still get cards.

    Requests arriving mid-run are dropped rather than queued -- two seeding jobs
    at once would only duplicate work. So the run itself has to notice that
    something changed and go round again, otherwise that device waits for some
    later, unrelated change to trigger a run.
    """
    scheduler = seeded_hass.runtime_data.scheduler
    passes = 0

    async def run_once() -> None:
        nonlocal passes
        passes += 1
        if passes == 1:
            # Stands in for an exposure change arriving while we were busy.
            scheduler._async_schedule(scheduler._async_run(), "concurrent change")

    scheduler._async_run_once = run_once
    await scheduler._async_run()

    assert passes == 2


async def test_a_run_without_changes_does_not_repeat(
    hass: HomeAssistant, seeded_hass: MockConfigEntry
) -> None:
    """The retry loop must not spin when nothing arrived."""
    scheduler = seeded_hass.runtime_data.scheduler
    passes = 0

    async def run_once() -> None:
        nonlocal passes
        passes += 1

    scheduler._async_run_once = run_once
    await scheduler._async_run()

    assert passes == 1
