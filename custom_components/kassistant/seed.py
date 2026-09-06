"""Fill the card box with Home Assistant's own example sentences.

Without this, kassistant starts out empty and stays useless until the user has
spoken enough for the learning loop to catch up. Home Assistant already ships
thousands of curated, human-written example sentences per language -- the same
ones its built-in matcher uses -- so there is no reason to start from nothing.

How the sentences become cards:

1. Sample sentences from the templates, with a placeholder standing in for the
   device name.
2. Run each sampled sentence back through Home Assistant's own recogniser. What
   it reports is what gets stored -- we never guess which slots a sentence
   carries. Anything it fails to recognise is dropped.
3. Substitute the real device names into the placeholder. That happens as plain
   text replacement, so recognition runs once rather than once per device --
   the difference between seconds and minutes on a Raspberry Pi.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import home_assistant_intents
from hassil.intents import Intents, TextSlotList
from hassil.recognize import recognize
from hassil.sample import sample_intents
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import (
    async_listen_entity_updates,
    async_should_expose,
)
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.start import async_at_started

from .const import DOMAIN, ISSUE_SEED_FAILED
from .data import KassistantData
from .embeddings import EmbeddingError
from .store import SOURCE_SEED, canonical_action
from .text import normalize

_LOGGER = logging.getLogger(__name__)

# Stands in for a device name while sampling. A single made-up token, so it
# survives the templates unchanged and can be swapped out afterwards.
PLACEHOLDER = "Kassistantplaceholder"

# Intents whose meaning depends on which device is named. Only these are
# sampled per device; everything else is sampled once.
ENTITY_INTENTS = frozenset(
    {
        "HassTurnOn",
        "HassTurnOff",
        "HassGetState",
        "HassLightSet",
        "HassSetPosition",
        "HassSetVolume",
        "HassClimateSetTemperature",
        "HassClimateGetTemperature",
        "HassMediaPause",
        "HassMediaUnpause",
        "HassMediaNext",
        "HassMediaPrevious",
        "HassFanSetSpeed",
        "HassVacuumStart",
        "HassVacuumReturnToBase",
        "HassLockLock",
        "HassLockUnlock",
    }
)

DEFAULT_MAX_PER_INTENT = 12
# A ceiling so a house with hundreds of exposed entities cannot start an
# embedding job that runs for an hour.
DEFAULT_MAX_SENTENCES = 4000
EMBED_BATCH = 64


@dataclass(slots=True, frozen=True)
class SeedResult:
    """What a seeding run accomplished."""

    language: str
    names: int
    sampled: int
    stored: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "names": self.names,
            "sampled": self.sampled,
            "stored": self.stored,
        }


@dataclass(slots=True, frozen=True)
class _Template:
    """A recognised sentence, ready to have a real device name put in."""

    intent: str
    sentence: str
    slots: dict[str, Any]


# -- Public entry point -------------------------------------------------------


async def async_seed(
    hass: HomeAssistant,
    data: KassistantData,
    *,
    language: str | None = None,
    max_per_intent: int = DEFAULT_MAX_PER_INTENT,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
) -> SeedResult:
    """Sample Home Assistant's example sentences and store them as cards."""
    resolved = _resolve_language(language or hass.config.language)
    if resolved is None:
        raise ValueError(
            f"Home Assistant ships no example sentences for {language or hass.config.language!r}"
        )

    names = async_exposed_names(hass)
    if not names:
        _LOGGER.warning(
            "No entities are exposed to Assist, so only generic sentences are seeded"
        )

    entity_templates, generic_templates = await hass.async_add_executor_job(
        _build_templates, resolved, max_per_intent
    )

    pairs = _expand(entity_templates, generic_templates, names, max_sentences)
    _LOGGER.info(
        "Seeding %d sentences for %d names in %s", len(pairs), len(names), resolved
    )

    stored = await _store_all(hass, data, resolved, pairs)

    return SeedResult(
        language=resolved, names=len(names), sampled=len(pairs), stored=stored
    )


@callback
def async_exposed_names(hass: HomeAssistant) -> list[str]:
    """Every name the user could plausibly say for an exposed entity.

    Aliases count as separate names on purpose -- an alias exists precisely
    because someone says that instead of the official name.
    """
    registry = er.async_get(hass)
    names: list[str] = []
    seen: set[str] = set()

    for state in hass.states.async_all():
        if not async_should_expose(hass, conversation.DOMAIN, state.entity_id):
            continue

        candidates = [state.name]
        if (entry := registry.async_get(state.entity_id)) is not None:
            candidates.extend(entry.aliases)

        for candidate in candidates:
            if not candidate:
                continue
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            names.append(candidate)

    return names


# -- Sampling (blocking, runs in the executor) --------------------------------


def _resolve_language(language: str) -> str | None:
    """Match Home Assistant's language against the shipped sentence sets."""
    available = set(home_assistant_intents.get_languages())
    if language in available:
        return language
    base = language.split("-", 1)[0]
    return base if base in available else None


def _build_templates(
    language: str, max_per_intent: int
) -> tuple[list[_Template], list[_Template]]:
    """Sample and recognise sentences once, with a placeholder for the name."""
    intents = Intents.from_dict(home_assistant_intents.get_intents(language))

    named = _sample(
        intents,
        language,
        max_per_intent,
        name_values=[PLACEHOLDER],
        intent_names=ENTITY_INTENTS,
        require_placeholder=True,
    )
    generic = _sample(
        intents,
        language,
        max_per_intent,
        name_values=[],
        intent_names=None,
        require_placeholder=False,
    )
    return named, generic


def _sample(
    intents: Intents,
    language: str,
    max_per_intent: int,
    *,
    name_values: list[str],
    intent_names: frozenset[str] | None,
    require_placeholder: bool,
) -> list[_Template]:
    """Sample sentences and keep only those the recogniser agrees on."""
    slot_lists = {
        "name": TextSlotList.from_tuples([(v, v) for v in name_values]),
        # Seeding works per device name. Areas and floors are left out because
        # a sentence naming an area means something different for every house.
        "area": TextSlotList.from_tuples([]),
        "floor": TextSlotList.from_tuples([]),
    }

    templates: list[_Template] = []
    seen: set[str] = set()

    sampled = sample_intents(
        intents,
        slot_lists=slot_lists,
        max_sentences_per_intent=max_per_intent,
        intent_names=set(intent_names) if intent_names else None,
        language=language,
        # Expanding every value of a numeric range explodes combinatorially and
        # takes minutes. Sentences that need one are dropped by the recogniser.
        expand_ranges=False,
    )

    for _, sentence in sampled:
        if require_placeholder and PLACEHOLDER not in sentence:
            continue

        result = recognize(sentence, intents, slot_lists=slot_lists)
        if result is None:
            continue

        slots = {name: entity.value for name, entity in result.entities.items()}
        if require_placeholder and slots.get("name") != PLACEHOLDER:
            continue

        key = normalize(sentence)
        if not key or key in seen:
            continue
        seen.add(key)

        templates.append(
            _Template(intent=result.intent.name, sentence=sentence, slots=slots)
        )

    return templates


# -- Turning templates into cards ---------------------------------------------


def _expand(
    entity_templates: list[_Template],
    generic_templates: list[_Template],
    names: list[str],
    max_sentences: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Put the real device names in, and build the action for each sentence.

    Templates are the outer loop and names the inner one, so hitting the ceiling
    costs every device a phrasing rather than costing the last devices all of
    them. The other way round, a house past the limit would end up with some
    lamps answering instantly and others never getting a card at all.
    """
    pairs: list[tuple[str, dict[str, Any]]] = []

    for template in generic_templates:
        if len(pairs) >= max_sentences:
            break
        pairs.append((template.sentence, _action(template.intent, template.slots)))

    for template in entity_templates:
        for name in names:
            if len(pairs) >= max_sentences:
                _LOGGER.warning(
                    "Stopped seeding at %d sentences; raise max_sentences to go further",
                    max_sentences,
                )
                return pairs
            slots = dict(template.slots)
            slots["name"] = name
            pairs.append(
                (
                    template.sentence.replace(PLACEHOLDER, name),
                    _action(template.intent, slots),
                )
            )

    return pairs


def _action(intent_name: str, slots: dict[str, Any]) -> dict[str, Any]:
    """Wrap an intent and its slots the way the agent expects to replay it."""
    return {
        "type": "intent",
        "intent": intent_name,
        "slots": {name: {"value": value} for name, value in slots.items()},
    }


async def _store_all(
    hass: HomeAssistant,
    data: KassistantData,
    language: str,
    pairs: list[tuple[str, dict[str, Any]]],
) -> int:
    """Embed the sentences in batches and write them to the card box.

    Sentences already in the box are dropped before anything is embedded. That
    makes a repeat run cost almost nothing, which is what allows seeding to be
    kept up to date automatically instead of being a job the user has to start.
    """
    known = await hass.async_add_executor_job(data.store.known_keys, language)
    fresh = [
        (sentence, action)
        for sentence, action in pairs
        if (normalize(sentence), canonical_action(action)) not in known
    ]

    if len(fresh) != len(pairs):
        _LOGGER.debug(
            "Skipping %d sentences that are already stored", len(pairs) - len(fresh)
        )
    pairs = fresh
    stored = 0

    for start in range(0, len(pairs), EMBED_BATCH):
        batch = pairs[start : start + EMBED_BATCH]
        sentences = [sentence for sentence, _ in batch]
        norms = [normalize(sentence) for sentence in sentences]

        try:
            vectors = await data.embeddings.embed(norms)
        except Exception:
            # Cards are still worth keeping without a vector; the missing ones
            # can be filled in later rather than losing the whole run.
            _LOGGER.exception("Embedding a batch failed, storing it without vectors")
            vectors = None

        for index, (sentence, action) in enumerate(batch):
            example_id = await hass.async_add_executor_job(
                _add,
                data.store,
                sentence,
                norms[index],
                language,
                action,
                None if vectors is None else vectors[index],
            )
            if example_id is not None:
                stored += 1

    return stored


def _add(store, utterance, norm, language, action, embedding):
    """add_example only takes keyword arguments."""
    return store.add_example(
        utterance=utterance,
        norm=norm,
        language=language,
        action=action,
        source=SOURCE_SEED,
        embedding=embedding,
    )


# -- Keeping the card box current ---------------------------------------------

# Exposing a handful of devices produces a burst of updates. Waiting a little
# collapses them into a single seeding run.
RESEED_COOLDOWN = 30.0

# How many missing vectors to fetch per request when catching up.
BACKFILL_BATCH = 128


class SeedScheduler:
    """Seeds on startup and again whenever the exposed devices change.

    Seeding is deliberately not something the user has to remember. It only
    writes cards, it never acts on them -- the agent still obeys its configured
    mode -- so there is nothing to hold back for. What matters is that it stays
    current: expose a new lamp and its sentences should appear without anyone
    having to know that an action exists.

    Repeat runs are cheap because the store is asked which sentences it already
    has before anything is sent to the embedding service.
    """

    def __init__(self, hass: HomeAssistant, data: KassistantData) -> None:
        self._hass = hass
        self._data = data
        # The running seeding job, so unloading can wait for it. Without that
        # the job keeps writing into a card box that teardown has closed.
        self._task: asyncio.Task[None] | None = None
        # Something changed while a run was in flight. Dropping it would leave
        # a device that was exposed mid-run without cards until the next,
        # unrelated change came along.
        self._pending = False
        self._debouncer = Debouncer(
            hass,
            _LOGGER,
            cooldown=RESEED_COOLDOWN,
            immediate=False,
            function=self._async_run,
            background=True,
        )
        self._last_names: tuple[str, ...] | None = None

    @callback
    def async_setup(self) -> list[CALLBACK_TYPE]:
        """Start listening. Returns the unsubscribe callbacks."""
        return [
            async_at_started(self._hass, self._async_started),
            async_listen_entity_updates(
                self._hass, conversation.DOMAIN, self._async_exposure_changed
            ),
            self._hass.bus.async_listen(
                er.EVENT_ENTITY_REGISTRY_UPDATED, self._async_registry_changed
            ),
        ]

    async def async_shutdown(self) -> None:
        """Stop seeding and wait for a run in flight.

        Must finish before the store is closed. A seeding job writes in batches,
        so tearing the database out from under it raises halfway through.
        """
        self._debouncer.async_shutdown()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    @callback
    def _async_started(self, _hass: HomeAssistant) -> None:
        """Home Assistant has finished starting; entities are known by now.

        Runs straight away rather than through the debouncer -- there is nothing
        to collapse on a first run, and waiting would only leave the card box
        empty for no reason.
        """
        self._async_schedule(self._async_run(), "kassistant seed on start")

    @callback
    def _async_exposure_changed(self) -> None:
        self._async_schedule(self._debouncer.async_call(), "kassistant reseed")

    @callback
    def _async_registry_changed(self, event: Event) -> None:
        """A rename or a new alias changes what the user is likely to say."""
        if event.data.get("action") == "update" and not (
            {"name", "aliases"} & set(event.data.get("changes") or {})
        ):
            return
        self._async_schedule(self._debouncer.async_call(), "kassistant reseed")

    @callback
    def _async_schedule(self, coro: Coroutine[Any, Any, None], name: str) -> None:
        """Run a seeding job in the background, unless one is already going."""
        if self._task is not None and not self._task.done():
            coro.close()
            self._pending = True
            return
        self._task = self._hass.async_create_background_task(coro, name)

    async def _async_run(self) -> None:
        """Seed, then seed again if anything changed while we were busy."""
        while True:
            self._pending = False
            await self._async_run_once()
            if not self._pending:
                return

    async def _async_run_once(self) -> None:
        """Seed, unless nothing about the exposed devices has changed."""
        names = tuple(async_exposed_names(self._hass))
        if names == self._last_names:
            return

        try:
            async with self._data.seeding:
                result = await async_seed(self._hass, self._data)
        except ValueError as err:
            # No sentences ship for this language. Nothing to retry.
            _LOGGER.warning("Not seeding: %s", err)
            self._last_names = names
            return
        except Exception:
            # Ollama may simply be down. Leave _last_names alone so the next
            # change tries again, and tell the user -- an empty card box is
            # otherwise invisible, kassistant would just stay slow forever.
            _LOGGER.exception("Seeding failed")
            ir.async_create_issue(
                self._hass,
                DOMAIN,
                ISSUE_SEED_FAILED,
                is_fixable=False,
                severity=ir.IssueSeverity.WARNING,
                translation_key=ISSUE_SEED_FAILED,
            )
            return

        self._last_names = names
        if result.stored:
            _LOGGER.info(
                "Seeded %d new cards for %d names (%s)",
                result.stored,
                result.names,
                result.language,
            )

        # Cards stored while the embedding service was down have no vector, and
        # a later run would skip them as already known. Without this they would
        # sit in the box unsearchable forever.
        await self.async_backfill()
        await self.async_report_health()

    async def async_backfill(self) -> None:
        """Give vectors to cards that were stored without one."""
        store = self._data.store
        filled = 0

        while True:
            rows = await self._hass.async_add_executor_job(
                store.examples_without_embedding, BACKFILL_BATCH
            )
            if not rows:
                break

            try:
                vectors = await self._data.embeddings.embed([r["norm"] for r in rows])
            except EmbeddingError as err:
                _LOGGER.debug("Backfilling vectors failed, will retry later: %s", err)
                return

            for row, vector in zip(rows, vectors, strict=True):
                await self._hass.async_add_executor_job(
                    store.set_embedding, row["id"], vector
                )
            filled += len(rows)

            if len(rows) < BACKFILL_BATCH:
                break

        if filled:
            _LOGGER.info("Filled in vectors for %d cards", filled)

    async def async_report_health(self) -> None:
        """Raise or clear the repair notice based on what is actually usable.

        Cards without vectors cannot be found, so a box full of them is just as
        useless as an empty one -- and just as invisible.
        """
        stats = await self._hass.async_add_executor_job(self._data.store.stats)

        if stats["examples"] and not stats["indexed"]:
            ir.async_create_issue(
                self._hass,
                DOMAIN,
                ISSUE_SEED_FAILED,
                is_fixable=False,
                severity=ir.IssueSeverity.WARNING,
                translation_key=ISSUE_SEED_FAILED,
            )
        else:
            ir.async_delete_issue(self._hass, DOMAIN, ISSUE_SEED_FAILED)
