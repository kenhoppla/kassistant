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

import logging
from dataclasses import dataclass
from typing import Any

import home_assistant_intents
from hassil.intents import Intents, TextSlotList
from hassil.recognize import recognize
from hassil.sample import sample_intents
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er

from .data import KassistantData
from .store import SOURCE_SEED
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
    """Put the real device names in, and build the action for each sentence."""
    pairs: list[tuple[str, dict[str, Any]]] = []

    for template in generic_templates:
        pairs.append((template.sentence, _action(template.intent, template.slots)))

    for name in names:
        for template in entity_templates:
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
    """Embed the sentences in batches and write them to the card box."""
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
