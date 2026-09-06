"""kassistant -- a learning card box in front of your voice assistant."""

from __future__ import annotations

import logging

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CONF_EMBED_MODEL,
    CONF_EMBED_URL,
    CONF_FALLBACK_AGENT,
    CONF_THRESHOLD,
    DB_FILENAME,
    DEFAULT_THRESHOLD,
    DOMAIN,
)
from .data import KassistantData
from .embeddings import EmbeddingError, OllamaEmbeddings
from .learn import ActionRecorder
from .router import Router
from .seed import (
    DEFAULT_MAX_PER_INTENT,
    DEFAULT_MAX_SENTENCES,
    SeedScheduler,
    async_seed,
)
from .store import Store

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.CONVERSATION]

type KassistantConfigEntry = ConfigEntry[KassistantData]

SERVICE_SEED = "seed"
ATTR_LANGUAGE = "language"
ATTR_MAX_PER_INTENT = "max_per_intent"
ATTR_MAX_SENTENCES = "max_sentences"

SEED_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_LANGUAGE): cv.string,
        vol.Optional(ATTR_MAX_PER_INTENT, default=DEFAULT_MAX_PER_INTENT): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=50)
        ),
        vol.Optional(ATTR_MAX_SENTENCES, default=DEFAULT_MAX_SENTENCES): vol.All(
            vol.Coerce(int), vol.Range(min=100, max=50000)
        ),
    }
)


async def async_setup_entry(hass: HomeAssistant, entry: KassistantConfigEntry) -> bool:
    """Set up kassistant from a config entry."""
    store = Store(hass.config.path(DB_FILENAME))
    await hass.async_add_executor_job(store.setup)

    embeddings = OllamaEmbeddings(
        session=async_get_clientsession(hass),
        base_url=entry.data[CONF_EMBED_URL],
        model=entry.data[CONF_EMBED_MODEL],
    )

    router = Router(
        hass=hass,
        store=store,
        embeddings=embeddings,
        threshold=entry.options.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
    )

    recorder = ActionRecorder(hass)
    recorder.start()

    entry.runtime_data = KassistantData(
        store=store,
        embeddings=embeddings,
        router=router,
        recorder=recorder,
        fallback_agent=entry.data[CONF_FALLBACK_AGENT],
    )

    try:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    except Exception:
        # Home Assistant retries a failed setup, so release the database handle
        # and the event listener first -- otherwise both leak on every attempt.
        recorder.stop()
        await hass.async_add_executor_job(store.close)
        raise

    entry.async_on_unload(entry.add_update_listener(_async_options_updated))
    _async_register_services(hass, entry)

    # Keep the card box filled from Home Assistant's own example sentences, and
    # keep it current as devices and aliases change. This only writes cards; the
    # agent still obeys its configured mode, so there is nothing to hold back.
    scheduler = SeedScheduler(hass, entry.runtime_data)
    entry.runtime_data.scheduler = scheduler
    for unsubscribe in scheduler.async_setup():
        entry.async_on_unload(unsubscribe)

    stats = await hass.async_add_executor_job(store.stats)
    _LOGGER.info(
        "kassistant ready: %d cards, %d of them searchable, %d decisions logged",
        stats["examples"],
        stats["indexed"],
        stats["decisions"],
    )
    return True


def _async_register_services(hass: HomeAssistant, entry: KassistantConfigEntry) -> None:
    """Expose ``kassistant.seed`` while an entry is loaded.

    Only one entry can exist, so the handler may close over it. On a reload the
    service is removed and registered again with the fresh entry.
    """

    async def handle_seed(call: ServiceCall) -> ServiceResponse:
        """Fill the card box from Home Assistant's own example sentences."""
        data = entry.runtime_data
        try:
            # The scheduler may be seeding right now; doing both at once would
            # only duplicate the work and hammer the embedding service.
            async with data.seeding:
                result = await async_seed(
                    hass,
                    data,
                    language=call.data.get(ATTR_LANGUAGE),
                    max_per_intent=call.data[ATTR_MAX_PER_INTENT],
                    max_sentences=call.data[ATTR_MAX_SENTENCES],
                )
        except ValueError as err:
            # Unsupported language -- the user's mistake, not a crash.
            raise ServiceValidationError(str(err)) from err
        except EmbeddingError as err:
            raise ServiceValidationError(
                f"Could not reach the embedding service: {err}"
            ) from err

        _LOGGER.info(
            "Seeded %d cards from %d sampled sentences for %d names (%s)",
            result.stored,
            result.sampled,
            result.names,
            result.language,
        )
        # A manual run should also repair what an earlier outage left behind,
        # and then say plainly whether the box is usable.
        if (scheduler := data.scheduler) is not None:
            await scheduler.async_backfill()
            await scheduler.async_report_health()

        return result.as_dict()

    hass.services.async_register(
        DOMAIN,
        SERVICE_SEED,
        handle_seed,
        schema=SEED_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )


async def async_unload_entry(hass: HomeAssistant, entry: KassistantConfigEntry) -> bool:
    """Tear down."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        hass.services.async_remove(DOMAIN, SERVICE_SEED)
        data = entry.runtime_data
        data.recorder.stop()
        # Order matters: a seeding job writes in batches, so it has to be
        # stopped before the database it writes into is closed.
        if data.scheduler is not None:
            await data.scheduler.async_shutdown()
        await hass.async_add_executor_job(data.store.close)
    return unloaded


async def _async_options_updated(
    hass: HomeAssistant, entry: KassistantConfigEntry
) -> None:
    """Apply changed options without a restart where possible."""
    entry.runtime_data.router.set_threshold(
        entry.options.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
    )
