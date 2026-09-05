"""kassistant -- a learning card box in front of your voice assistant."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CONF_EMBED_MODEL,
    CONF_EMBED_URL,
    CONF_FALLBACK_AGENT,
    CONF_THRESHOLD,
    DB_FILENAME,
    DEFAULT_THRESHOLD,
)
from .data import KassistantData
from .embeddings import OllamaEmbeddings
from .learn import ActionRecorder
from .router import Router
from .store import Store

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.CONVERSATION]

type KassistantConfigEntry = ConfigEntry[KassistantData]


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

    stats = await hass.async_add_executor_job(store.stats)
    _LOGGER.info(
        "kassistant ready: %d cards, %d of them searchable, %d decisions logged",
        stats["examples"],
        stats["indexed"],
        stats["decisions"],
    )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: KassistantConfigEntry) -> bool:
    """Tear down."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        data = entry.runtime_data
        data.recorder.stop()
        await hass.async_add_executor_job(data.store.close)
    return unloaded


async def _async_options_updated(
    hass: HomeAssistant, entry: KassistantConfigEntry
) -> None:
    """Apply changed options without a restart where possible."""
    entry.runtime_data.router.set_threshold(
        entry.options.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
    )
