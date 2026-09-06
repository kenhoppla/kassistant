"""Reads the card box on a timer, once for every sensor that needs it."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .data import KassistantData

_LOGGER = logging.getLogger(__name__)

SCAN_INTERVAL = timedelta(seconds=60)


class KassistantCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Polls the statistics the diagnostic sensors report."""

    def __init__(self, hass: HomeAssistant, data: KassistantData) -> None:
        super().__init__(
            hass,
            logger=_LOGGER,
            name="kassistant",
            update_interval=SCAN_INTERVAL,
        )
        self._store = data.store

    async def _async_update_data(self) -> dict[str, Any]:
        cards = await self.hass.async_add_executor_job(self._store.card_stats)
        decisions = await self.hass.async_add_executor_job(self._store.decision_stats)
        return {"cards": cards, "decisions": decisions}
