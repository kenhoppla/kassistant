"""Diagnostic sensors.

kassistant is only worth arming once it is actually recognising what you say,
and that is a question about numbers rather than a feeling. These two sensors
put those numbers where they can be watched over days:

* how many cards exist, and how many of them are searchable
* what share of recent requests kassistant recognised confidently

The second one is the deciding number. Watch it in observe or shadow mode, and
switch to active when it stops climbing -- with the average score telling you
whether the confidence threshold sits where it should.
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .coordinator import KassistantCoordinator
from .entity import device_info


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the diagnostic sensors."""
    coordinator = entry.runtime_data.coordinator
    # Deliberately not async_config_entry_first_refresh: that raises
    # ConfigEntryNotReady on failure, which would stop the whole integration --
    # and the voice agent with it -- because a statistics query went wrong. The
    # sensors simply report nothing until the next poll succeeds.
    await coordinator.async_refresh()
    async_add_entities(
        [CardsSensor(coordinator, entry), HandledSensor(coordinator, entry)]
    )


class _Base(CoordinatorEntity[KassistantCoordinator], SensorEntity):
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_should_poll = False
    # The name comes from the device plus a translated entity name, so the
    # German UI says "Karteikarten" rather than an English string baked into
    # the code.
    _attr_has_entity_name = True

    def __init__(self, coordinator: KassistantCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_device_info = device_info(entry)


class CardsSensor(_Base):
    """How much kassistant knows."""

    _attr_translation_key = "cards"
    _attr_icon = "mdi:cards-outline"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, coordinator: KassistantCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_cards"

    @property
    def native_value(self) -> int | None:
        if not self.coordinator.data:
            return None
        return self.coordinator.data["cards"]["total"]

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        cards = self.coordinator.data["cards"]
        by_source = cards["by_source"]
        return {
            # Cards without a vector cannot be found, so a gap between these two
            # means the embedding service was unreachable while they were stored.
            "searchable": cards["searchable"],
            "seeded": by_source.get("seed", 0),
            "learned": by_source.get("learned", 0),
        }


class HandledSensor(_Base):
    """Share of recent requests kassistant recognised confidently.

    Counts decisions, not executions -- so the number means the same thing
    before and after you switch the mode to active.
    """

    _attr_translation_key = "handled"
    _attr_icon = "mdi:lightning-bolt"
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, coordinator: KassistantCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_handled"

    @property
    def native_value(self) -> float | None:
        if not self.coordinator.data:
            return None
        return self.coordinator.data["decisions"]["handled_pct"]

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        decisions = self.coordinator.data["decisions"]
        return {
            "sampled": decisions["sampled"],
            "recognised": decisions["handled"],
            # The best similarity found per request, averaged. Compare it with
            # the configured threshold: far below means the threshold is out of
            # reach, hovering just under means it is set slightly too high.
            "average_score": decisions.get("avg_score"),
            "average_lookup_ms": decisions.get("avg_latency_ms"),
            "by_outcome": decisions.get("tiers"),
        }
