"""What every entity of this integration has in common."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo

from .const import DOMAIN


def device_info(entry: ConfigEntry) -> DeviceInfo:
    """One device holding every entity of this integration.

    Grouping them matters twice over: the interface shows the agent and its
    diagnostics together, and entity names can come from the device plus a
    translated part instead of an English string baked into the code.
    """
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name="kassistant",
        manufacturer="kassistant",
        entry_type=DeviceEntryType.SERVICE,
    )
