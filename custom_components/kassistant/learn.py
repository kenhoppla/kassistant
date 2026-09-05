"""Record what actually happened in response to a sentence.

When the fallback agent (typically an LLM) answers a request, it calls services
along the way -- ``light.turn_on`` and friends. Home Assistant attaches the
context of the triggering conversation to each of those calls. That is what we
hook into: remember the context of a request, collect the service calls that
belong to it, and you have the back of the card -- without the fallback agent
needing to know anything about us.
"""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.const import ATTR_DOMAIN, ATTR_SERVICE, EVENT_CALL_SERVICE
from homeassistant.core import Event, HomeAssistant, callback

from .const import IGNORED_ACTION_DOMAINS

_LOGGER = logging.getLogger(__name__)

# Fields of service_data that actually describe an action. Everything else is
# noise when comparing two actions.
_RELEVANT_FIELDS = frozenset(
    {
        "entity_id",
        "area_id",
        "device_id",
        "floor_id",
        "label_id",
        "brightness",
        "brightness_pct",
        "color_temp_kelvin",
        "color_name",
        "rgb_color",
        "temperature",
        "hvac_mode",
        "preset_mode",
        "fan_mode",
        "position",
        "tilt_position",
        "volume_level",
        "media_content_id",
        "option",
        "value",
    }
)

# Recording more calls than this per request does not produce a usable card --
# that is an automation rather than a voice command.
MAX_ACTIONS = 8


class ActionRecorder:
    """Attributes service calls to the conversation that triggered them."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass
        # Context id -> collected actions. Only populated while a request is in
        # flight, empty again afterwards.
        self._watched: dict[str, list[dict[str, Any]]] = {}
        # Child context -> root context, so nested calls (script calls scene
        # calls light) end up with the right conversation.
        self._parents: dict[str, str] = {}
        # Conversations that triggered more calls than we are willing to store.
        self._overflowed: set[str] = set()
        self._unsubscribe: Any = None

    def start(self) -> None:
        self._unsubscribe = self._hass.bus.async_listen(
            EVENT_CALL_SERVICE, self._handle_call_service
        )

    def stop(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._watched.clear()
        self._parents.clear()
        self._overflowed.clear()

    @callback
    def watch(self, context_id: str) -> None:
        """Start collecting calls for this context."""
        self._watched.setdefault(context_id, [])

    @callback
    def collect(self, context_id: str) -> list[dict[str, Any]]:
        """Stop collecting and return the result.

        A conversation that triggered more calls than ``MAX_ACTIONS`` yields
        nothing at all. Returning the first few would produce a card that does
        only part of what the sentence asked for -- and a card that does the
        wrong thing is worse than no card.
        """
        actions = self._watched.pop(context_id, [])
        overflowed = context_id in self._overflowed
        self._overflowed.discard(context_id)
        self._parents = {
            child: root for child, root in self._parents.items() if root != context_id
        }

        if overflowed:
            _LOGGER.debug(
                "Conversation triggered more than %d calls, learning nothing",
                MAX_ACTIONS,
            )
            return []
        return actions

    @callback
    def _handle_call_service(self, event: Event) -> None:
        if not self._watched:
            return

        root = self._resolve_root(event)
        if root is None:
            return

        # Register the ancestry before any filtering. Whatever this call
        # triggers in turn belongs to the same conversation -- even when the
        # call itself is one we do not store, such as a script or a tts action.
        if event.context.id not in self._parents:
            self._parents[event.context.id] = root

        domain = event.data.get(ATTR_DOMAIN)
        service = event.data.get(ATTR_SERVICE)
        if not domain or not service or domain in IGNORED_ACTION_DOMAINS:
            return

        bucket = self._watched.get(root)
        if bucket is None:
            return
        if len(bucket) >= MAX_ACTIONS:
            self._overflowed.add(root)
            return

        bucket.append(
            {
                "domain": domain,
                "service": service,
                "data": _relevant_data(event.data.get("service_data") or {}),
            }
        )

    def _resolve_root(self, event: Event) -> str | None:
        """Work out which watched conversation an event belongs to."""
        context = event.context
        if context.id in self._watched:
            return context.id
        if context.id in self._parents:
            return self._parents[context.id]
        parent = context.parent_id
        if parent is None:
            return None
        if parent in self._watched:
            return parent
        return self._parents.get(parent)


def _relevant_data(service_data: dict[str, Any]) -> dict[str, Any]:
    """Reduce the payload to what defines the action."""
    reduced: dict[str, Any] = {}
    for key, value in service_data.items():
        if key not in _RELEVANT_FIELDS:
            continue
        # A single-element list of entities is the same as the entity itself.
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        reduced[key] = value
    return reduced
