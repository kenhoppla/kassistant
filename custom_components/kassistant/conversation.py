"""The conversation agent Home Assistant offers in the voice pipeline.

How a request flows:

1. The router looks the sentence up in the card box (except in "observe" mode,
   where we do not even look).
2. On a confident hit *and* in "active" mode, kassistant runs the stored action
   itself -- that takes milliseconds.
3. In every other case the request is passed on unchanged to the configured
   fallback agent. While it works, we record which actions it triggers and turn
   them into a new card.

Step 3 is why kassistant gets faster over time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_call_later

from .const import (
    CONF_LEARN,
    CONF_LEARN_DELAY,
    CONF_MODE,
    DEFAULT_LEARN,
    DEFAULT_LEARN_DELAY,
    DEFAULT_MODE,
    DOMAIN,
    MODE_ACTIVE,
    MODE_OBSERVE,
)
from .data import KassistantData
from .router import TIER_OBSERVE, Decision
from .store import SOURCE_LEARNED

_LOGGER = logging.getLogger(__name__)

ACTION_TYPE_ACTIONS = "actions"
ACTION_TYPE_INTENT = "intent"


class PartialExecutionError(Exception):
    """A card failed after some of its calls had already gone through.

    Retrying such a card through the fallback agent would repeat the calls that
    already succeeded, so this case must not fall back.
    """


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation agent for this entry."""
    async_add_entities([KassistantAgent(entry)])


class KassistantAgent(conversation.ConversationEntity):
    """Front-line agent with a card box and a learning loop."""

    # No device is created for this entry, so the entity carries its own name.
    # With has_entity_name the name would come from a device, and without one
    # Home Assistant falls back to the unique id -- which would leave users with
    # an entity called conversation.kassistant_01m1sdqdby2y83aqzz0ezt04mk.
    _attr_name = "kassistant"
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry: ConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = entry.entry_id
        # Per conversation, the scheduled learning job, so an immediate
        # follow-up question can cancel it.
        self._pending_learn: dict[str, Any] = {}
        # Learning jobs already running. They write to the card box, so teardown
        # waits for them -- otherwise they land in a closed database.
        self._learning: set[asyncio.Task[None]] = set()

    @property
    def supported_languages(self) -> list[str] | str:
        """We speak whatever the fallback agent speaks."""
        return MATCH_ALL

    @property
    def _data(self) -> KassistantData:
        return self._entry.runtime_data

    @property
    def _mode(self) -> str:
        return self._entry.options.get(CONF_MODE, DEFAULT_MODE)

    # -- Core -----------------------------------------------------------------

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Answer a request -- ourselves or via the fallback."""
        data = self._data
        mode = self._mode
        language = user_input.language or self.hass.config.language

        self._cancel_pending_learn(user_input.conversation_id)

        if mode == MODE_OBSERVE:
            decision = Decision(tier=TIER_OBSERVE)
        else:
            decision = await data.router.decide(user_input.text, language)

        if decision.is_fastpath and mode == MODE_ACTIVE:
            return await self._handle_fastpath(user_input, decision, language)

        # MODE_SHADOW lands here on purpose: the decision was made and gets
        # logged below, but only MODE_ACTIVE is ever allowed to act on it.
        return await self._handle_delegate(user_input, decision, language, mode)

    # -- Fast path ------------------------------------------------------------

    async def _handle_fastpath(
        self,
        user_input: conversation.ConversationInput,
        decision: Decision,
        language: str,
    ) -> conversation.ConversationResult:
        """Run the stored action ourselves."""
        data = self._data
        started = time.monotonic()
        action = decision.action or {}

        card = decision.match.example_id if decision.match else "?"

        try:
            response = await self._execute_action(action, user_input, language)
        except PartialExecutionError:
            # Some calls already went through. Handing this to the fallback
            # agent would run them a second time, so we stop and say so.
            _LOGGER.exception("Card %s failed halfway through", card)
            return self._error_result(
                user_input,
                language,
                "Only part of that worked. Please check and try again.",
            )
        except Exception:
            _LOGGER.exception("Card %s could not be executed, passing on", card)
            return await self._handle_delegate(
                user_input, decision, language, self._mode, fastpath_failed=True
            )

        latency = int((time.monotonic() - started) * 1000)

        if decision.match is not None:
            await self.hass.async_add_executor_job(
                data.store.mark_hit, decision.match.example_id
            )

        await self.hass.async_add_executor_job(
            _log_decision,
            data.store,
            {
                "utterance": user_input.text,
                "language": language,
                "mode": self._mode,
                "tier": decision.tier,
                "score": decision.score,
                "example_id": decision.match.example_id if decision.match else None,
                "proposed": action,
                "executed": True,
                "latency_ms": latency,
            },
        )

        return conversation.ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
        )

    async def _execute_action(
        self,
        action: dict[str, Any],
        user_input: conversation.ConversationInput,
        language: str,
    ) -> intent.IntentResponse:
        """Replay a stored action."""
        response = intent.IntentResponse(language=language)
        action_type = action.get("type")

        if action_type == ACTION_TYPE_INTENT:
            return await intent.async_handle(
                self.hass,
                DOMAIN,
                action["intent"],
                action.get("slots") or {},
                user_input.text,
                user_input.context,
                language,
            )

        if action_type != ACTION_TYPE_ACTIONS:
            raise ValueError(f"Unknown action type: {action_type!r}")

        for index, call in enumerate(action.get("actions") or []):
            try:
                await self.hass.services.async_call(
                    call["domain"],
                    call["service"],
                    call.get("data") or {},
                    blocking=True,
                    context=user_input.context,
                )
            except Exception as err:
                if index:
                    raise PartialExecutionError(
                        f"{index} of {len(action['actions'])} calls already ran"
                    ) from err
                raise

        # Not localised yet. "Ok" is understood widely enough to ship with,
        # but a proper per-language confirmation is still owed.
        response.async_set_speech("Ok")
        return response

    # -- Passing on and learning ----------------------------------------------

    async def _handle_delegate(
        self,
        user_input: conversation.ConversationInput,
        decision: Decision,
        language: str,
        mode: str,
        fastpath_failed: bool = False,
    ) -> conversation.ConversationResult:
        """Hand the request to the fallback agent and take notes."""
        data = self._data
        recorder = data.recorder
        context_id = user_input.context.id

        if data.fallback_agent == self.entity_id:
            # Would otherwise call itself forever. Can only happen if someone
            # edits the configuration by hand.
            return self._error_result(
                user_input,
                language,
                "kassistant is configured as its own fallback agent. "
                "Please pick a different agent in the settings.",
            )

        # Pass the origin along so the fallback agent knows which room was
        # spoken from. satellite_id only exists in newer core versions, so it is
        # forwarded only when present.
        extra: dict[str, Any] = {"device_id": user_input.device_id}
        if (satellite_id := getattr(user_input, "satellite_id", None)) is not None:
            extra["satellite_id"] = satellite_id

        recorder.watch(context_id)
        started = time.monotonic()
        try:
            result = await conversation.async_converse(
                self.hass,
                text=user_input.text,
                conversation_id=user_input.conversation_id,
                context=user_input.context,
                language=user_input.language,
                agent_id=data.fallback_agent,
                **extra,
            )
        finally:
            observed = recorder.collect(context_id)

        latency = int((time.monotonic() - started) * 1000)
        tier = "fastpath_failed" if fastpath_failed else decision.tier

        decision_id = await self.hass.async_add_executor_job(
            _log_decision,
            data.store,
            {
                "utterance": user_input.text,
                "language": language,
                "mode": mode,
                "tier": tier,
                "score": decision.score,
                "example_id": decision.match.example_id if decision.match else None,
                "proposed": decision.action,
                "executed": False,
                "observed": observed,
                "latency_ms": latency,
            },
        )

        if observed and self._entry.options.get(CONF_LEARN, DEFAULT_LEARN):
            # Use the conversation id Home Assistant handed back, not the one
            # that came in: a first utterance arrives without one, and a
            # follow-up would then be unable to cancel this.
            self._schedule_learn(
                user_input, language, observed, decision_id, result.conversation_id
            )

        return result

    @callback
    def _schedule_learn(
        self,
        user_input: conversation.ConversationInput,
        language: str,
        observed: list[dict[str, Any]],
        decision_id: int,
        conversation_id: str | None,
    ) -> None:
        """Remember the sentence -- but only if no objection follows shortly.

        If the user says something else in the same conversation right away, the
        answer probably was not what they wanted. Then we prefer to learn
        nothing. This is deliberately cautious: a missing card only costs time,
        a wrong one costs trust.

        A one-shot voice command carries no conversation id on the way in, so
        the key comes from the id Home Assistant assigned on the way out -- that
        is the one a follow-up will arrive with. The context id is only a last
        resort, and cancellation cannot work in that case.
        """
        key = conversation_id or user_input.context.id

        delay = float(self._entry.options.get(CONF_LEARN_DELAY, DEFAULT_LEARN_DELAY))
        utterance = user_input.text
        action = {"type": ACTION_TYPE_ACTIONS, "actions": observed}

        async def _learn() -> None:
            example_id = await self._data.router.remember(
                utterance=utterance,
                language=language,
                action=action,
                source=SOURCE_LEARNED,
            )
            _LOGGER.debug("Learned: %r -> %s (card %s)", utterance, action, example_id)
            if example_id is not None:
                await self.hass.async_add_executor_job(
                    self._data.store.attach_observation, decision_id, action
                )

        @callback
        def _fire(_now: Any) -> None:
            # Track the job, because it writes to the card box: teardown has to
            # be able to wait for it before the database is closed.
            self._pending_learn.pop(key, None)
            task = self.hass.async_create_task(_learn(), eager_start=False)
            self._learning.add(task)
            task.add_done_callback(self._learning.discard)

        self._pending_learn[key] = async_call_later(self.hass, delay, _fire)

    @callback
    def _cancel_pending_learn(self, conversation_id: str | None) -> None:
        """Cancel a scheduled learning job (the user is following up)."""
        if conversation_id is None:
            return
        if (cancel := self._pending_learn.pop(conversation_id, None)) is not None:
            cancel()
            _LOGGER.debug(
                "Follow-up in the same conversation -- previous sentence not learned"
            )

    @callback
    def _error_result(
        self,
        user_input: conversation.ConversationInput,
        language: str,
        message: str,
    ) -> conversation.ConversationResult:
        """Build an error response the user also hears spoken."""
        response = intent.IntentResponse(language=language)
        response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, message)
        return conversation.ConversationResult(
            response=response, conversation_id=user_input.conversation_id
        )

    async def async_will_remove_from_hass(self) -> None:
        """Stop learning and wait for anything already writing.

        Runs before the config entry closes the card box, so a job in flight
        must finish here rather than into a database that is about to go away.
        """
        for cancel in self._pending_learn.values():
            cancel()
        self._pending_learn.clear()

        if self._learning:
            await asyncio.gather(*self._learning, return_exceptions=True)
            self._learning.clear()


def _log_decision(store, fields: dict[str, Any]) -> int:
    return store.log_decision(**fields)
