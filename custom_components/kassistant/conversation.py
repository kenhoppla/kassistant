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

import logging
import time
from typing import Any

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_LEARN,
    CONF_MODE,
    DEFAULT_LEARN,
    DEFAULT_MODE,
    DOMAIN,
    MODE_ACTIVE,
    MODE_OBSERVE,
)
from .data import KassistantData
from .entity import device_info
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

    # The name comes from the device. Getting this wrong once produced an
    # entity called conversation.kassistant_01m1sdqdby2y83aqzz0ezt04mk: with
    # has_entity_name and no device, Home Assistant falls back to the unique id.
    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry: ConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = device_info(entry)

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

        # Nothing to learn from a sentence the card box already recognises. In
        # shadow mode every request is passed on even when the router knew the
        # answer, and storing it again would leave a second card for the same
        # sentence -- one that can never be confirmed, because the card that
        # already answers keeps winning the match.
        already_known = decision.is_fastpath
        if (
            observed
            and not already_known
            and self._entry.options.get(CONF_LEARN, DEFAULT_LEARN)
        ):
            await self._async_learn(user_input, language, observed, decision_id)

        return result

    async def _async_learn(
        self,
        user_input: conversation.ConversationInput,
        language: str,
        observed: list[dict[str, Any]],
        decision_id: int,
    ) -> None:
        """Remember what the fallback agent just did.

        Stored straight away rather than after a pause. Nothing has to be judged
        here: a card only answers on its own once the same sentence has produced
        the same action a second time, so a one-off mistake by the fallback agent
        is filed and never used.

        Awaited rather than run in the background. It costs one embedding call --
        milliseconds against the seconds the fallback agent just spent -- and in
        exchange there is no timer to cancel and no job that can outlive the card
        box it writes into.
        """
        action = {"type": ACTION_TYPE_ACTIONS, "actions": observed}
        example_id = await self._data.router.remember(
            utterance=user_input.text,
            language=language,
            action=action,
            source=SOURCE_LEARNED,
        )
        _LOGGER.debug(
            "Learned: %r -> %s (card %s)", user_input.text, action, example_id
        )
        if example_id is not None:
            await self.hass.async_add_executor_job(
                self._data.store.attach_observation, decision_id, action
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


def _log_decision(store, fields: dict[str, Any]) -> int:
    return store.log_decision(**fields)
