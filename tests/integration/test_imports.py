"""Every Home Assistant interface kassistant relies on must actually exist.

This is the guard against silent API drift: when a Home Assistant release
renames or removes something we use, this test fails instead of the voice
pipeline failing in someone's kitchen.
"""

from __future__ import annotations

import inspect

from homeassistant.components import conversation
from homeassistant.config_entries import HANDLERS
from homeassistant.const import MATCH_ALL, Platform
from homeassistant.helpers import intent, selector


def test_all_modules_import() -> None:
    import kassistant
    from kassistant import (  # noqa: F401
        config_flow,
        const,
        data,
        embeddings,
        learn,
        router,
        store,
        text,
    )
    from kassistant import conversation as agent_module  # noqa: F401

    assert kassistant.PLATFORMS == [Platform.CONVERSATION]


def test_config_flow_is_registered() -> None:
    from kassistant.config_flow import KassistantConfigFlow

    assert HANDLERS.get("kassistant") is KassistantConfigFlow


def test_agent_declares_the_expected_surface() -> None:
    from kassistant.conversation import KassistantAgent

    assert issubclass(KassistantAgent, conversation.ConversationEntity)
    # The method name changed once already; async_process is deprecated.
    assert "_async_handle_message" in KassistantAgent.__dict__


def test_conversation_input_carries_what_we_forward() -> None:
    """We pass device_id and satellite_id on to the fallback agent."""
    fields = conversation.ConversationInput.__dataclass_fields__
    assert {"text", "context", "conversation_id", "language", "device_id"} <= set(
        fields
    )


def test_async_converse_accepts_our_arguments() -> None:
    parameters = inspect.signature(conversation.async_converse).parameters
    for name in (
        "text",
        "conversation_id",
        "context",
        "language",
        "agent_id",
        "device_id",
    ):
        assert name in parameters, name


def test_intent_helpers_exist() -> None:
    assert hasattr(intent.IntentResponse, "async_set_speech")
    assert hasattr(intent.IntentResponse, "async_set_error")
    assert intent.IntentResponseErrorCode.UNKNOWN
    assert "slots" in inspect.signature(intent.async_handle).parameters


def test_selectors_exist() -> None:
    assert selector.ConversationAgentSelector.selector_type == "conversation_agent"
    assert selector.TextSelectorType.URL
    assert selector.NumberSelectorMode.SLIDER
    assert selector.SelectSelectorMode.LIST


def test_supported_languages_constant() -> None:
    assert MATCH_ALL == "*"
