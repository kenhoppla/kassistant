"""The setup dialog.

If a probe failure did not come back as a visible error, the dialog would look
like it does nothing when you press OK -- so the error path matters as much as
the happy one.
"""

from __future__ import annotations

import json

from homeassistant import config_entries
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

URL = "http://localhost:11434"
INPUT = {
    "embed_url": URL,
    "embed_model": "embeddinggemma",
    "fallback_agent": conversation.HOME_ASSISTANT_AGENT,
}


async def start(hass: HomeAssistant):
    return await hass.config_entries.flow.async_init(
        "kassistant", context={"source": config_entries.SOURCE_USER}
    )


async def test_the_form_opens(hass: HomeAssistant, custom_integration) -> None:
    result = await start(hass)

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert not result["errors"]


async def test_a_working_ollama_creates_the_entry(
    hass: HomeAssistant, custom_integration, aioclient_mock: AiohttpClientMocker
) -> None:
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    aioclient_mock.post(f"{URL}/api/embed", json={"embeddings": [[1.0, 0.0, 0.0, 0.0]]})

    result = await hass.config_entries.flow.async_configure(
        (await start(hass))["flow_id"], INPUT
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"]["embed_model"] == "embeddinggemma"
    assert result["options"]["mode"] == "observe"


async def test_an_unreachable_ollama_is_reported_not_swallowed(
    hass: HomeAssistant, custom_integration, aioclient_mock: AiohttpClientMocker
) -> None:
    """Pressing OK must never look like nothing happened."""
    aioclient_mock.post(f"{URL}/api/embed", exc=TimeoutError())

    result = await hass.config_entries.flow.async_configure(
        (await start(hass))["flow_id"], INPUT
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}


async def test_a_missing_model_is_reported(
    hass: HomeAssistant, custom_integration, aioclient_mock: AiohttpClientMocker
) -> None:
    """Ollama answers 200 with an error body when the model is not pulled."""
    aioclient_mock.post(
        f"{URL}/api/embed",
        json={"error": 'model "embeddinggemma" not found, try pulling it first'},
    )

    result = await hass.config_entries.flow.async_configure(
        (await start(hass))["flow_id"], INPUT
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}


async def test_the_error_code_has_a_translation(hass: HomeAssistant) -> None:
    """An untranslated code would show as a blank line under the form."""
    import pathlib

    strings = json.loads(
        (
            pathlib.Path(__file__).resolve().parents[2]
            / "custom_components"
            / "kassistant"
            / "strings.json"
        ).read_text(encoding="utf-8")
    )

    assert "cannot_connect" in strings["config"]["error"]
    assert strings["config"]["error"]["cannot_connect"]


async def test_only_one_instance_is_allowed(
    hass: HomeAssistant, custom_integration
) -> None:
    MockConfigEntry(domain="kassistant", data=INPUT).add_to_hass(hass)

    result = await start(hass)

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "single_instance_allowed"


async def test_every_field_can_be_rendered_by_the_frontend(
    hass: HomeAssistant, custom_integration
) -> None:
    """A selector the frontend cannot render blanks the entire dialog.

    ConversationAgentSelector is exactly such a case: valid on the backend,
    unrenderable in a config flow, and the symptom is a popup with nothing in it
    but a title and an OK button. So assert on the wire format the frontend
    actually receives.
    """
    import voluptuous_serialize
    from homeassistant.helpers import config_validation as cv

    result = await start(hass)
    fields = voluptuous_serialize.convert(
        result["data_schema"], custom_serializer=cv.custom_serializer
    )

    assert [field["name"] for field in fields] == [
        "embed_url",
        "embed_model",
        "fallback_agent",
    ]
    # Selector types that have shipped in the frontend for years.
    renderable = {"text", "entity", "number", "boolean", "select"}
    for field in fields:
        kind = next(iter(field["selector"]))
        assert kind in renderable, f"{field['name']} uses unrenderable {kind!r}"
