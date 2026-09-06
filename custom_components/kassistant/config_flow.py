"""UI-driven setup -- no YAML required."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import selector
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CONF_EMBED_MODEL,
    CONF_EMBED_URL,
    CONF_FALLBACK_AGENT,
    CONF_LEARN,
    CONF_MODE,
    CONF_THRESHOLD,
    DEFAULT_EMBED_MODEL,
    DEFAULT_EMBED_URL,
    DEFAULT_LEARN,
    DEFAULT_MODE,
    DEFAULT_THRESHOLD,
    DOMAIN,
    MODES,
)
from .embeddings import EmbeddingError, OllamaEmbeddings

_LOGGER = logging.getLogger(__name__)


def _schema(exclude_agent: str | None = None) -> vol.Schema:
    """The connection form.

    On reconfigure kassistant's own agent already exists and would show up in
    the list -- picking it would make it call itself forever, so it is taken
    out rather than caught later.
    """
    agent = selector.EntitySelectorConfig(domain="conversation")
    if exclude_agent:
        agent["exclude_entities"] = [exclude_agent]
    return vol.Schema(
        {
            vol.Required(
                CONF_EMBED_URL, default=DEFAULT_EMBED_URL
            ): selector.TextSelector(
                selector.TextSelectorConfig(type=selector.TextSelectorType.URL)
            ),
            vol.Required(
                CONF_EMBED_MODEL, default=DEFAULT_EMBED_MODEL
            ): selector.TextSelector(),
            # An entity selector, not ConversationAgentSelector: the latter
            # exists in the backend but the frontend has no renderer for it
            # inside a config flow, and one unrenderable field blanks the whole
            # dialog. Every modern conversation agent is an entity anyway.
            vol.Required(CONF_FALLBACK_AGENT): selector.EntitySelector(agent),
        }
    )


STEP_USER_SCHEMA = _schema()


class KassistantConfigFlow(ConfigFlow, domain=DOMAIN):
    """Walks the user through the initial setup."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        # Only one instance. All entries would share the same card box file but
        # keep separate in-memory indexes, and vectors from two different
        # embedding models are not comparable at all.
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        errors: dict[str, str] = {}

        if user_input is not None:
            error = await self._async_probe_embeddings(
                user_input[CONF_EMBED_URL], user_input[CONF_EMBED_MODEL]
            )
            if error is not None:
                errors["base"] = error
            else:
                return self.async_create_entry(
                    title="kassistant",
                    data=user_input,
                    options={
                        CONF_MODE: DEFAULT_MODE,
                        CONF_THRESHOLD: DEFAULT_THRESHOLD,
                        CONF_LEARN: DEFAULT_LEARN,
                    },
                )

        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                STEP_USER_SCHEMA, user_input or {}
            ),
            errors=errors,
        )

    async def _async_probe_embeddings(self, url: str, model: str) -> str | None:
        """Check that the embedding service answers. Returns an error code."""
        embeddings = OllamaEmbeddings(
            session=async_get_clientsession(self.hass), base_url=url, model=model
        )
        try:
            await embeddings.embed_one("Setup test sentence")
        except EmbeddingError as err:
            _LOGGER.debug("Probe failed: %s", err)
            return "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected error while probing")
            return "unknown"
        return None

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Change the Ollama address, the model or the fallback agent.

        These live in the entry data rather than the options, so without this
        step a typo in the address -- or moving Ollama to another machine --
        would mean deleting the integration and losing the card box statistics
        along with it.
        """
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}

        if user_input is not None:
            error = await self._async_probe_embeddings(
                user_input[CONF_EMBED_URL], user_input[CONF_EMBED_MODEL]
            )
            if error is not None:
                errors["base"] = error
            else:
                return self.async_update_reload_and_abort(
                    entry, data_updates=user_input
                )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                _schema(exclude_agent=_own_agent(self.hass, entry)),
                user_input or dict(entry.data),
            ),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return KassistantOptionsFlow()


class KassistantOptionsFlow(OptionsFlow):
    """Settings that can be changed after setup."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(data=user_input)

        options = self.config_entry.options
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_MODE, default=options.get(CONF_MODE, DEFAULT_MODE)
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=MODES,
                        translation_key="mode",
                        mode=selector.SelectSelectorMode.LIST,
                    )
                ),
                vol.Required(
                    CONF_THRESHOLD,
                    default=options.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.5,
                        max=1.0,
                        step=0.01,
                        mode=selector.NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Required(
                    CONF_LEARN, default=options.get(CONF_LEARN, DEFAULT_LEARN)
                ): selector.BooleanSelector(),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema)


@callback
def _own_agent(hass: HomeAssistant, entry: ConfigEntry) -> str | None:
    """kassistant's own conversation entity, if it exists yet."""
    return er.async_get(hass).async_get_entity_id(
        "conversation", DOMAIN, entry.entry_id
    )
