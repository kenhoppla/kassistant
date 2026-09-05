"""Constants for kassistant."""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "kassistant"

# --- Config entry keys -------------------------------------------------------
CONF_EMBED_URL: Final = "embed_url"
CONF_EMBED_MODEL: Final = "embed_model"
CONF_FALLBACK_AGENT: Final = "fallback_agent"

# --- Option keys (changeable after setup) ------------------------------------
CONF_MODE: Final = "mode"
CONF_THRESHOLD: Final = "threshold"
CONF_LEARN: Final = "learn"
CONF_LEARN_DELAY: Final = "learn_delay"

# --- Modes -------------------------------------------------------------------
# Stage 1: listen and take notes only. Every request is passed through to the
#          fallback agent unchanged. Cannot break anything.
MODE_OBSERVE: Final = "observe"
# Stage 2: also decide what kassistant *would* have done and log it. Still
#          nothing is executed.
MODE_SHADOW: Final = "shadow"
# Stage 3: execute confident matches directly, pass everything else on.
MODE_ACTIVE: Final = "active"

MODES: Final = [MODE_OBSERVE, MODE_SHADOW, MODE_ACTIVE]

# --- Defaults ----------------------------------------------------------------
DEFAULT_EMBED_URL: Final = "http://localhost:11434"
DEFAULT_EMBED_MODEL: Final = "embeddinggemma"
DEFAULT_MODE: Final = MODE_OBSERVE
DEFAULT_THRESHOLD: Final = 0.92
DEFAULT_LEARN: Final = True
# Seconds to wait for the user to object before a card is considered good enough
# to keep.
DEFAULT_LEARN_DELAY: Final = 25.0

# --- Storage -----------------------------------------------------------------
DB_FILENAME: Final = "kassistant.db"

# Service calls in these domains never count as a learned action. They are the
# side noise of speaking a reply, not something the user asked for.
IGNORED_ACTION_DOMAINS: Final = frozenset(
    {
        "persistent_notification",
        "logbook",
        "system_log",
        "tts",
        "conversation",
        "recorder",
        DOMAIN,
    }
)
