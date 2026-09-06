"""Fixtures for the tests that need the full Home Assistant package.

Kept apart from the fast unit tests, which run on numpy alone. Run these with:

    uv run pytest tests/integration
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

# The integration imports itself as a package, so custom_components has to be
# importable the same way Home Assistant makes it importable at runtime.
sys.path.insert(0, str(REPO / "custom_components"))

# pytest-homeassistant-custom-component registers itself through a pytest11
# entry point, so it loads on its own when installed -- and the fast unit
# tests keep working when it is not.


@pytest.fixture(scope="session", autouse=True)
def link_into_test_config() -> object:
    """Make Home Assistant's loader see kassistant as an installed component.

    Home Assistant only scans the ``custom_components`` folder of its config
    directory, and under pytest that directory lives inside the
    pytest-homeassistant-custom-component package. A symlink created for the
    duration of the session is the least invasive way to bridge that -- it needs
    no copy step and nothing to keep in sync.
    """
    from pytest_homeassistant_custom_component.common import get_test_config_dir

    target = pathlib.Path(get_test_config_dir("custom_components")) / "kassistant"
    source = REPO / "custom_components" / "kassistant"

    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    if not target.exists():
        target.symlink_to(source, target_is_directory=True)
        created = True

    yield target

    if created:
        target.unlink()


@pytest.fixture(autouse=True)
def fresh_card_box() -> object:
    """Give every test an empty card box.

    kassistant stores its database in Home Assistant's config directory, which
    is correct in production but is a fixed, shared folder under pytest. Without
    this, cards pile up from test to test -- and across whole pytest runs, which
    makes assertions about how many cards were stored quietly meaningless.
    """
    from pytest_homeassistant_custom_component.common import get_test_config_dir

    config_dir = pathlib.Path(get_test_config_dir())

    def wipe() -> None:
        for leftover in config_dir.glob("kassistant.db*"):
            leftover.unlink()

    wipe()
    yield
    wipe()


@pytest.fixture
def custom_integration(enable_custom_integrations: None) -> None:
    """Home Assistant refuses to load custom components in tests without this.

    Requested explicitly rather than autouse, because the API-surface tests are
    synchronous and must not drag in the async ``hass`` fixture.
    """
    return enable_custom_integrations
