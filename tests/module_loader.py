"""Loads single integration modules without pulling in Home Assistant.

``store``, ``text`` and ``embeddings`` depend on numpy at most, so they can be
tested in isolation -- and that is exactly where the logic lives that cannot be
verified by reading.

They import each other relatively (``from .text import normalize``), which only
works inside a package. Importing the real package would execute its
``__init__.py`` and drag in all of Home Assistant, so a stand-in package is put
in its place: same directory, no module body.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

COMPONENT = (
    pathlib.Path(__file__).resolve().parent.parent / "custom_components" / "kassistant"
)

# Prefixed so a module called "store" or "text" cannot shadow an unrelated
# package for the rest of the test session.
PACKAGE = "kassistant_isolated"


def _package() -> types.ModuleType:
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(COMPONENT)]
        sys.modules[PACKAGE] = package
    return sys.modules[PACKAGE]


def load(name: str) -> types.ModuleType:
    """Load a single module of the integration."""
    _package()
    qualified = f"{PACKAGE}.{name}"
    if qualified in sys.modules:
        return sys.modules[qualified]

    spec = importlib.util.spec_from_file_location(qualified, COMPONENT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module
