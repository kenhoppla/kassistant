"""Loads single integration modules without pulling in Home Assistant.

``store`` and ``text`` depend on numpy only, so they can be tested in isolation
-- and that is exactly where the logic lives that you cannot verify by reading.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

COMPONENT = (
    pathlib.Path(__file__).resolve().parent.parent / "custom_components" / "kassistant"
)


def load(name: str) -> types.ModuleType:
    """Load a single module of the integration.

    Registered under a prefixed name so a module called ``store`` or ``text``
    cannot shadow an unrelated package for the rest of the test session.
    """
    key = f"kassistant_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, COMPONENT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module
