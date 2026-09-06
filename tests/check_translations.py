"""Check that every translation carries the same keys as strings.json.

Missing keys otherwise only surface when a user stares at an empty field.
"""

from __future__ import annotations

import json
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent.parent
COMPONENT = BASE / "custom_components" / "kassistant"


def keys(node: object, prefix: str = "") -> set[str]:
    """All leaf paths of a nested mapping."""
    if not isinstance(node, dict):
        return {prefix}
    found: set[str] = set()
    for key, value in node.items():
        found |= keys(value, f"{prefix}.{key}" if prefix else key)
    return found


def translation_keys_are_wired() -> list[str]:
    """Entity translations must match the keys the code actually uses.

    Writing a translated name without pointing an entity at it is invisible:
    the file looks complete, every language agrees, and the name never shows up
    anywhere. The reverse -- a key in the code with no translation -- leaves a
    blank name in the interface.
    """
    used: set[str] = set()
    for source in COMPONENT.glob("*.py"):
        for line in source.read_text(encoding="utf-8").splitlines():
            if "_attr_translation_key" in line and "=" in line:
                value = line.split("=", 1)[1].strip().strip("\"'")
                if value and not value.startswith("_"):
                    used.add(value)

    strings = json.loads((COMPONENT / "strings.json").read_text(encoding="utf-8"))
    declared = {
        key for platform in strings.get("entity", {}).values() for key in platform
    }

    problems = []
    if missing := used - declared:
        problems.append(f"used in code but not translated: {sorted(missing)}")
    if unused := declared - used:
        problems.append(f"translated but never used: {sorted(unused)}")
    return problems


def main() -> int:
    reference = json.loads((COMPONENT / "strings.json").read_text(encoding="utf-8"))
    expected = keys(reference)

    problems: list[str] = []
    translations = sorted((COMPONENT / "translations").glob("*.json"))
    if not translations:
        print("no translations found")
        return 1

    for path in translations:
        actual = keys(json.loads(path.read_text(encoding="utf-8")))
        if missing := expected - actual:
            problems.append(f"{path.name}: missing {sorted(missing)}")
        if extra := actual - expected:
            problems.append(f"{path.name}: unknown {sorted(extra)}")

    problems.extend(translation_keys_are_wired())

    if problems:
        print("\n".join(problems))
        return 1

    print(f"{len(translations)} translations complete ({len(expected)} keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
