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

    if problems:
        print("\n".join(problems))
        return 1

    print(f"{len(translations)} translations complete ({len(expected)} keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
