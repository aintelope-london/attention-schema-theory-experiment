"""UI schema utilities — parses @ui annotations from default_config.yaml."""

from pathlib import Path
from typing import Optional


_CONFIG_PATH = Path(__file__).parents[1] / "config" / "default_config.yaml"


def load_ui_schema() -> dict:
    """Parse @ui annotations from default_config.yaml into a nested spec dict.

    Format: key: value  # @ui <type> [<min> <max> | <choice1>,<choice2>,...]

    - @ui bool            → [None, "bool"]           → checkbox
    - @ui str             → [None, "str"]            → free text entry
    - @ui str a,b,c       → [["a","b","c"], "str"]   → combobox
    - @ui int 0 100       → [[0, 100], "int"]        → spinbox
    - @ui float 0.0 1.0   → [[0.0, 1.0], "float"]   → spinbox
    - (no @ui)            → [None, "locked"]         → read-only display
    """
    schema = {}
    indent_stack = []  # [(indent_level, key_name)]

    for line in _CONFIG_PATH.read_text().splitlines():
        stripped = line.lstrip()

        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue

        indent = len(line) - len(stripped)
        key, rest = stripped.split(":", 1)
        key = key.strip()

        while indent_stack and indent_stack[-1][0] >= indent:
            indent_stack.pop()

        path = [k for _, k in indent_stack] + [key]

        # Section header vs leaf
        rest_stripped = rest.lstrip()
        if not rest_stripped or rest_stripped.startswith("#"):
            indent_stack.append((indent, key))
            continue

        # Leaf — extract spec from @ui annotation or default to locked
        spec = _parse_annotation(rest) if "@ui" in rest else [None, "locked"]
        _set_nested(schema, path, spec)

    return schema


def get_field_spec(schema: dict, key_path: str) -> Optional[list]:
    """Navigate nested schema dict by dotted path. Returns spec list or None."""
    current = schema
    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]

    # Intermediate node (dict) is a branch, not a leaf spec
    return None if isinstance(current, dict) else current


def _parse_annotation(rest: str) -> list:
    """Extract spec from the portion after ':' containing @ui."""
    text = rest.split("@ui", 1)[1].strip()
    parts = text.split()
    type_str = parts[0]

    if type_str == "bool":
        return [None, "bool"]
    if type_str == "str":
        return [parts[1].split(","), "str"] if len(parts) > 1 else [None, "str"]
    if type_str in ("int", "float"):
        cast = int if type_str == "int" else float
        return [[cast(parts[1]), cast(parts[2])], type_str]
    return [None, "locked"]


def _set_nested(d: dict, keys: list, value):
    """Set value in nested dict, creating intermediate dicts."""
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value