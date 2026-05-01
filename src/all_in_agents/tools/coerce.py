"""Safe type coercion for tool arguments based on JSON Schema."""

from __future__ import annotations

import re

_INT_RE = re.compile(r"^-?\d+$")

_BOOL_MAP = {
    "true": True,
    "false": False,
    "1": True,
    "0": False,
}


def coerce_args(args: dict, schema: dict) -> dict:
    """Return a new dict with values coerced to match *schema* types.

    Only string values are coerced; non-string values pass through unchanged.
    If a coercion fails the original value is kept so that downstream
    jsonschema validation can produce a clear error.
    """
    properties = schema.get("properties", {})
    out = dict(args)
    for key, value in out.items():
        if not isinstance(value, str):
            continue
        prop = properties.get(key)
        if prop is None:
            continue
        declared_type = prop.get("type")
        if declared_type == "integer":
            if _INT_RE.match(value):
                out[key] = int(value)
        elif declared_type == "number":
            try:
                out[key] = float(value)
            except (ValueError, OverflowError):
                pass
        elif declared_type == "boolean":
            lowered = value.lower()
            if lowered in _BOOL_MAP:
                out[key] = _BOOL_MAP[lowered]
    return out
