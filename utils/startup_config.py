"""
utils/startup_config.py

Optional YAML startup configuration for llamagraph.

Design notes (MVP separation):
  - This module is pure data/file logic: no Tkinter, no Matplotlib.
  - It implements a small stdlib-only YAML subset parser for flat
    ``key: value`` mappings (no nesting, no anchors, no multi-line
    scalars). This keeps the feature dependency-free on purpose;
    a full YAML library (PyYAML) is intentionally not required.
  - llamagraph.py owns CLI/config merging and passes plain startup
    values into MainWindow/PlotterPresenter (View/Presenter stay dumb).

Config file lookup:
  - ``llamagraph.config.yml`` next to the program script is used
    automatically when present (see find_auto_config()).
  - ``--config <file>`` points at any other config file instead.
  - ``--no-config`` ignores every config file (plain built-in defaults).
  - Explicit CLI options always win over config values.

All user-facing messages, comments, and keys are English.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

#: File name searched automatically next to the program script.
AUTO_CONFIG_FILENAME = "llamagraph.config.yml"

#: Committed, documented example (never auto-loaded).
EXAMPLE_CONFIG_FILENAME = "example.llamagraph.config.yml"

#: Valid surface styles (toolbar combo values).
SURFACE_STYLES = ("Solid", "Shaded", "Colormap")

#: Valid interpolation modes (toolbar combo values).
INTERP_METHODS = ("Cubic", "Cubic+Clamp", "Linear")

#: Valid Z label modes (toolbar combo values).
Z_LABEL_MODES = ("pp", "tg", "both-norm", "%")

#: Built-in startup defaults (identical to the previous no-config behavior).
DEFAULTS: dict[str, Any] = {
    "data_path": ".",
    "latency_ns": False,
    "use_md": True,
    "normalize": False,
    "mode_3d": False,
    "show_level": False,
    "level_value": 50,
    "surface_visible": True,
    "surface_style": "Solid",
    "subdiv_level": 0,
    "interp_method": "Cubic",
    "mask_gaps": False,
    "show_wireframe": False,
    "show_errors": True,
    "show_projections": False,
    "dolly": True,
    "z_label_mode": "both-norm",
    "unify": False,
    "show_pp": True,
    "show_tg": True,
}

_BOOL_TRUE = frozenset({"true", "yes", "on"})
_BOOL_FALSE = frozenset({"false", "no", "off"})


def script_dir() -> Path:
    """Directory containing the running program script."""
    try:
        return Path(sys.argv[0]).absolute().parent
    except Exception:
        return Path.cwd()


def find_auto_config(base_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Return the auto-loaded config path when it exists, else None.

    The search location is the program script directory
    (``<script-dir>/llamagraph.config.yml``), never the current
    working directory, so startup stays independent of where the
    user invokes the tool from.
    """
    directory = base_dir if base_dir is not None else script_dir()
    candidate = Path(directory) / AUTO_CONFIG_FILENAME
    try:
        if candidate.is_file():
            return candidate
    except OSError:
        return None
    return None


def _strip_inline_comment(text: str) -> str:
    """
    Remove a trailing ``#`` comment (only when the ``#`` starts a
    comment, i.e. at the beginning or after whitespace, and outside
    of single/double quotes).
    """
    in_single = False
    in_double = False
    for i, ch in enumerate(text):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if i == 0 or text[i - 1] in (" ", "\t"):
                return text[:i].rstrip()
    return text


def _parse_scalar(raw: str) -> Any:
    """Coerce a flat YAML scalar to bool/int/float/str."""
    text = raw.strip()
    if len(text) >= 2 and (
        (text[0] == '"' and text[-1] == '"')
        or (text[0] == "'" and text[-1] == "'")
    ):
        return text[1:-1]
    lowered = text.lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    if lowered in ("null", "none", "~", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def parse_simple_yaml(text: str, source: str = "<config>") -> dict[str, Any]:
    """
    Parse a flat ``key: value`` YAML subset into a raw dict.

    Supported: one ``key: value`` per line, ``#`` comments, quoted
    strings, booleans (true/false/yes/no/on/off), null, ints,
    floats, and plain strings. Nested blocks, lists (``- item``),
    anchors, and multi-line scalars are rejected with a
    ValueError carrying an English message.
    """
    raw: dict[str, Any] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.strip().startswith("- "):
            raise ValueError(
                f"{source}:{lineno}: list items are not supported "
                f"(flat 'key: value' pairs only)"
            )
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise ValueError(
                f"{source}:{lineno}: tab indentation is not supported "
                f"(use spaces or no indentation)"
            )
        stripped = line.strip()
        if stripped.startswith(":") or ":" not in line:
            raise ValueError(
                f"{source}:{lineno}: expected a 'key: value' pair, "
                f"got {line.strip()!r}"
            )
        indent = len(line) - len(line.lstrip(" "))
        if indent > 0:
            raise ValueError(
                f"{source}:{lineno}: nested mappings are not supported "
                f"(flat 'key: value' pairs only)"
            )
        key, _, value_part = line.partition(":")
        key = key.strip()
        if not key or any(ch.isspace() for ch in key):
            raise ValueError(
                f"{source}:{lineno}: invalid key {key!r} "
                f"(use a single word like 'mode_3d')"
            )
        value_text = _strip_inline_comment(value_part).strip()
        if key in raw:
            raise ValueError(
                f"{source}:{lineno}: duplicate key {key!r}"
            )
        raw[key] = _parse_scalar(value_text)
    return raw


def _warn(message: str) -> None:
    """English warning on stderr (config never fails silently)."""
    print(f"Warning: {message}", file=sys.stderr)


def _coerce_bool(key: str, value: Any, source: str) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _BOOL_TRUE:
            return True
        if lowered in _BOOL_FALSE:
            return False
    _warn(f"ignoring '{source}': key '{key}' expects true/false, "
          f"got {value!r}.")
    return None


def _coerce_int(key: str, value: Any, source: str,
                low: int, high: int) -> Optional[int]:
    number: Optional[int] = None
    if isinstance(value, bool):
        number = None
    elif isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    elif isinstance(value, str):
        try:
            number = int(value.strip())
        except ValueError:
            number = None
    if number is None or not (low <= number <= high):
        _warn(f"ignoring '{source}': key '{key}' expects an integer "
              f"between {low} and {high}, got {value!r}.")
        return None
    return number


def _coerce_choice(key: str, value: Any, source: str,
                   choices: tuple[str, ...]) -> Optional[str]:
    if isinstance(value, str) and value.strip() in choices:
        return value.strip()
    _warn(f"ignoring '{source}': key '{key}' expects one of "
          f"{list(choices)}, got {value!r}.")
    return None


def coerce_config(raw: dict[str, Any], source: str = "<config>") -> dict[str, Any]:
    """
    Validate raw parsed values and return only known, well-typed keys.

    Unknown keys produce an English warning and are ignored.
    Invalid values produce an English warning and fall back to the
    built-in default (never raise: a bad config must not break startup).
    """
    typed: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in DEFAULTS:
            _warn(f"ignoring '{source}': unknown key '{key}'.")
            continue
        if key in ("data_path",):
            if isinstance(value, str) and value.strip():
                typed[key] = value.strip()
            else:
                _warn(f"ignoring '{source}': key '{key}' expects a "
                      f"non-empty path, got {value!r}.")
        elif key in ("latency_ns", "use_md", "normalize", "mode_3d",
                     "show_level", "surface_visible", "mask_gaps",
                     "show_wireframe", "show_errors", "show_projections",
                     "dolly", "unify", "show_pp", "show_tg"):
            coerced = _coerce_bool(key, value, source)
            if coerced is not None:
                typed[key] = coerced
        elif key == "level_value":
            coerced_int = _coerce_int(key, value, source, 0, 100)
            if coerced_int is not None:
                typed[key] = coerced_int
        elif key == "subdiv_level":
            coerced_int = _coerce_int(key, value, source, 0, 4)
            if coerced_int is not None:
                typed[key] = coerced_int
        elif key == "surface_style":
            coerced = _coerce_choice(key, value, source, SURFACE_STYLES)
            if coerced is not None:
                typed[key] = coerced
        elif key == "interp_method":
            coerced = _coerce_choice(key, value, source, INTERP_METHODS)
            if coerced is not None:
                typed[key] = coerced
        elif key == "z_label_mode":
            coerced = _coerce_choice(key, value, source, Z_LABEL_MODES)
            if coerced is not None:
                typed[key] = coerced
    return typed


def load_config_file(path: Path) -> dict[str, Any]:
    """
    Load and validate a YAML startup config file.

    Returns the typed dict (possibly empty). Malformed files print
    an English warning and yield {} so startup keeps working.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        _warn(f"cannot read config '{path}': {exc}. Using defaults.")
        return {}
    try:
        raw = parse_simple_yaml(text, source=str(path))
    except ValueError as exc:
        _warn(f"invalid config: {exc}. Using defaults.")
        return {}
    return coerce_config(raw, source=str(path))


def resolve_config_path(explicit: Optional[Path],
                        no_config: bool,
                        base_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Decide which config file (if any) applies.

    Priority: ``--no-config`` (none) > ``--config <file>`` (that file)
    > auto-found ``llamagraph.config.yml`` next to the script.
    """
    if no_config:
        return None
    if explicit is not None:
        return explicit
    return find_auto_config(base_dir)


def effective_value(cli_value: Any, config: dict[str, Any],
                    key: str) -> Any:
    """
    Merge one setting: an explicitly given CLI value (not None) wins
    over the config value, which wins over the built-in default.
    """
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return DEFAULTS[key]
