#!/usr/bin/env python3
"""
Auto-generate .env.example from app/utils/config.py Settings model.

Reads every field in ``Settings.model_fields`` and writes one block per
field to ``.env.example``:

    # <description or field name>
    FIELD_NAME=<default>

Goals:
  - Keep .env.example in sync with the typed Settings surface.
  - Provide >= 80% coverage of settings fields (up from the hand-written
    baseline of ~79).
  - Non-destructive: running the script overwrites .env.example but does
    not touch .env or .env.production.

Usage:
    python scripts/generate_env_example.py            # regenerate in place
    python scripts/generate_env_example.py --check    # exit 1 if stale
    python scripts/generate_env_example.py --stdout   # print to stdout

Environment:
    LOCOPILOT_SKIP_PATH_CHECKS=1 is set automatically so the generator
    works on fresh clones where the .pt weights are not yet downloaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


# Resolve repo root and ensure it's importable before touching the Settings
# model, so the script runs from any cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Bypass model-path existence checks inside the validator — we only need the
# schema here, not working inference.
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")

from app.utils.config import Settings  # noqa: E402


# Fields whose defaults are computed at import time from the host machine
# (e.g. tempfile.gettempdir(), __file__) and therefore leak user-specific
# absolute paths into the generated file. Override them with portable
# placeholders so the checked-in .env.example is host-agnostic.
_PORTABLE_OVERRIDES: dict[str, str] = {
    "upload_dir": "/tmp/locopilot_uploads",
    "output_dir": "locopilot_evidence",
}


HEADER = """# =============================================================================
# Locopilot Monitoring System - Environment Configuration Template
# =============================================================================
# Auto-generated from app/utils/config.py Settings.model_fields.
# Do not edit manually. Regenerate with:
#
#     python scripts/generate_env_example.py
#
# Copy this file to .env and override only the settings you need to change.
# NEVER commit .env files containing real credentials.
# =============================================================================
"""


def _format_default(value: Any) -> str:
    """
    Render a field default as an env var value.

    - None -> empty string (env var unset).
    - bool -> 1 / 0 (matches the ``bool(int(os.getenv(...)))`` pattern
      used across config.py).
    - list / dict -> JSON so the parser can round-trip it.
    - everything else -> str().
    """
    if value is None:
        return ""
    # PydanticUndefined sentinel shows up when no default is given.
    if value.__class__.__name__ == "PydanticUndefinedType":
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(list(value) if isinstance(value, tuple) else value)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _iter_blocks(settings_cls: type[Settings]) -> Iterable[str]:
    """Yield one textual block per field in ``settings_cls.model_fields``."""
    for name, field in settings_cls.model_fields.items():
        description = (field.description or "").strip()
        if not description:
            # Fall back to a humanised field name so generated lines always
            # carry a comment.
            description = name.replace("_", " ").strip()
        if name in _PORTABLE_OVERRIDES:
            default = _PORTABLE_OVERRIDES[name]
        else:
            default = _format_default(field.default)
        env_name = name.upper()
        yield f"# {description}\n{env_name}={default}\n"


def generate(settings_cls: type[Settings] = Settings) -> str:
    """Produce the full .env.example contents as a single string."""
    body = "\n".join(_iter_blocks(settings_cls))
    return f"{HEADER}\n{body}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-generate .env.example from Settings model fields."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 without writing if .env.example is out of date.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write generated content to stdout instead of .env.example.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / ".env.example"),
        help="Path to write (default: <repo>/.env.example).",
    )
    args = parser.parse_args()

    content = generate()
    field_count = len(Settings.model_fields)

    if args.stdout:
        sys.stdout.write(content)
        print(
            f"\n# Generated {field_count} fields",
            file=sys.stderr,
        )
        return 0

    out_path = Path(args.output)

    if args.check:
        existing = out_path.read_text() if out_path.exists() else ""
        if existing != content:
            print(
                f"ERROR: {out_path} is out of date. "
                f"Run `python scripts/generate_env_example.py` to regenerate.",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {out_path} is up to date ({field_count} fields).")
        return 0

    out_path.write_text(content)
    print(f"Wrote {out_path} ({field_count} fields).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
