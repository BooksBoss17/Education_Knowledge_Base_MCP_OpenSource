from __future__ import annotations

import os
import tomllib
from pathlib import Path

CONFIG_ENV = "BEMARKDOWN_CONFIG"
OUTPUT_ROOT_ENV = "BEMARKDOWN_OUTPUT_ROOT"
CONFIG_NAME = "bemarkdown.toml"


class OutputRootConfigurationError(RuntimeError):
    pass


def resolve_output_root(
    explicit: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> Path:
    """Resolve explicit > environment > workspace configuration > local fallback."""

    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    environment = os.environ.get(OUTPUT_ROOT_ENV)
    if environment:
        return Path(environment).expanduser().resolve()
    config, payload = load_workspace_config(config_path)
    if config is not None:
        try:
            configured = payload["workspace"]["output_root"]
        except (KeyError, TypeError) as exc:
            raise OutputRootConfigurationError(
                f"Invalid BeMarkdown workspace configuration {config}: {exc}"
            ) from exc
        if not isinstance(configured, str) or not configured.strip():
            raise OutputRootConfigurationError(
                f"workspace.output_root must be a non-empty string in {config}"
            )
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = config.parent / candidate
        return candidate.resolve()
    return (Path.cwd() / "tmp" / "bemarkdown").resolve()


def load_workspace_config(
    explicit: str | Path | None = None,
) -> tuple[Path | None, dict]:
    """Load the discovered workspace configuration once for runtime resolvers."""

    config = _resolve_config_path(explicit)
    if config is None:
        return None, {}
    try:
        payload = tomllib.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise OutputRootConfigurationError(
            f"Invalid BeMarkdown workspace configuration {config}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise OutputRootConfigurationError(
            f"BeMarkdown workspace configuration root must be a table: {config}"
        )
    return config, payload


def _resolve_config_path(explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise OutputRootConfigurationError(
                f"BeMarkdown configuration file does not exist: {candidate}"
            )
        return candidate
    environment = os.environ.get(CONFIG_ENV)
    if environment:
        return _resolve_config_path(environment)
    for parent in (Path.cwd(), *Path.cwd().parents):
        candidate = parent / CONFIG_NAME
        if candidate.is_file():
            return candidate.resolve()
    editable_root = Path(__file__).resolve().parents[2]
    candidate = editable_root / CONFIG_NAME
    return candidate.resolve() if candidate.is_file() else None
