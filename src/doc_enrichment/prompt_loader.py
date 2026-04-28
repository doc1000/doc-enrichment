from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from doc_enrichment.errors import PromptLoadError

_DEFAULT_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str, version: str = "v1", prompts_dir: Path | None = None) -> dict[str, Any]:
    """Load and return the full YAML content for a named prompt at a given version.

    Args:
        name: Prompt name, e.g. ``"document"``, ``"parent"``, ``"contrast"``.
        version: Version string, e.g. ``"v1"``.
        prompts_dir: Override the directory to search. Defaults to the
            package-bundled ``prompts/`` directory.

    Returns:
        Parsed YAML as a dict.

    Raises:
        PromptLoadError: If the file is not found or cannot be parsed.
    """
    directory = prompts_dir or _DEFAULT_PROMPTS_DIR
    path = directory / f"{name}_{version}.yaml"

    if not path.exists():
        raise PromptLoadError(
            f"Prompt file not found: {path}. "
            f"Available prompts in {directory}: "
            f"{[p.name for p in directory.glob('*.yaml')]}"
        )

    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise PromptLoadError(f"Failed to parse prompt file {path}: {exc}") from exc

    return data


def get_template(
    name: str,
    version: str = "v1",
    sub_key: str | None = None,
    prompts_dir: Path | None = None,
) -> str:
    """Return a single template string from a prompt YAML file.

    For prompts with a single template (e.g. ``parent_v1.yaml``), omit
    ``sub_key``.  For prompts with multiple templates (e.g. ``document_v1.yaml``
    which has ``extract`` and ``reduce`` sections), pass ``sub_key`` to select
    the correct one.

    Args:
        name: Prompt name.
        version: Version string.
        sub_key: Optional section key for multi-template prompt files.
        prompts_dir: Override directory.

    Returns:
        Raw template string suitable for use with
        ``ChatPromptTemplate.from_template``.

    Raises:
        PromptLoadError: If the key or template field is missing.
    """
    data = load_prompt(name, version, prompts_dir=prompts_dir)

    try:
        if sub_key is not None:
            return data[sub_key]["template"]
        return data["template"]
    except (KeyError, TypeError) as exc:
        key_path = f"{sub_key}.template" if sub_key else "template"
        raise PromptLoadError(
            f"Key '{key_path}' not found in prompt '{name}_{version}.yaml'."
        ) from exc
