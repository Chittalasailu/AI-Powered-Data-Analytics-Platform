"""Shared config.yaml loading and environment-aware path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_environment(
    raw_config: dict[str, Any], env_override: Optional[str] = None
) -> tuple[str, str, str]:
    """Resolve which environment is active and its raw/processed data dirs.

    Local relative paths (under `paths.local`) are resolved against the
    project root so the pipeline behaves the same regardless of the caller's
    working directory. Databricks paths (`/mnt/...`, `dbfs:/...`) are used
    as-is, since they're already absolute.
    """
    environment = env_override or raw_config["environment"]
    if environment not in ("local", "databricks"):
        raise ValueError(f"Unknown environment '{environment}': expected 'local' or 'databricks'")

    paths = raw_config["paths"][environment]
    raw_data_dir = paths["raw_data_dir"]
    processed_data_dir = paths["processed_data_dir"]
    if environment == "local":
        raw_data_dir = str(ROOT_DIR / raw_data_dir)
        processed_data_dir = str(ROOT_DIR / processed_data_dir)

    return environment, raw_data_dir, processed_data_dir
