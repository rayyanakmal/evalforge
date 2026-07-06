"""Configuration loader for EvalForge.

Provides GateConfig model and load/save functions for evalforge.yaml.
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class SuiteConfig(BaseModel):
    """Per-suite configuration within a GateConfig."""
    path: str
    allowed_regression_pct: float = 5.0


class ProviderConfig(BaseModel):
    """LLM provider configuration."""
    provider: str
    model: str


class GateConfig(BaseModel):
    """Top-level EvalForge configuration model.

    Loaded from evalforge.yaml (or evalforge.toml in future).
    Defaults match what `evalforge init` scaffolds.
    """
    baseline_dir: str = "evalforge-baselines/"
    suites: list[SuiteConfig] = Field(default_factory=list)
    judge: Optional[ProviderConfig] = None
    target: Optional[ProviderConfig] = None
    concurrency: int = 10


def load_config(path: Path) -> GateConfig:
    """Load a GateConfig from a YAML file.

    Args:
        path: Path to evalforge.yaml.

    Returns:
        Validated GateConfig.

    Raises:
        FileNotFoundError: If the config file does not exist, with a
            message suggesting `evalforge init`.
        ValueError: If the YAML is invalid or doesn't match the schema.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No config found at {path}. "
            "Run `evalforge init` to create one."
        )

    raw = path.read_text()
    if not raw.strip():
        raise ValueError(f"Config file at {path} is empty.")

    data = yaml.safe_load(raw)
    if data is None:
        raise ValueError(f"Config file at {path} contains no YAML data.")

    return GateConfig.model_validate(data)


def save_config(config: GateConfig, path: Path) -> None:
    """Write a GateConfig to a YAML file.

    Creates parent directories if they don't exist.

    Args:
        config: The GateConfig to save.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(exclude_none=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
