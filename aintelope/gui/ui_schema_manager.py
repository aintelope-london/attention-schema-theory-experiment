"""UI schema utilities for config parameter rendering hints."""

from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf, DictConfig


def load_ui_schema() -> DictConfig:
    """Load UI schema from ui_schema.yaml."""
    schema_path = Path(__file__).parent / "ui_schema.yaml"
    return OmegaConf.load(schema_path)


def get_field_spec(ui_schema: DictConfig, key_path: str) -> Optional[list]:
    """
    Get UI rendering spec for a parameter.

    Returns:
        List like [[min, max], "type"] or None if not in schema
    """
    keys = key_path.split(".")
    current = ui_schema

    for key in keys:
        if key not in current:
            return None
        current = current[key]

    result = OmegaConf.to_container(current, resolve=True)

    # Return None if not a leaf spec (i.e., if it's a dict/intermediate node)
    if isinstance(result, dict):
        return None

    return result
