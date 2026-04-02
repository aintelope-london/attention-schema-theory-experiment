# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from ast import literal_eval
from pathlib import Path
import zipfile
import os
import sys
import time

from omegaconf import DictConfig, OmegaConf


CONFIG_DIR = Path("aintelope") / "config"
DEFAULT_CONFIG = "default_config.yaml"
MODEL_LIBRARY = "model_library.yaml"
PROTECTED_CONFIGS = {DEFAULT_CONFIG, "example_config.yaml", MODEL_LIBRARY}


def init_config(main_config):
    """Load defaults, merge library, inject architecture, resolve timestamps."""
    cfg = OmegaConf.load(CONFIG_DIR / DEFAULT_CONFIG)
    cfg = OmegaConf.merge(cfg, list(main_config.values())[0])
    library = OmegaConf.load(CONFIG_DIR / MODEL_LIBRARY)
    cfg = OmegaConf.merge(cfg, {"models": library.models})
    for agent_key in cfg.agent_params.agents:
        model_name = cfg.agent_params.agents[agent_key].model
        OmegaConf.update(
            cfg,
            f"agent_params.agents.{agent_key}.architecture",
            library.architectures[model_name],
            force_add=True,
        )
    OmegaConf.update(cfg, "run.outputs_dir", str(cfg.run.outputs_dir))
    return cfg


def prepare_experiment_cfg(cfg, overrides, experiment_name, trial_seed):
    """Merge overrides and derive runtime config values."""
    merged = OmegaConf.merge(cfg, overrides)
    OmegaConf.update(merged, "experiment_name", experiment_name, force_add=True)
    OmegaConf.update(merged.run, "seed", trial_seed, force_add=True)
    OmegaConf.update(
        merged.env_params,
        "amount_agents",
        sum(1 for _ in merged.agent_params.agents),
    )
    return merged


def to_picklable(cfg):
    """Convert DictConfig to plain dict for multiprocessing."""
    return OmegaConf.to_container(cfg, resolve=True)


def from_picklable(cfg_dict):
    """Reconstruct DictConfig from plain dict."""
    return OmegaConf.create(cfg_dict)


def list_loadable_configs():
    """List configs available for loading (excludes protected configs)."""
    return sorted(
        f.name for f in CONFIG_DIR.glob("*.yaml") if f.name not in PROTECTED_CONFIGS
    )


def load_experiment_config(filename):
    """Load a multi-block experiment config. Returns {block_name: overrides}."""
    return OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / filename), resolve=False)


def save_experiment_config(blocks, filename):
    """Save {block_name: overrides} to config file."""
    if filename in PROTECTED_CONFIGS:
        raise ValueError(f"Cannot overwrite protected config: {filename}")
    if not filename.endswith(".yaml"):
        filename += ".yaml"
    OmegaConf.save(OmegaConf.create(blocks), CONFIG_DIR / filename)


def set_console_title(title):
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW(title)
        else:
            term = os.getenv("TERM")
            if term[:5] == "xterm" or term == "vt100":
                print("\x1B]0;{}\x07".format(title))
            elif os.name == "posix":
                import platform
                system = platform.system()
                if system == "Linux":
                    sys.stdout.write("\x1B]2;{}\x07".format(title))
                elif system == "Darwin":
                    sys.stdout.write("\x1B]0;{}\x07".format(title))
    except Exception:
        pass


def get_project_path(path_from_root: str) -> Path:
    return Path(__file__).parents[2] / path_from_root


def custom_now(format: str = "%Y%m%d%H%M%S") -> str:
    return time.strftime(format)


def create_range(start, exclusive_end):
    return list(range(start, exclusive_end))


def muldiv(entry, multiplier, divisor):
    if entry is None:
        return None
    elif hasattr(entry, "__iter__"):
        return [int(x * multiplier / divisor) for x in entry]
    else:
        return int(entry * multiplier / divisor)


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("custom_now", custom_now, replace=True)
    OmegaConf.register_new_resolver("now", custom_now, replace=True)
    OmegaConf.register_new_resolver("abs_path", get_project_path, replace=True)
    OmegaConf.register_new_resolver("muldiv", muldiv, replace=True)
    OmegaConf.register_new_resolver("range", create_range, replace=True)


register_resolvers()