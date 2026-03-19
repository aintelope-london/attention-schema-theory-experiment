# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/aintelope-london/attention-schema-theory-experiment

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
    for agent_key in (k for k in cfg.agent_params if k.startswith("agent_")):
        model_name = cfg.agent_params[agent_key].model
        OmegaConf.update(
            cfg,
            f"agent_params.{agent_key}.architecture",
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
        sum(1 for k in merged.agent_params if k.startswith("agent_")),
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


def minus_3(entry):
    if entry is None:
        return None
    elif hasattr(entry, "__iter__"):
        return [x - 3 for x in entry]
    else:
        return entry - 3


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
    OmegaConf.register_new_resolver("minus_3", minus_3, replace=True)
    OmegaConf.register_new_resolver("muldiv", muldiv, replace=True)
    OmegaConf.register_new_resolver("range", create_range, replace=True)
    OmegaConf.register_new_resolver(
        "count_prefix",
        lambda parent_key, prefix, *, _root_: sum(
            1 for k in OmegaConf.select(_root_, parent_key) if k.startswith(prefix)
        ),
        replace=True,
    )


def get_score_dimensions(cfg: DictConfig):
    scores = cfg.env_params.scores
    dimensions = set()
    for event_name, score_dims_dict in scores.items():
        score_dims_dict = literal_eval(score_dims_dict)
        for dimension, value in score_dims_dict.items():
            if value != 0:
                dimensions.add(dimension)
    return sorted(dimensions)


def set_priorities():
    try:
        import psutil

        if hasattr(psutil, "Process"):
            p = psutil.Process(os.getpid())
            p.nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 20)
            p.ionice(0 if os.name == "nt" else psutil.IOPRIO_CLASS_IDLE)
    except Exception:
        print("run pip install psutil")

    if os.name == "nt":
        try:
            import win32process

            win32process.SetThreadPriority(-2, -15)
            win32process.SetThreadPriorityBoost(-2, False)
            win32process.SetProcessPriorityBoost(-1, False)
        except Exception:
            print("run pip install pywin32")


def set_memory_limits():
    if os.name == "nt":
        from aintelope.config.windows_jobobject import set_mem_commit_limit

        try:
            set_mem_commit_limit(os.getpid(), 40 * 1024**3, 5 * 1024**3)
        except Exception:
            print("run pip install psutil")
    else:
        from aintelope.config.linux_rlimit import set_mem_limits

        set_mem_limits(40 * 1024**3, 400 * 1024**3)


def select_gpu(gpu_index=None):
    if gpu_index is None:
        gpu_index = os.environ.get("AINTELOPE_GPU")
    if gpu_index is not None:
        import torch

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print(
                "No CUDA GPU available, ignoring assigned GPU index, will be using CPU as a CUDA device"
            )
            return
        gpu_index = int(gpu_index)
        torch.cuda.set_device(gpu_index)
        print(f"Using CUDA GPU {gpu_index} : {torch.cuda.get_device_name(gpu_index)}")
    else:
        rotate_active_gpu_selection()


def rotate_active_gpu_selection():
    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("No CUDA GPU available, will be using CPU as a CUDA device")
        return
    elif gpu_count == 1:
        gpu_counter = torch.cuda.current_device()
        print("Using the only available CUDA GPU")
    else:
        import sqlite3

        conn = sqlite3.connect("gpu_counter2.db")
        cursor = conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS gpu_counter_table
            (dummy INTEGER UNIQUE, gpu_counter INTEGER);
            INSERT OR IGNORE INTO gpu_counter_table (dummy, gpu_counter) VALUES (0, 0);
        """
        )
        conn.commit()
        cursor.execute("BEGIN EXCLUSIVE TRANSACTION")
        cursor.execute("UPDATE gpu_counter_table SET gpu_counter = gpu_counter + 1")
        gpu_counter = cursor.execute(
            "SELECT gpu_counter FROM gpu_counter_table"
        ).fetchone()[0]
        conn.commit()
        conn.close()
        gpu_counter = gpu_counter % gpu_count
        torch.cuda.set_device(gpu_counter)
    print(f"Using CUDA GPU {gpu_counter} : {torch.cuda.get_device_name(gpu_counter)}")


def archive_code(cfg):
    """Archives the current version of the program to log folder."""
    code_directory_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), ".."
    )
    archive_code_in_dir(
        code_directory_path,
        os.path.join(
            os.path.normpath(cfg.run.outputs_dir), "aintelope_code_archive.zip"
        ),
    )
    archive_code_in_dir(
        os.path.join(code_directory_path, "..", "ai_safety_gridworlds"),
        os.path.join(
            os.path.normpath(cfg.run.outputs_dir), "gridworlds_code_archive.zip"
        ),
    )


def archive_code_in_dir(directory_path, zip_path):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as ziph:
        for root, dirs, files in os.walk(
            directory_path, topdown=True, followlinks=False
        ):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for file in files:
                if os.path.splitext(file)[1] in {".py", ".ipynb", ".yaml"}:
                    full_path = os.path.join(root, file)
                    ziph.write(
                        full_path,
                        os.path.relpath(full_path, os.path.join(directory_path, "..")),
                    )


class TeeStream:
    """Writes to multiple streams simultaneously. Used to tee stdout to a file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


register_resolvers()
