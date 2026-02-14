"""Config editor GUI for creating/editing experiment configs."""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from omegaconf import DictConfig, OmegaConf

from aintelope.gui.gui import (
    Frame,
    Label,
    Separator,
    ScrollableFrame,
    SelectorBar,
    ActionBar,
    StatusBar,
    launch_window,
    create_widget,
    get_range_display,
    W, X, LEFT, HORIZONTAL,
)
from aintelope.gui.ui_schema_manager import load_ui_schema, get_field_spec

CONFIG_DIR = Path("aintelope") / "config"


class ConfigGUI:
    def __init__(self, root, default_cfg: DictConfig):
        self.root = root
        self.root.title("Config Editor")
        self.root.geometry("1000x700")

        self.default_cfg = default_cfg
        self.ui_schema = load_ui_schema()

        self.current_config = OmegaConf.to_container(default_cfg, resolve=True)
        self.diff_config = {}
        self.result = None
        self.widgets = {}

        self._create_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _get_default_value(self, keys: list) -> Any:
        """Get value from default config by key path."""
        current = self.default_cfg
        for key in keys:
            current = current[key]
        return current

    def _set_nested_diff(self, keys: list, value: Any):
        """Set value in diff_config, creating nested dicts as needed."""
        current = self.diff_config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def _create_ui(self):
        self.selector = SelectorBar(
            self.root,
            label="Config",
            values=self._list_configs(),
            on_select=self._load_config,
        )
        self.selector.pack(fill=X)

        self.editor = ScrollableFrame(self.root, text="Configuration", padding=10)
        self.editor.pack(fill="both", expand=True, padx=10, pady=5)

        self._build_param_tree(self.editor.content, self.default_cfg)

        self.actions = ActionBar(
            self.root,
            inputs=[("Save As", self._generate_filename())],
            buttons=[
                ("Save Config", self._save_config),
                ("Run", self._run),
                ("Cancel", self._on_close),
            ],
        )
        self.actions.pack(fill=X)

        self.status = StatusBar(
            self.root, "Ready. Edit parameters or load existing config."
        )
        self.status.pack(fill=X, padx=10, pady=(0, 10))

    def _build_param_tree(self, parent, config, path="", level=0):
        """Recursively build parameter tree from config."""
        row = 0

        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, (dict, DictConfig)):
                frame = Frame(parent)
                frame.grid(
                    row=row,
                    column=0,
                    columnspan=3,
                    sticky=W,
                    pady=(10 if level == 0 else 5, 2),
                )

                label = Label(
                    frame,
                    text=key.replace("_", " ").title(),
                    font=("Arial", 10 if level == 0 else 9, "bold"),
                )
                label.pack(side=LEFT, padx=(level * 20, 0))

                if level == 0:
                    sep = Separator(parent, orient=HORIZONTAL)
                    sep.grid(
                        row=row + 1, column=0, columnspan=3, sticky="ew", pady=(0, 5)
                    )
                    row += 2
                else:
                    row += 1

                nested_frame = Frame(parent)
                nested_frame.grid(row=row, column=0, columnspan=3, sticky=W)
                self._build_param_tree(nested_frame, value, current_path, level + 1)
                row += 1
            else:
                param_frame = Frame(parent)
                param_frame.grid(row=row, column=0, columnspan=3, sticky=W, pady=2)

                label = Label(param_frame, text=f"{key}:", width=25, anchor=W)
                label.pack(side=LEFT, padx=(level * 20, 10))

                spec = get_field_spec(self.ui_schema, current_path)

                widget, refresher = create_widget(
                    param_frame,
                    key,
                    value,
                    spec,
                    lambda v, p=current_path: self._update_value(p, v),
                )
                widget.pack(side=LEFT, padx=5)

                range_display = get_range_display(spec)
                if range_display:
                    info_label = Label(
                        param_frame, text=range_display, foreground="gray"
                    )
                    info_label.pack(side=LEFT, padx=10)

                self.widgets[current_path] = (widget, refresher)
                row += 1

    def _update_value(self, path: str, value: Any):
        """Update config when a value changes."""
        keys = path.split(".")

        spec = get_field_spec(self.ui_schema, path)
        if spec:
            value_type = spec[1]
            if value_type == "bool":
                value = bool(value)
            elif value_type == "int":
                value = int(float(value))
            elif value_type == "float":
                value = float(value)

        current = self.current_config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value

        default_value = self._get_default_value(keys)
        if value != default_value:
            self._set_nested_diff(keys, value)

        self.status.set(f"Modified: {path}")

    def _generate_filename(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"config_{timestamp}.yaml"

    def _list_configs(self) -> list:
        return sorted([f.name for f in CONFIG_DIR.glob("config_*.yaml")])

    def _load_config(self, config_name: str):
        """Load config file (a diff), applying it over default."""
        config_path = CONFIG_DIR / config_name

        self.diff_config = OmegaConf.to_container(OmegaConf.load(config_path))

        merged = OmegaConf.merge(self.default_cfg, self.diff_config)
        self.current_config = OmegaConf.to_container(merged, resolve=True)

        self._refresh_widgets()
        self.status.set(f"Loaded: {config_name}")

    def _save_config(self):
        """Save diff config."""
        output_name = self.actions.get_input("Save As")
        if not output_name.endswith(".yaml"):
            output_name += ".yaml"

        output_path = CONFIG_DIR / output_name
        OmegaConf.save(OmegaConf.create(self.diff_config), output_path)

        self.status.set(f"Saved: {output_name}")

    def _run(self):
        """Return diff config as orchestrator config and close GUI."""
        hparams_diff = self.diff_config.get("hparams", {})
        self.result = OmegaConf.create({"gui_run": hparams_diff})
        self.root.quit()

    def _on_close(self):
        """Handle window close (cancel)."""
        self.result = None
        self.root.quit()

    def _refresh_widgets(self):
        """Update widget values based on current config."""

        def update_widgets(config, path=""):
            for key, value in config.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, dict):
                    update_widgets(value, current_path)
                elif current_path in self.widgets:
                    _, refresher = self.widgets[current_path]
                    refresher(value)

        update_widgets(self.current_config)


def run_gui(default_cfg: DictConfig) -> Optional[DictConfig]:
    """Launch config editor GUI. Returns diff config or None if cancelled."""
    return launch_window(ConfigGUI, default_cfg)