"""Config editor GUI for creating/editing experiment configs."""

from typing import Optional
from omegaconf import DictConfig, OmegaConf

from aintelope.gui.gui import (
    Frame,
    Label,
    Separator,
    ScrollableFrame,
    SelectorBar,
    ActionBar,
    StatusBar,
    Notebook,
    Button,
    Entry,
    StringVar,
    launch_window,
    create_widget,
    get_range_display,
    ask_yes_no,
    W,
    X,
    LEFT,
    BOTH,
    HORIZONTAL,
)
from aintelope.gui.ui_schema_manager import load_ui_schema, get_field_spec
from aintelope.config.config_utils import (
    list_loadable_configs,
    load_experiment_config,
    save_experiment_config,
)


class ConfigGUI:
    def __init__(self, root, default_cfg: DictConfig):
        self.root = root
        self.root.title("Config Editor")
        self.root.geometry("1000x700")

        self.default_cfg = default_cfg
        self.default_hparams = OmegaConf.to_container(default_cfg.hparams, resolve=True)
        self.ui_schema = load_ui_schema()
        self.result = None
        self.tabs = []

        self._create_ui()
        self._load_config("example_config.yaml")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_ui(self):
        self.selector = SelectorBar(
            self.root,
            label="Config",
            values=list_loadable_configs(),
            on_select=self._load_config,
        )
        self.selector.pack(fill=X)

        tab_controls = Frame(self.root, padding=(10, 5))
        tab_controls.pack(fill=X)
        Button(tab_controls, text="Add Experiment", command=self._add_tab).pack(
            side=LEFT, padx=5
        )
        Button(tab_controls, text="Delete Experiment", command=self._delete_tab).pack(
            side=LEFT, padx=5
        )

        self.notebook = Notebook(self.root)
        self.notebook.pack(fill=BOTH, expand=True, padx=10, pady=5)

        self.actions = ActionBar(
            self.root,
            inputs=[("Save As", "my_config.yaml")],
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

    # =========================================================================
    # Tab management
    # =========================================================================

    def _create_tab(self, block_name, overrides=None):
        """Create a tab for one experiment block."""
        if overrides is None:
            overrides = {}

        merged = OmegaConf.to_container(
            OmegaConf.merge(self.default_cfg.hparams, overrides), resolve=True
        )

        tab = {
            "diff": dict(overrides),
            "current": merged,
            "widgets": {},
        }

        frame = ScrollableFrame(self.notebook, padding=5)
        tab["frame"] = frame

        # Block name field
        name_frame = Frame(frame.content)
        name_frame.grid(row=0, column=0, columnspan=3, sticky=W, pady=(5, 10))
        Label(name_frame, text="Experiment name:", width=25, anchor=W).pack(
            side=LEFT, padx=(0, 10)
        )
        name_var = StringVar(value=block_name)
        Entry(name_frame, textvariable=name_var, width=30).pack(side=LEFT, padx=5)
        name_var.trace_add("write", lambda *_: self._update_tab_label(tab))
        tab["name_var"] = name_var

        Separator(frame.content, orient=HORIZONTAL).grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(0, 5)
        )

        self._build_param_tree(frame.content, merged, tab, row_offset=2)

        self.notebook.add(frame, text=block_name)
        self.tabs.append(tab)
        self.notebook.select(frame)

    def _update_tab_label(self, tab):
        """Sync notebook tab text with the name field."""
        self.notebook.tab(tab["frame"], text=tab["name_var"].get())

    def _add_tab(self):
        """Add tab copying current tab's overrides."""
        name = f"experiment_{len(self.tabs)}"
        overrides = {}
        if self.tabs:
            current_idx = self.notebook.index(self.notebook.select())
            overrides = dict(self.tabs[current_idx]["diff"])
        self._create_tab(name, overrides)
        self.status.set(f"Added: {name}")

    def _delete_tab(self):
        """Delete current tab with confirmation. Minimum 1 tab."""
        if len(self.tabs) <= 1:
            self.status.set("Cannot delete the last experiment.")
            return

        current_idx = self.notebook.index(self.notebook.select())
        tab = self.tabs[current_idx]
        name = tab["name_var"].get()

        if not ask_yes_no("Delete Experiment", f"Delete '{name}'?"):
            return

        self.notebook.forget(tab["frame"])
        self.tabs.pop(current_idx)
        self.status.set(f"Deleted: {name}")

    # =========================================================================
    # Parameter tree
    # =========================================================================

    def _build_param_tree(self, parent, config, tab, path="", level=0, row_offset=0):
        """Recursively build parameter tree for one tab."""
        row = row_offset

        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, dict):
                frame = Frame(parent)
                frame.grid(
                    row=row,
                    column=0,
                    columnspan=3,
                    sticky=W,
                    pady=(10 if level == 0 else 5, 2),
                )
                Label(
                    frame,
                    text=key.replace("_", " ").title(),
                    font=("Arial", 10 if level == 0 else 9, "bold"),
                ).pack(side=LEFT, padx=(level * 20, 0))

                if level == 0:
                    Separator(parent, orient=HORIZONTAL).grid(
                        row=row + 1,
                        column=0,
                        columnspan=3,
                        sticky="ew",
                        pady=(0, 5),
                    )
                    row += 2
                else:
                    row += 1

                nested = Frame(parent)
                nested.grid(row=row, column=0, columnspan=3, sticky=W)
                self._build_param_tree(nested, value, tab, current_path, level + 1)
                row += 1
            else:
                param_frame = Frame(parent)
                param_frame.grid(row=row, column=0, columnspan=3, sticky=W, pady=2)

                Label(param_frame, text=f"{key}:", width=25, anchor=W).pack(
                    side=LEFT, padx=(level * 20, 10)
                )

                schema_path = f"hparams.{current_path}"
                spec = get_field_spec(self.ui_schema, schema_path)

                widget, refresher = create_widget(
                    param_frame,
                    key,
                    value,
                    spec,
                    lambda v, p=current_path, t=tab: self._update_value(t, p, v),
                )
                widget.pack(side=LEFT, padx=5)

                range_display = get_range_display(spec)
                if range_display:
                    Label(param_frame, text=range_display, foreground="gray").pack(
                        side=LEFT, padx=10
                    )

                tab["widgets"][current_path] = (widget, refresher)
                row += 1

    # =========================================================================
    # Value tracking
    # =========================================================================

    def _update_value(self, tab, path, value):
        """Update a value in a specific tab's config."""
        keys = path.split(".")

        spec = get_field_spec(self.ui_schema, f"hparams.{path}")
        if spec:
            vtype = spec[1]
            if vtype == "bool":
                value = bool(value)
            elif vtype == "int":
                value = int(float(value))
            elif vtype == "float":
                value = float(value)

        # Update display config
        current = tab["current"]
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value

        # Track diff against defaults
        default = self.default_hparams
        for key in keys:
            default = default[key]

        if value != default:
            self._set_nested(tab["diff"], keys, value)

        self.status.set(f"Modified: {path}")

    def _set_nested(self, d, keys, value):
        """Set a value in a nested dict, creating intermediates."""
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    # =========================================================================
    # Load / Save / Run
    # =========================================================================

    def _load_config(self, config_name):
        """Load config file, creating one tab per block."""
        blocks = load_experiment_config(config_name)

        for tab in self.tabs:
            self.notebook.forget(tab["frame"])
        self.tabs = []

        for block_name, overrides in blocks.items():
            self._create_tab(block_name, overrides)

        self.status.set(f"Loaded: {config_name}")

    def _collect_blocks(self):
        """Gather all tabs into {name: overrides} dict."""
        return {tab["name_var"].get(): tab["diff"] for tab in self.tabs}

    def _save_config(self):
        """Save all tabs as a multi-block config."""
        filename = self.actions.get_input("Save As")
        save_experiment_config(self._collect_blocks(), filename)
        self.status.set(f"Saved: {filename}")
        self.selector.set_values(list_loadable_configs())

    def _run(self):
        """Return blocks dict and close."""
        self.result = OmegaConf.create(self._collect_blocks())
        self.root.quit()

    def _on_close(self):
        self.result = None
        self.root.quit()


def run_gui(default_cfg: DictConfig) -> Optional[DictConfig]:
    """Launch config editor GUI. Returns multi-block config or None."""
    return launch_window(ConfigGUI, default_cfg)
