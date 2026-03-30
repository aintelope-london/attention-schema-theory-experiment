# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Main GUI window — Run config editor and Results viewer in a single window."""

import copy
import tkinter as tk
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from omegaconf import DictConfig, OmegaConf

from aintelope.analytics.plot_primitives import create_figure, save_figure, PLOT_TYPES
import aintelope.analytics.plots  # noqa: F401 — registers plot types
from aintelope.analytics.recording import (
    list_runs,
    list_blocks,
    read_events,
    save_frames,
)
from aintelope.config.config_utils import (
    list_loadable_configs,
    load_experiment_config,
    save_experiment_config,
)
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
    Combobox,
    StringVar,
    ValueSlider,
    PlaybackControl,
    create_widget,
    get_range_display,
    ask_yes_no,
    W,
    X,
    LEFT,
    BOTH,
    HORIZONTAL,
)
from aintelope.gui.renderer import (
    SavannaInterpreter,
    StateRenderer,
    Tileset,
    find_tileset,
    overlay,
)
from aintelope.gui.ui_schema_manager import load_ui_schema, get_field_spec

AGENT_PREFIX = "agent_"

# Results viewer display variables (special permission — DOCUMENTATION.md §6)
ROI_COLOR = (255, 255, 0)
ROI_ALPHA = 76


class MainWindow:
    def __init__(
        self,
        root,
        default_cfg: DictConfig,
        initial_tab: str = "run",
        outputs_dir: str = "outputs",
    ):
        self.root = root
        self.root.title("Aintelope")
        self.root.geometry("1200x800")
        self.result = None

        # Run tab state
        self.default_cfg = default_cfg
        self.default_values = OmegaConf.to_container(default_cfg, resolve=True)
        self.ui_schema = load_ui_schema()
        self.agent_template = copy.deepcopy(
            self.default_values["agent_params"]["agent_0"]
        )
        self.exp_tabs = []

        # Results tab state
        self.outputs_dir = outputs_dir
        self.df = None
        self.states = None
        self.interpreter = None
        self.renderer = StateRenderer(Tileset(find_tileset()))
        self._photo = None

        self._create_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(1, lambda: self._init_deferred(initial_tab))

    def _init_deferred(self, initial_tab):
        self._load_config("example_config.yaml")
        self._refresh_runs()
        if initial_tab == "results":
            self.main_notebook.select(self.results_frame)

    # =========================================================================
    # Top-level layout
    # =========================================================================

    def _create_ui(self):
        self.main_notebook = Notebook(self.root)
        self.main_notebook.pack(fill=BOTH, expand=True)

        self.run_frame = Frame(self.main_notebook)
        self.main_notebook.add(self.run_frame, text="Run")

        self.results_frame = Frame(self.main_notebook)
        self.main_notebook.add(self.results_frame, text="Results")

        self._create_run_tab()
        self._create_results_tab()

        self.status = StatusBar(self.root, "Ready.")
        self.status.pack(fill=X, padx=10, pady=(0, 10))

    # =========================================================================
    # Run tab
    # =========================================================================

    def _create_run_tab(self):
        self.config_selector = SelectorBar(
            self.run_frame,
            label="Config",
            values=list_loadable_configs(),
            on_select=self._load_config,
        )
        self.config_selector.pack(fill=X)

        tab_controls = Frame(self.run_frame, padding=(10, 5))
        tab_controls.pack(fill=X)
        Button(tab_controls, text="Add Experiment", command=self._add_exp_tab).pack(
            side=LEFT, padx=5
        )
        Button(
            tab_controls, text="Delete Experiment", command=self._delete_exp_tab
        ).pack(side=LEFT, padx=5)

        self.exp_notebook = Notebook(self.run_frame)
        self.exp_notebook.pack(fill=BOTH, expand=True, padx=10, pady=5)
        self.exp_notebook.bind("<<NotebookTabChanged>>", self._on_exp_tab_changed)

        self.run_actions = ActionBar(
            self.run_frame,
            inputs=[("Save As", "my_config.yaml")],
            buttons=[
                ("Save Config", self._save_config),
                ("Run", self._run),
                ("Cancel", self._on_close),
            ],
        )
        self.run_actions.pack(fill=X)

    # -- Experiment tab management --

    def _create_exp_tab(self, block_name, overrides=None, build=False):
        """Create an experiment block tab. Widgets built lazily on first selection."""
        if overrides is None:
            overrides = {}

        merged = OmegaConf.to_container(
            OmegaConf.merge(self.default_cfg, overrides), resolve=True
        )
        tab = {
            "diff": dict(overrides),
            "current": merged,
            "widgets": {},
            "built": False,
        }

        frame = ScrollableFrame(self.exp_notebook, padding=5)
        tab["frame"] = frame

        name_frame = Frame(frame.content)
        name_frame.grid(row=0, column=0, columnspan=3, sticky=W, pady=(5, 10))
        Label(name_frame, text="Experiment name:", width=25, anchor=W).pack(
            side=LEFT, padx=(0, 10)
        )
        name_var = StringVar(value=block_name)
        Entry(name_frame, textvariable=name_var, width=30).pack(side=LEFT, padx=5)
        name_var.trace_add("write", lambda *_: self._update_exp_tab_label(tab))
        tab["name_var"] = name_var

        Separator(frame.content, orient=HORIZONTAL).grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(0, 5)
        )

        self.exp_notebook.add(frame, text=block_name)
        self.exp_tabs.append(tab)
        self.exp_notebook.select(frame)

        if build:
            self._build_exp_tab(tab)

    def _build_exp_tab(self, tab):
        """Build full widget tree for a tab. Called once, lazily."""
        if tab["built"]:
            return
        self._build_param_tree(tab["frame"].content, tab["current"], tab, row_offset=2)
        tab["built"] = True

    def _on_exp_tab_changed(self, event=None):
        idx = self.exp_notebook.index(self.exp_notebook.select())
        if idx < len(self.exp_tabs):
            self._build_exp_tab(self.exp_tabs[idx])

    def _refresh_exp_tab(self, tab, block_name, overrides):
        """Update an existing tab's values without rebuilding widgets."""
        if overrides is None:
            overrides = {}
        merged = OmegaConf.to_container(
            OmegaConf.merge(self.default_cfg, overrides), resolve=True
        )
        tab["diff"] = dict(overrides)
        tab["current"] = merged
        tab["name_var"].set(block_name)
        if not tab["built"]:
            return
        for path, (widget, refresher) in tab["widgets"].items():
            value = self._get_nested(merged, path)
            if value is not None:
                refresher(value)
        if "agents_container" in tab:
            self._populate_agents(tab)

    def _update_exp_tab_label(self, tab):
        self.exp_notebook.tab(tab["frame"], text=tab["name_var"].get())

    def _add_exp_tab(self):
        name = f"experiment_{len(self.exp_tabs)}"
        overrides = {}
        if self.exp_tabs:
            current_idx = self.exp_notebook.index(self.exp_notebook.select())
            overrides = dict(self.exp_tabs[current_idx]["diff"])
        self._create_exp_tab(name, overrides, build=True)
        self.status.set(f"Added: {name}")

    def _delete_exp_tab(self):
        if len(self.exp_tabs) <= 1:
            self.status.set("Cannot delete the last experiment.")
            return
        current_idx = self.exp_notebook.index(self.exp_notebook.select())
        tab = self.exp_tabs[current_idx]
        name = tab["name_var"].get()
        if not ask_yes_no("Delete Experiment", f"Delete '{name}'?"):
            return
        self.exp_notebook.forget(tab["frame"])
        self.exp_tabs.pop(current_idx)
        self.status.set(f"Deleted: {name}")

    # -- Schema / param tree --

    def _schema_path(self, path):
        """Map agent_N paths to agent_0 for schema lookup."""
        parts = path.split(".")
        for i, part in enumerate(parts):
            if (
                part.startswith(AGENT_PREFIX)
                and part[len(AGENT_PREFIX) :].isdigit()
                and part != "agent_0"
            ):
                parts[i] = "agent_0"
        return ".".join(parts)

    def _build_param_tree(self, parent, config, tab, path="", level=0, row_offset=0):
        """Recursively build parameter tree for one tab."""
        row = row_offset

        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key

            # Agent entries are rendered by _build_agents_area
            if path == "agent_params" and key.startswith(AGENT_PREFIX):
                continue

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
                        row=row + 1, column=0, columnspan=3, sticky="ew", pady=(0, 5)
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

                spec = get_field_spec(self.ui_schema, self._schema_path(current_path))

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

        # After non-agent fields, render the agents area
        if path == "agent_params":
            self._build_agents_area(parent, tab, level, row)

        return row

    # -- Agent management --

    def _build_agents_area(self, parent, tab, level, row):
        container = Frame(parent)
        container.grid(row=row, column=0, columnspan=3, sticky=W)
        tab["agents_container"] = container
        tab["agents_level"] = level
        self._populate_agents(tab)

    def _populate_agents(self, tab):
        container = tab["agents_container"]
        level = tab["agents_level"]

        for child in container.winfo_children():
            child.destroy()
        for key in list(tab["widgets"]):
            if key.startswith(f"agent_params.{AGENT_PREFIX}"):
                del tab["widgets"][key]

        row = 0
        for agent_id, agent_cfg in self._get_agent_configs(tab).items():
            header = Frame(container)
            header.grid(row=row, column=0, columnspan=3, sticky=W, pady=(10, 2))
            Label(
                header,
                text=agent_id.replace("_", " ").title(),
                font=("Arial", 9, "bold"),
            ).pack(side=LEFT, padx=(level * 20, 10))
            if agent_id != "agent_0":
                Button(
                    header,
                    text="Remove",
                    command=lambda aid=agent_id: self._remove_agent(tab, aid),
                ).pack(side=LEFT)
            row += 1

            fields_frame = Frame(container)
            fields_frame.grid(row=row, column=0, columnspan=3, sticky=W)
            self._build_param_tree(
                fields_frame, agent_cfg, tab, f"agent_params.{agent_id}", level + 1
            )
            row += 1

        btn_frame = Frame(container)
        btn_frame.grid(row=row, column=0, columnspan=3, sticky=W, pady=(10, 5))
        Button(btn_frame, text="Add Agent", command=lambda: self._add_agent(tab)).pack(
            side=LEFT, padx=(level * 20, 0)
        )

    def _get_agent_configs(self, tab):
        return {
            k: v
            for k, v in sorted(tab["current"]["agent_params"].items())
            if k.startswith(AGENT_PREFIX)
        }

    def _add_agent(self, tab):
        agents = self._get_agent_configs(tab)
        new_id = f"agent_{len(agents)}"
        new_cfg = copy.deepcopy(self.agent_template)
        tab["current"]["agent_params"][new_id] = new_cfg
        self._set_nested(tab["diff"], ["agent_params", new_id], copy.deepcopy(new_cfg))
        self._populate_agents(tab)
        self.status.set(f"Added: {new_id}")

    def _remove_agent(self, tab, agent_id):
        agents = self._get_agent_configs(tab)
        remaining = [(k, v) for k, v in agents.items() if k != agent_id]
        for k in list(tab["current"]["agent_params"]):
            if k.startswith(AGENT_PREFIX):
                del tab["current"]["agent_params"][k]
        if "agent_params" in tab["diff"]:
            for k in list(tab["diff"]["agent_params"]):
                if k.startswith(AGENT_PREFIX):
                    del tab["diff"]["agent_params"][k]
        for i, (_, cfg) in enumerate(remaining):
            new_id = f"agent_{i}"
            tab["current"]["agent_params"][new_id] = cfg
            default_agent = self.default_values["agent_params"].get(new_id)
            if default_agent is None or cfg != default_agent:
                self._set_nested(
                    tab["diff"], ["agent_params", new_id], copy.deepcopy(cfg)
                )
        self._populate_agents(tab)
        self.status.set(f"Removed: {agent_id}")

    # -- Value tracking --

    def _get_nested(self, d, dotted_path):
        current = d
        for key in dotted_path.split("."):
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _set_nested(self, d, keys, value):
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    def _remove_nested(self, d, keys):
        stack = []
        current = d
        for key in keys[:-1]:
            if not isinstance(current, dict) or key not in current:
                return
            stack.append((current, key))
            current = current[key]
        current.pop(keys[-1], None)
        for parent, key in reversed(stack):
            if not parent[key]:
                del parent[key]

    def _update_value(self, tab, path, value):
        keys = path.split(".")
        spec = get_field_spec(self.ui_schema, self._schema_path(path))
        if spec:
            vtype = spec[1]
            if vtype == "bool":
                value = bool(value)
            elif vtype == "int":
                value = int(float(value))
            elif vtype == "float":
                value = float(value)
        current = tab["current"]
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value
        default = self.default_values
        for key in keys:
            if not isinstance(default, dict) or key not in default:
                default = None
                break
            default = default[key]
        if default is None or value != default:
            self._set_nested(tab["diff"], keys, value)
        else:
            self._remove_nested(tab["diff"], keys)
        self.status.set(f"Modified: {path}")

    # -- Load / Save / Run --

    def _load_config(self, config_name):
        blocks = load_experiment_config(config_name)
        block_items = list(blocks.items())
        if len(block_items) == len(self.exp_tabs):
            for tab, (block_name, overrides) in zip(self.exp_tabs, block_items):
                self._refresh_exp_tab(tab, block_name, overrides)
        else:
            for tab in self.exp_tabs:
                self.exp_notebook.forget(tab["frame"])
            self.exp_tabs = []
            for i, (block_name, overrides) in enumerate(block_items):
                self._create_exp_tab(block_name, overrides, build=(i == 0))
        self.status.set(f"Loaded: {config_name}")

    def _collect_blocks(self):
        return {tab["name_var"].get(): tab["diff"] for tab in self.exp_tabs}

    def _save_config(self):
        self.root.focus_set()
        self.root.update()
        filename = self.run_actions.get_input("Save As")
        save_experiment_config(self._collect_blocks(), filename)
        self.status.set(f"Saved: {filename}")
        self.config_selector.set_values(list_loadable_configs())

    def _run(self):
        self.root.focus_set()
        self.root.update()
        self.result = OmegaConf.create(self._collect_blocks())
        self.root.quit()

    # =========================================================================
    # Results tab
    # =========================================================================

    def _create_results_tab(self):
        selector_frame = Frame(self.results_frame, padding=10)
        selector_frame.pack(fill=X)

        Label(selector_frame, text="Run:").pack(side=LEFT, padx=5)
        self.run_var = StringVar()
        self.run_combo = Combobox(
            selector_frame, textvariable=self.run_var, width=40, state="readonly"
        )
        self.run_combo.pack(side=LEFT, padx=5)
        self.run_combo.bind("<<ComboboxSelected>>", self._on_run_selected)

        Label(selector_frame, text="Block:").pack(side=LEFT, padx=(20, 5))
        self.block_var = StringVar()
        self.block_combo = Combobox(
            selector_frame, textvariable=self.block_var, width=20, state="readonly"
        )
        self.block_combo.pack(side=LEFT, padx=5)
        self.block_combo.bind("<<ComboboxSelected>>", self._on_block_selected)

        Button(selector_frame, text="Load", command=self._load_selected).pack(
            side=LEFT, padx=(10, 5)
        )

        self.results_notebook = Notebook(self.results_frame)
        self.results_notebook.pack(fill=BOTH, expand=True, padx=10, pady=5)

        self._create_plots_tab()
        self._create_playback_tab()

        self.results_actions = ActionBar(
            self.results_frame,
            inputs=[("Save As", "plot.png")],
            buttons=[("Export Plot", self._export_plot)],
        )
        self.results_actions.pack(fill=X)

    def _refresh_runs(self):
        runs = list_runs(self.outputs_dir)
        self.run_combo["values"] = runs
        if runs:
            self.run_combo.current(0)
            self.run_var.set(runs[0])
            self._on_run_selected()
            self._on_block_selected()

    def _on_run_selected(self, event=None):
        run_name = self.run_var.get()
        blocks = list_blocks(str(Path(self.outputs_dir) / run_name))
        self.block_combo["values"] = blocks
        if blocks:
            self.block_combo.current(0)
            self.block_var.set(blocks[0])

    def _load_selected(self):
        self._on_run_selected()
        self._on_block_selected()

    def _on_block_selected(self, event=None):
        block_dir = Path(self.outputs_dir) / self.run_var.get() / self.block_var.get()

        self.df = read_events(str(block_dir / "events.csv"))
        self.states = read_events(str(block_dir / "states.csv"))
        self.interpreter = SavannaInterpreter()

        index_cols = {"Trial", "Episode", "Step", "Agent_id", "Experiment"}
        numeric_cols = self.df.select_dtypes(include="number").columns
        metric_cols = [c for c in numeric_cols if c not in index_cols]
        self.metric_combo["values"] = metric_cols

        agents = sorted(self.df["Agent_id"].unique())
        self.agent_combo["values"] = ["all"] + list(agents)
        self.agent_var.set("all")

        if "Reward" in metric_cols:
            self.metric_var.set("Reward")
        elif metric_cols:
            self.metric_var.set(metric_cols[0])

        episodes = sorted(self.df["Episode"].unique())
        self.episode_combo["values"] = [str(int(e)) for e in episodes]
        self.playback_agent_combo["values"] = list(agents)

        if episodes:
            self.episode_combo.current(0)
            self.episode_var.set(str(int(episodes[0])))
        if agents:
            self.playback_agent_combo.current(0)
            self.playback_agent_var.set(str(agents[0]))

        self._update_playback_range()
        self._redraw()
        self._render_state()

        self.status.set(
            f"Loaded {self.block_var.get()}: {len(self.df)} rows, "
            f"{len(metric_cols)} metric columns"
        )

    def _create_plots_tab(self):
        plots_tab = Frame(self.results_notebook)
        self.results_notebook.add(plots_tab, text="Plots")

        controls = Frame(plots_tab, padding=(10, 5))
        controls.pack(fill=X)

        Label(controls, text="Plot:").pack(side=LEFT, padx=5)
        self.plot_type_var = StringVar()
        self.plot_type_combo = Combobox(
            controls, textvariable=self.plot_type_var, width=16, state="readonly"
        )
        self.plot_type_combo["values"] = list(PLOT_TYPES.keys())
        self.plot_type_combo.current(0)
        self.plot_type_combo.pack(side=LEFT, padx=5)
        self.plot_type_combo.bind("<<ComboboxSelected>>", self._on_plot_changed)

        Label(controls, text="Metric:").pack(side=LEFT, padx=(15, 5))
        self.metric_var = StringVar()
        self.metric_combo = Combobox(
            controls, textvariable=self.metric_var, width=15, state="readonly"
        )
        self.metric_combo.pack(side=LEFT, padx=5)
        self.metric_combo.bind("<<ComboboxSelected>>", self._on_plot_changed)

        Label(controls, text="Agent:").pack(side=LEFT, padx=(15, 5))
        self.agent_var = StringVar(value="all")
        self.agent_combo = Combobox(
            controls, textvariable=self.agent_var, width=15, state="readonly"
        )
        self.agent_combo.pack(side=LEFT, padx=5)
        self.agent_combo.bind("<<ComboboxSelected>>", self._on_plot_changed)

        self.figure, self.ax = create_figure()
        self.canvas = FigureCanvasTkAgg(self.figure, master=plots_tab)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=5)

    def _create_playback_tab(self):
        playback_tab = Frame(self.results_notebook)
        self.results_notebook.add(playback_tab, text="Playback")

        controls = Frame(playback_tab, padding=(10, 5))
        controls.pack(fill=X)

        Label(controls, text="Episode:").pack(side=LEFT, padx=5)
        self.episode_var = StringVar()
        self.episode_combo = Combobox(
            controls, textvariable=self.episode_var, width=10, state="readonly"
        )
        self.episode_combo.pack(side=LEFT, padx=5)
        self.episode_combo.bind("<<ComboboxSelected>>", self._on_episode_changed)

        Label(controls, text="Agent:").pack(side=LEFT, padx=(15, 5))
        self.playback_agent_var = StringVar()
        self.playback_agent_combo = Combobox(
            controls, textvariable=self.playback_agent_var, width=15, state="readonly"
        )
        self.playback_agent_combo.pack(side=LEFT, padx=5)
        self.playback_agent_combo.bind(
            "<<ComboboxSelected>>", self._on_playback_agent_changed
        )

        self.step_slider = PlaybackControl(
            controls,
            label="Step",
            from_=0,
            to=0,
            on_change=self._on_step_changed,
            get_interval=lambda: int(float(self.export_duration_var.get()) * 1000),
        )
        self.step_slider.pack(side=LEFT, fill=X, expand=True, padx=(15, 5))

        self.state_frame = Frame(playback_tab)
        self.state_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        self.state_display = Label(self.state_frame)
        self.state_display.place(relx=0.5, rely=0.5, anchor="center")
        self.state_frame.bind("<Configure>", lambda e: self._render_state())

        export_frame = Frame(playback_tab, padding=(10, 5))
        export_frame.pack(fill=X)

        self.export_start_slider = ValueSlider(
            export_frame, label="From", from_=0, to=0
        )
        self.export_start_slider.pack(side=LEFT, fill=X, expand=True, padx=5)

        self.export_end_slider = ValueSlider(export_frame, label="To", from_=0, to=0)
        self.export_end_slider.pack(side=LEFT, fill=X, expand=True, padx=5)

        Label(export_frame, text="Duration (s):").pack(side=LEFT, padx=(10, 2))
        self.export_duration_var = StringVar(value="0.7")
        Entry(export_frame, textvariable=self.export_duration_var, width=5).pack(
            side=LEFT, padx=2
        )

        Label(export_frame, text="File:").pack(side=LEFT, padx=(10, 2))
        self.export_filename_var = StringVar(value="playback.mp4")
        Entry(export_frame, textvariable=self.export_filename_var, width=16).pack(
            side=LEFT, padx=2
        )

        Button(export_frame, text="Export", command=self._export_frames).pack(
            side=LEFT, padx=(10, 5)
        )

    def _on_plot_changed(self, event=None):
        self._redraw()

    def _redraw(self):
        if self.df is None:
            return
        plot_type = self.plot_type_var.get()
        metric = self.metric_var.get()
        if not plot_type or not metric:
            return
        agent = self.agent_var.get()
        agents = sorted(self.df["Agent_id"].unique())
        if agent != "all":
            agents = [agent]
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        PLOT_TYPES[plot_type](self.ax, self.df, metric, agents, "Agent_id")
        self.canvas.draw()

    def _on_episode_changed(self, event=None):
        self._update_playback_range()
        self._render_state()

    def _on_playback_agent_changed(self, event=None):
        self._update_playback_range()
        self._render_state()

    def _on_step_changed(self, value):
        self._render_state()

    def _update_playback_range(self):
        episode = self.episode_var.get()
        agent = self.playback_agent_var.get()
        if not episode or not agent:
            return
        filtered = self.df[
            (self.df["Episode"] == int(episode)) & (self.df["Agent_id"] == agent)
        ]
        max_step = int(filtered["Step"].max()) if len(filtered) > 0 else 0
        self.step_slider.set_range(0, max_step)
        self.step_slider.set(0)
        self.export_start_slider.set_range(0, max_step)
        self.export_start_slider.set(0)
        self.export_end_slider.set_range(0, max_step)
        self.export_end_slider.set(max_step)

    def _get_frame(self, step):
        episode = int(self.episode_var.get())
        state_row = self.states[
            (self.states["Episode"] == episode) & (self.states["Step"] == step)
        ].iloc[0]
        board_data = state_row["Board"]
        img = self.renderer.render(
            *self.interpreter.interpret((board_data[0], board_data[1]))
        )
        colors = [ROI_COLOR] * board_data[2].shape[0]
        img = overlay(img, board_data[2], colors, ROI_ALPHA)
        return img

    def _render_state(self):
        if self.states is None or self.interpreter is None:
            return
        episode = self.episode_var.get()
        agent = self.playback_agent_var.get()
        if not episode or not agent:
            return
        img = self._get_frame(self.step_slider.get())
        w, h = self.state_frame.winfo_width(), self.state_frame.winfo_height()
        scale = min(w / img.width, h / img.height)
        scaled = img.resize(
            (int(img.width * scale), int(img.height * scale)), Image.NEAREST
        )
        self._photo = ImageTk.PhotoImage(scaled)
        self.state_display.configure(image=self._photo)
        self.status.set(
            f"Episode {episode}, Agent {agent}, Step {self.step_slider.get()}"
        )

    def _export_plot(self):
        filename = self.results_actions.get_input("Save As")
        save_dir = Path(self.outputs_dir) / self.run_var.get() / self.block_var.get()
        save_figure(self.figure, str(save_dir / filename))
        self.status.set(f"Exported: {save_dir / filename}")

    def _export_frames(self):
        frames = [
            self._get_frame(step)
            for step in range(
                self.export_start_slider.get(), self.export_end_slider.get() + 1
            )
        ]
        output_path = (
            Path(self.outputs_dir)
            / self.run_var.get()
            / self.block_var.get()
            / self.export_filename_var.get()
        )
        save_frames(
            frames,
            str(output_path),
            frame_duration=float(self.export_duration_var.get()),
        )
        self.status.set(f"Exported: {output_path}")

    # =========================================================================
    # Shared
    # =========================================================================

    def _on_close(self):
        self.result = None
        self.root.quit()


def run_gui(
    default_cfg: DictConfig,
    initial_tab: str = "run",
    outputs_dir: str = "outputs",
) -> Optional[DictConfig]:
    """Launch main window. Returns experiment config if Run was clicked, else None."""
    root = tk.Tk()
    root.tk.call("tk", "scaling", 1.0)
    window = MainWindow(root, default_cfg, initial_tab, outputs_dir)
    root.mainloop()
    result = window.result
    root.destroy()
    return result
