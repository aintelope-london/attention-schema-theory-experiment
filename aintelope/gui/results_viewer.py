"""Results viewer GUI for inspecting experiment outputs."""

from pathlib import Path

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from aintelope.analytics.plotting import (
    create_figure,
    save_figure,
    prepare_plot_data,
    PLOT_TYPES,
)
from aintelope.analytics.recording import list_runs, list_blocks, EventLog, frames_to_video
from aintelope.gui.gui import (
    Button,
    Entry,
    Frame,
    Label,
    Combobox,
    Notebook,
    ActionBar,
    StatusBar,
    ValueSlider,
    StringVar,
    Text,
    launch_window,
    X,
    LEFT,
    BOTH,
)
from aintelope.gui.renderer import SavannaInterpreter, StateRenderer


class ResultsViewer:
    def __init__(self, root, outputs_dir):
        self.root = root
        self.root.title("Results Viewer")
        self.root.geometry("1200x800")
        self.outputs_dir = outputs_dir
        self.df = None
        self.metadata = None
        self.interpreter = None
        self.renderer = StateRenderer()
        self.result = None

        self._create_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_ui(self):
        # Top bar — run and block selectors (shared across tabs)
        selector_frame = Frame(self.root, padding=10)
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

        # Notebook with tabs
        self.notebook = Notebook(self.root)
        self.notebook.pack(fill=BOTH, expand=True, padx=10, pady=5)

        self._create_plots_tab()
        self._create_playback_tab()

        # Actions
        self.actions = ActionBar(
            self.root,
            inputs=[("Save As", "plot.png")],
            buttons=[
                ("Export Plot", self._export_plot),
                ("Close", self._on_close),
            ],
        )
        self.actions.pack(fill=X)

        # Status bar
        self.status = StatusBar(self.root, "Select a run to begin.")
        self.status.pack(fill=X, padx=10, pady=(0, 10))

        self._refresh_runs()

    # =========================================================================
    # Plots tab
    # =========================================================================

    def _create_plots_tab(self):
        self.plots_tab = Frame(self.notebook)
        self.notebook.add(self.plots_tab, text="Plots")

        controls = Frame(self.plots_tab, padding=(10, 5))
        controls.pack(fill=X)

        Label(controls, text="Plot:").pack(side=LEFT, padx=5)
        self.plot_type_var = StringVar()
        self.plot_type_combo = Combobox(
            controls, textvariable=self.plot_type_var, width=12, state="readonly"
        )
        self.plot_type_combo["values"] = list(PLOT_TYPES.keys())
        self.plot_type_combo.current(0)
        self.plot_type_combo.pack(side=LEFT, padx=5)
        self.plot_type_combo.bind("<<ComboboxSelected>>", self._on_plot_changed)

        Label(controls, text="Agent:").pack(side=LEFT, padx=(15, 5))
        self.agent_var = StringVar(value="all")
        self.agent_combo = Combobox(
            controls, textvariable=self.agent_var, width=15, state="readonly"
        )
        self.agent_combo.pack(side=LEFT, padx=5)
        self.agent_combo.bind("<<ComboboxSelected>>", self._on_plot_changed)

        Label(controls, text="X:").pack(side=LEFT, padx=(15, 5))
        self.x_var = StringVar()
        self.x_combo = Combobox(
            controls, textvariable=self.x_var, width=15, state="readonly"
        )
        self.x_combo.pack(side=LEFT, padx=5)
        self.x_combo.bind("<<ComboboxSelected>>", self._on_plot_changed)

        Label(controls, text="Y:").pack(side=LEFT, padx=(15, 5))
        self.y_var = StringVar()
        self.y_combo = Combobox(
            controls, textvariable=self.y_var, width=15, state="readonly"
        )
        self.y_combo.pack(side=LEFT, padx=5)
        self.y_combo.bind("<<ComboboxSelected>>", self._on_plot_changed)

        # Matplotlib canvas
        self.figure, self.ax = create_figure()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plots_tab)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=5)

    # =========================================================================
    # Playback tab
    # =========================================================================

    def _create_playback_tab(self):
        self.playback_tab = Frame(self.notebook)
        self.notebook.add(self.playback_tab, text="Playback")

        controls = Frame(self.playback_tab, padding=(10, 5))
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
            controls,
            textvariable=self.playback_agent_var,
            width=15,
            state="readonly",
        )
        self.playback_agent_combo.pack(side=LEFT, padx=5)
        self.playback_agent_combo.bind(
            "<<ComboboxSelected>>", self._on_playback_agent_changed
        )

        self.step_slider = ValueSlider(
            controls, label="Step", from_=0, to=0, on_change=self._on_step_changed
        )
        self.step_slider.pack(side=LEFT, fill=X, expand=True, padx=(15, 5))

        # State display — monospaced text
        self.state_display = Text(
            self.playback_tab, font=("Courier", 14), state="disabled", wrap="none"
        )
        self.state_display.pack(fill=BOTH, expand=True, padx=10, pady=5)

        # Video export controls
        export_frame = Frame(self.playback_tab, padding=(10, 5))
        export_frame.pack(fill=X)

        self.export_start_slider = ValueSlider(
            export_frame, label="From", from_=0, to=0
        )
        self.export_start_slider.pack(side=LEFT, fill=X, expand=True, padx=5)

        self.export_end_slider = ValueSlider(
            export_frame, label="To", from_=0, to=0
        )
        self.export_end_slider.pack(side=LEFT, fill=X, expand=True, padx=5)

        Label(export_frame, text="Duration (s):").pack(side=LEFT, padx=(10, 2))
        self.export_duration_var = StringVar(value="0.7")
        Entry(export_frame, textvariable=self.export_duration_var, width=5).pack(
            side=LEFT, padx=2
        )

        Button(export_frame, text="Export Video", command=self._export_video).pack(
            side=LEFT, padx=(10, 5)
        )

    # =========================================================================
    # Data loading
    # =========================================================================

    def _refresh_runs(self):
        runs = list_runs(self.outputs_dir)
        self.run_combo["values"] = runs
        if runs:
            self.run_combo.current(0)
            self._on_run_selected()

    def _on_run_selected(self, event=None):
        run_name = self.run_var.get()
        run_dir = str(Path(self.outputs_dir) / run_name)
        blocks = list_blocks(run_dir)
        self.block_combo["values"] = blocks
        if blocks:
            self.block_combo.current(0)
            self._on_block_selected()
        self.status.set(f"Run: {run_name} ({len(blocks)} blocks)")

    def _on_block_selected(self, event=None):
        run_name = self.run_var.get()
        block_name = self.block_var.get()
        block_dir = Path(self.outputs_dir) / run_name / block_name

        self.df = EventLog.read(str(block_dir / "events.csv"))
        self.metadata = EventLog.read_metadata(str(block_dir))
        self.interpreter = SavannaInterpreter(self.metadata["layer_order"])

        # Populate plots tab controls
        numeric_cols = list(self.df.select_dtypes(include="number").columns)
        self.x_combo["values"] = numeric_cols
        self.y_combo["values"] = numeric_cols

        agents = sorted(self.df["Agent_id"].unique())
        self.agent_combo["values"] = ["all"] + list(agents)
        self.agent_var.set("all")

        if "Episode" in numeric_cols:
            self.x_var.set("Episode")
        elif numeric_cols:
            self.x_combo.current(0)

        if "Reward" in numeric_cols:
            self.y_var.set("Reward")
        elif len(numeric_cols) > 1:
            self.y_combo.current(1)

        # Populate playback tab controls
        episodes = sorted(self.df["Episode"].unique())
        self.episode_combo["values"] = [str(int(e)) for e in episodes]
        self.playback_agent_combo["values"] = list(agents)

        if episodes:
            self.episode_combo.current(0)
        if agents:
            self.playback_agent_combo.current(0)

        self._update_playback_range()
        self._redraw()
        self._render_state()

        self.status.set(
            f"Loaded {block_name}: {len(self.df)} rows, "
            f"{len(numeric_cols)} numeric columns"
        )

    # =========================================================================
    # Plots tab callbacks
    # =========================================================================

    def _on_plot_changed(self, event=None):
        self._redraw()

    def _redraw(self):
        if self.df is None:
            return
        x_col = self.x_var.get()
        y_col = self.y_var.get()
        plot_type = self.plot_type_var.get()
        if not x_col or not y_col or not plot_type:
            return

        filter_by = {}
        agent = self.agent_var.get()
        if agent != "all":
            filter_by["Agent_id"] = agent

        plot_df = prepare_plot_data(self.df, filter_by=filter_by)
        PLOT_TYPES[plot_type](self.ax, plot_df, x_col=x_col, y_cols=[y_col])
        self.canvas.draw()

    # =========================================================================
    # Playback tab callbacks
    # =========================================================================

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

    def _get_frame_lines(self, step):
        """Get rendered ASCII lines for a single step in current episode/agent."""
        row = self.df[
            (self.df["Episode"] == int(self.episode_var.get()))
            & (self.df["Agent_id"] == self.playback_agent_var.get())
            & (self.df["Step"] == step)
        ]
        state = row.iloc[0]["State"]
        return self.renderer.render(*self.interpreter.interpret(state))

    def _render_state(self):
        if self.df is None or self.interpreter is None:
            return
        episode = self.episode_var.get()
        agent = self.playback_agent_var.get()
        step = self.step_slider.get()
        if not episode or not agent:
            return

        lines = self._get_frame_lines(step)

        self.state_display.configure(state="normal")
        self.state_display.delete("1.0", "end")
        self.state_display.insert("1.0", "\n".join(lines))
        self.state_display.configure(state="disabled")

        self.status.set(f"Episode {episode}, Agent {agent}, Step {step}")

    # =========================================================================
    # Shared
    # =========================================================================

    def _export_plot(self):
        filename = self.actions.get_input("Save As")
        run_name = self.run_var.get()
        block_name = self.block_var.get()
        save_dir = Path(self.outputs_dir) / run_name / block_name
        save_path = save_dir / filename
        save_figure(self.figure, str(save_path))
        self.status.set(f"Exported: {save_path}")

    def _export_video(self):
        start = self.export_start_slider.get()
        end = self.export_end_slider.get()
        duration = float(self.export_duration_var.get())

        frames = [self._get_frame_lines(step) for step in range(start, end + 1)]

        run_name = self.run_var.get()
        block_name = self.block_var.get()
        save_dir = Path(self.outputs_dir) / run_name / block_name
        output_path = save_dir / "playback.mp4"

        frames_to_video(frames, str(output_path), frame_duration=duration)
        self.status.set(f"Exported: {output_path}")

    def _on_close(self):
        self.result = None
        self.root.quit()


def run_results_viewer(outputs_dir="outputs"):
    """Launch results viewer GUI."""
    return launch_window(ResultsViewer, outputs_dir)