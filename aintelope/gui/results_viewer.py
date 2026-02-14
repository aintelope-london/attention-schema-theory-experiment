"""Results viewer GUI for inspecting experiment outputs."""

from pathlib import Path

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from aintelope.analytics.plotting import create_figure, plot_line, save_figure
from aintelope.analytics.recording import list_runs, list_blocks, EventLog
from aintelope.gui.gui import (
    Frame,
    Label,
    Combobox,
    ActionBar,
    StatusBar,
    StringVar,
    launch_window,
    X,
    LEFT,
    BOTH,
)


class ResultsViewer:
    def __init__(self, root, outputs_dir):
        self.root = root
        self.root.title("Results Viewer")
        self.root.geometry("1200x800")
        self.outputs_dir = outputs_dir
        self.df = None
        self.result = None

        self._create_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_ui(self):
        # Top bar — run and block selectors
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

        # Plot controls — axis column selectors
        controls_frame = Frame(self.root, padding=(10, 0))
        controls_frame.pack(fill=X)

        Label(controls_frame, text="X:").pack(side=LEFT, padx=5)
        self.x_var = StringVar()
        self.x_combo = Combobox(
            controls_frame, textvariable=self.x_var, width=20, state="readonly"
        )
        self.x_combo.pack(side=LEFT, padx=5)
        self.x_combo.bind("<<ComboboxSelected>>", self._on_axis_changed)

        Label(controls_frame, text="Y:").pack(side=LEFT, padx=(20, 5))
        self.y_var = StringVar()
        self.y_combo = Combobox(
            controls_frame, textvariable=self.y_var, width=20, state="readonly"
        )
        self.y_combo.pack(side=LEFT, padx=5)
        self.y_combo.bind("<<ComboboxSelected>>", self._on_axis_changed)

        # Matplotlib canvas
        canvas_frame = Frame(self.root)
        canvas_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)

        self.figure, self.ax = create_figure()
        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

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
        filepath = Path(self.outputs_dir) / run_name / block_name / "events.csv"
        self.df = EventLog.read(str(filepath))

        numeric_cols = list(self.df.select_dtypes(include="number").columns)
        self.x_combo["values"] = numeric_cols
        self.y_combo["values"] = numeric_cols

        if "Episode" in numeric_cols:
            self.x_var.set("Episode")
        elif numeric_cols:
            self.x_combo.current(0)

        if "Reward" in numeric_cols:
            self.y_var.set("Reward")
        elif len(numeric_cols) > 1:
            self.y_combo.current(1)

        self._redraw()
        self.status.set(
            f"Loaded {block_name}: {len(self.df)} rows, "
            f"{len(numeric_cols)} numeric columns"
        )

    def _on_axis_changed(self, event=None):
        self._redraw()

    def _redraw(self):
        if self.df is None:
            return
        x_col = self.x_var.get()
        y_col = self.y_var.get()
        if not x_col or not y_col:
            return

        plot_line(self.ax, self.df, x_col, [y_col])
        self.canvas.draw()

    def _export_plot(self):
        filename = self.actions.get_input("Save As")
        run_name = self.run_var.get()
        block_name = self.block_var.get()
        save_dir = Path(self.outputs_dir) / run_name / block_name
        save_path = save_dir / filename
        save_figure(self.figure, str(save_path))
        self.status.set(f"Exported: {save_path}")

    def _on_close(self):
        self.result = None
        self.root.quit()


def run_results_viewer(outputs_dir="outputs"):
    """Launch results viewer GUI."""
    return launch_window(ResultsViewer, outputs_dir)
