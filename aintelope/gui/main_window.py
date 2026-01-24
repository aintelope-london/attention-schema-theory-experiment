"""Main GUI window for creating/editing configs."""

import tkinter as tk
from tkinter import ttk
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from omegaconf import DictConfig, OmegaConf

from gui.ui_schema_manager import load_ui_schema, get_field_spec
from gui.widgets import create_widget, get_range_display

CONFIG_DIR = Path("aintelope") / "config"

class ConfigGUI:
    def __init__(self, root: tk.Tk, default_cfg: DictConfig):
        self.root = root
        self.root.title("Config Editor")
        self.root.geometry("1000x700")
        
        self.default_cfg = default_cfg
        self.ui_schema = load_ui_schema()
        
        # Current edited config (starts as copy of default)
        self.current_config = OmegaConf.to_container(default_cfg, resolve=True)
        
        # Track result
        self.result = None
        
        # Widget references for updating values
        self.widgets = {}
        
        self._create_ui()
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _create_ui(self):
        # Top bar
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Config:").pack(side=tk.LEFT, padx=5)
        
        self.config_var = tk.StringVar(value="config_experiment.yaml")
        config_combo = ttk.Combobox(top_frame, textvariable=self.config_var, width=40)
        config_combo['values'] = self._list_configs()
        config_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(top_frame, text="Load", command=self._load_config).pack(side=tk.LEFT, padx=5)
        
        # Main editor area
        editor_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        editor_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollable canvas
        canvas = tk.Canvas(editor_frame)
        scrollbar = ttk.Scrollbar(editor_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Build parameter tree
        self._build_param_tree(self.scrollable_frame, self.default_cfg)
        
        # Bottom actions
        action_frame = ttk.Frame(self.root, padding=10)
        action_frame.pack(fill=tk.X)
        
        ttk.Label(action_frame, text="Save As:").pack(side=tk.LEFT, padx=5)
        self.output_var = tk.StringVar(value=self._generate_filename())
        ttk.Entry(action_frame, textvariable=self.output_var, width=40).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="Save Config", command=self._save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Run", command=self._run).pack(side=tk.LEFT, padx=20)
        ttk.Button(action_frame, text="Cancel", command=self._on_close).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Edit parameters or load existing config.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
    
    def _build_param_tree(self, parent, config, path="", level=0):
        """Recursively build parameter tree from config."""
        row = 0
        
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, (dict, DictConfig)):
                # Section header
                frame = ttk.Frame(parent)
                frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(10 if level == 0 else 5, 2))
                
                label = ttk.Label(
                    frame,
                    text=key.replace('_', ' ').title(),
                    font=('Arial', 10 if level == 0 else 9, 'bold')
                )
                label.pack(side=tk.LEFT, padx=(level * 20, 0))
                
                if level == 0:
                    sep = ttk.Separator(parent, orient=tk.HORIZONTAL)
                    sep.grid(row=row + 1, column=0, columnspan=3, sticky='ew', pady=(0, 5))
                    row += 2
                else:
                    row += 1
                
                nested_frame = ttk.Frame(parent)
                nested_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W)
                self._build_param_tree(nested_frame, value, current_path, level + 1)
                row += 1
            else:
                # Parameter row
                param_frame = ttk.Frame(parent)
                param_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)
                
                label = ttk.Label(param_frame, text=f"{key}:", width=25, anchor=tk.W)
                label.pack(side=tk.LEFT, padx=(level * 20, 10))
                
                spec = get_field_spec(self.ui_schema, current_path)
                
                widget = create_widget(
                    param_frame,
                    key,
                    value,
                    spec,
                    lambda v, p=current_path: self._update_value(p, v)
                )
                widget.pack(side=tk.LEFT, padx=5)
                
                range_display = get_range_display(spec)
                if range_display:
                    info_label = ttk.Label(param_frame, text=range_display, foreground='gray')
                    info_label.pack(side=tk.LEFT, padx=10)
                
                self.widgets[current_path] = widget
                row += 1
    
    def _update_value(self, path: str, value: Any):
        """Update config when a value changes."""
        keys = path.split('.')
        
        current = self.current_config
        for key in keys[:-1]:
            current = current[key]
        
        spec = get_field_spec(self.ui_schema, path)
        if spec:
            value_type = spec[1]
            if value_type == "bool":
                value = bool(value)
            elif value_type == "int":
                value = int(float(value))
            elif value_type == "float":
                value = float(value)
        
        current[keys[-1]] = value
        self.status_var.set(f"Modified: {path}")
    
    def _generate_filename(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"config_{timestamp}.yaml"
    
    def _list_configs(self) -> list:
        return sorted([f.name for f in CONFIG_DIR.glob("config_*.yaml")])
    
    def _load_config(self):
        """Load config file, applying it over default."""
        config_name = self.config_var.get()
        config_path = CONFIG_DIR / config_name
        
        loaded = OmegaConf.load(config_path)
        merged = OmegaConf.merge(self.default_cfg, loaded)
        self.current_config = OmegaConf.to_container(merged, resolve=True)
        
        self._refresh_widgets()
        self.status_var.set(f"Loaded: {config_name}")
    
    def _save_config(self):
        """Save current config."""
        output_name = self.output_var.get()
        if not output_name.endswith('.yaml'):
            output_name += '.yaml'
        
        output_path = CONFIG_DIR / output_name
        OmegaConf.save(OmegaConf.create(self.current_config), output_path)
        
        self.status_var.set(f"Saved: {output_name}")
        self.config_var.set(output_name)
    
    def _run(self):
        """Return config and close GUI."""
        self.result = OmegaConf.create(self.current_config)
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
                    widget = self.widgets[current_path]
                    if isinstance(widget, ttk.Checkbutton):
                        widget.state(['!alternate'])
                        if value:
                            widget.state(['selected'])
                        else:
                            widget.state(['!selected'])
                    elif isinstance(widget, (ttk.Combobox, ttk.Spinbox, ttk.Entry)):
                        widget.delete(0, tk.END)
                        widget.insert(0, str(value))
        
        update_widgets(self.current_config)


def run_gui(default_cfg: DictConfig) -> Optional[DictConfig]:
    """
    Launch GUI for creating/editing config.
    
    Args:
        default_cfg: Base config from Hydra (config_experiment.yaml)
        
    Returns:
        DictConfig: Config to run with
        None: User cancelled
    """
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 1.0)
    gui = ConfigGUI(root, default_cfg)
    root.mainloop()
    root.destroy()
    
    return gui.result
