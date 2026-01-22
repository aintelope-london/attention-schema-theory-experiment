"""Widget factory for config parameters based on UI schema specs."""

import tkinter as tk
from tkinter import ttk
from typing import Any, Optional, Callable


def create_widget(
    parent: tk.Widget,
    key: str,
    value: Any,
    spec: Optional[list],
    on_change: Callable[[Any], None]
) -> tk.Widget:
    """
    Create appropriate widget based on value type and UI schema spec.
    
    Args:
        parent: Parent widget
        key: Parameter name (for debugging)
        value: Current value from config
        spec: UI schema spec: [[min, max], "type"] or [null, "type"] or [[choices...], "type"]
        on_change: Callback when value changes
        
    Returns:
        Configured widget
    """
    range_or_choices = None
    value_type = None
    
    if spec:
        range_or_choices = spec[0]
        value_type = spec[1]
    
    # Boolean -> Checkbutton
    if value_type == "bool":
        var = tk.BooleanVar(value=bool(value))
        widget = ttk.Checkbutton(
            parent,
            variable=var,
            command=lambda: on_change(var.get())
        )
        return widget
    
    # String choices -> Combobox
    if isinstance(range_or_choices, list) and range_or_choices and isinstance(range_or_choices[0], str):
        var = tk.StringVar(value=str(value))
        widget = ttk.Combobox(
            parent,
            textvariable=var,
            values=range_or_choices,
            state='readonly',
            width=20
        )
        widget.bind('<<ComboboxSelected>>', lambda e: on_change(var.get()))
        return widget
    
    # Numeric range -> Spinbox
    if isinstance(range_or_choices, list) and len(range_or_choices) == 2:
        min_val, max_val = range_or_choices
        
        if value_type == "int":
            var = tk.IntVar(value=int(value))
            increment = 1
        else:
            var = tk.DoubleVar(value=float(value))
            range_size = max_val - min_val
            increment = 0.01 if range_size < 1 else (0.1 if range_size < 10 else 1.0)
        
        widget = ttk.Spinbox(
            parent,
            from_=min_val,
            to=max_val,
            textvariable=var,
            increment=increment,
            width=15
        )
        widget.bind('<FocusOut>', lambda e: on_change(var.get()))
        widget.bind('<Return>', lambda e: on_change(var.get()))
        return widget
    
    # Default -> Entry
    var = tk.StringVar(value=str(value))
    widget = ttk.Entry(parent, textvariable=var, width=30)
    widget.bind('<FocusOut>', lambda e: on_change(var.get()))
    widget.bind('<Return>', lambda e: on_change(var.get()))
    return widget


def get_range_display(spec: Optional[list]) -> str:
    """Get human-readable range/choices display from UI schema spec."""
    if not spec:
        return ""
    
    range_or_choices = spec[0]
    
    if range_or_choices is None:
        return ""
    
    if len(range_or_choices) == 2 and not isinstance(range_or_choices[0], str):
        return f"Range: {range_or_choices[0]} - {range_or_choices[1]}"
    else:
        return f"Choices: {', '.join(map(str, range_or_choices))}"