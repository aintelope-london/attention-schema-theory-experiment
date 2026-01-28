"""Widget factory for config parameters based on UI schema specs."""

import tkinter as tk
from tkinter import ttk
from typing import Any, Optional, Callable, Tuple


# =============================================================================
# SIZE CONFIGURATION - Edit these values to adjust widget sizes
# =============================================================================
FONT_FAMILY = "TkDefaultFont"
FONT_SIZE = 12

CHECKBOX_SCALE = 2  # Multiplier for checkbox indicator
COMBOBOX_WIDTH = 30
SPINBOX_WIDTH = 15
ENTRY_WIDTH = 30

SPINBOX_BUTTON_WIDTH = 20  # Arrow button width in pixels


# =============================================================================
# Widget creation
# =============================================================================


def create_widget(
    parent: tk.Widget,
    key: str,
    value: Any,
    spec: Optional[list],
    on_change: Callable[[Any], None],
) -> Tuple[tk.Widget, Callable[[Any], None]]:
    """
    Create appropriate widget based on value type and UI schema spec.

    Returns:
        Tuple of (widget, refresher) where refresher is a function that
        updates the widget display from a value.
    """
    font = (FONT_FAMILY, FONT_SIZE)

    range_or_choices = None
    value_type = None

    if spec:
        range_or_choices = spec[0]
        value_type = spec[1]

    # Boolean -> Checkbutton with custom indicator
    if value_type == "bool":
        var = tk.BooleanVar(value=bool(value))

        # Frame to hold custom checkbox
        frame = tk.Frame(parent)

        # Canvas-based checkbox for size control
        size = FONT_SIZE * CHECKBOX_SCALE
        canvas = tk.Canvas(
            frame,
            width=size,
            height=size,
            highlightthickness=1,
            highlightbackground="black",
        )
        canvas.pack(side=tk.LEFT)

        def draw_checkbox():
            canvas.delete("all")
            canvas.create_rectangle(
                2, 2, size - 2, size - 2, outline="black", fill="white"
            )
            if var.get():
                # Draw checkmark
                canvas.create_line(4, size // 2, size // 3, size - 4, width=2)
                canvas.create_line(size // 3, size - 4, size - 4, 4, width=2)

        def toggle(event=None):
            var.set(not var.get())
            draw_checkbox()
            on_change(var.get())

        canvas.bind("<Button-1>", toggle)
        draw_checkbox()

        def refresh_bool(v):
            var.set(bool(v))
            draw_checkbox()

        return frame, refresh_bool

    # String choices -> Combobox
    if (
        isinstance(range_or_choices, list)
        and range_or_choices
        and isinstance(range_or_choices[0], str)
    ):
        var = tk.StringVar(value=str(value))
        widget = ttk.Combobox(
            parent,
            textvariable=var,
            values=range_or_choices,
            state="readonly",
            width=COMBOBOX_WIDTH,
            font=font,
        )
        widget.bind("<<ComboboxSelected>>", lambda e: on_change(var.get()))

        def refresh_combobox(v):
            var.set(str(v))

        return widget, refresh_combobox

    # Numeric range -> Spinbox
    if isinstance(range_or_choices, list) and len(range_or_choices) == 2:
        min_val, max_val = range_or_choices

        if value_type == "int":
            var = tk.IntVar(value=int(value))
            increment = 1
        else:
            var = tk.DoubleVar(value=float(value))
            range_size = max_val - min_val
            increment = 0.01 if range_size <= 1 else (0.1 if range_size <= 10 else 1.0)

        # Frame for custom spinbox with larger buttons
        frame = tk.Frame(parent)

        entry = tk.Entry(frame, textvariable=var, width=SPINBOX_WIDTH, font=font)
        entry.pack(side=tk.LEFT)

        button_frame = tk.Frame(frame)
        button_frame.pack(side=tk.LEFT)

        def validate_and_clamp():
            try:
                val = float(entry.get())
                clamped = max(min_val, min(max_val, val))
                if value_type == "int":
                    clamped = int(clamped)
                var.set(clamped)
                on_change(clamped)
            except ValueError:
                var.set(value)

        def increment_value():
            try:
                val = float(entry.get()) + increment
                clamped = max(min_val, min(max_val, val))
                if value_type == "int":
                    clamped = int(clamped)
                var.set(clamped)
                on_change(clamped)
            except ValueError:
                pass

        def decrement_value():
            try:
                val = float(entry.get()) - increment
                clamped = max(min_val, min(max_val, val))
                if value_type == "int":
                    clamped = int(clamped)
                var.set(clamped)
                on_change(clamped)
            except ValueError:
                pass

        btn_up = tk.Button(
            button_frame,
            text="▲",
            command=increment_value,
            width=2,
            font=(FONT_FAMILY, FONT_SIZE),
        )
        btn_up.pack(side=tk.TOP)

        btn_down = tk.Button(
            button_frame,
            text="▼",
            command=decrement_value,
            width=2,
            font=(FONT_FAMILY, FONT_SIZE),
        )
        btn_down.pack(side=tk.TOP)

        entry.bind("<FocusOut>", lambda e: validate_and_clamp())
        entry.bind("<Return>", lambda e: validate_and_clamp())

        def refresh_spinbox(v):
            if value_type == "int":
                var.set(int(v))
            else:
                var.set(float(v))

        return frame, refresh_spinbox

    # Default -> Entry
    var = tk.StringVar(value=str(value))
    widget = tk.Entry(parent, textvariable=var, width=ENTRY_WIDTH, font=font)
    widget.bind("<FocusOut>", lambda e: on_change(var.get()))
    widget.bind("<Return>", lambda e: on_change(var.get()))

    def refresh_entry(v):
        var.set(str(v))

    return widget, refresh_entry


def get_range_display(spec: Optional[list]) -> str:
    """Get human-readable range/choices display from UI schema spec."""
    if not spec:
        return ""

    range_or_choices = spec[0]

    if range_or_choices is None:
        return ""

    if isinstance(range_or_choices, list):
        if len(range_or_choices) == 2 and not isinstance(range_or_choices[0], str):
            return f"Range: {range_or_choices[0]} - {range_or_choices[1]}"
        else:
            return f"Choices: {', '.join(map(str, range_or_choices))}"

    return ""
