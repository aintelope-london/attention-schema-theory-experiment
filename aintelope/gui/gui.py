"""Shared GUI building blocks for all viewer windows.

This is the only module in the project that imports tkinter.
Viewers import all GUI primitives from here.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Callable, Optional, Tuple


# =============================================================================
# Re-exports for viewers (so they never import tkinter directly)
# =============================================================================
Frame = ttk.Frame
Text = tk.Text
Label = ttk.Label
Separator = ttk.Separator
Notebook = ttk.Notebook
Combobox = ttk.Combobox
Button = ttk.Button
Entry = ttk.Entry
Scale = ttk.Scale
StringVar = tk.StringVar
IntVar = tk.IntVar

# Layout constants
W = tk.W
X = tk.X
Y = tk.Y
LEFT = tk.LEFT
RIGHT = tk.RIGHT
TOP = tk.TOP
BOTH = tk.BOTH
HORIZONTAL = tk.HORIZONTAL
VERTICAL = tk.VERTICAL


# =============================================================================
# Size configuration
# =============================================================================
FONT_FAMILY = "TkDefaultFont"
FONT_SIZE = 12

CHECKBOX_SCALE = 2
COMBOBOX_WIDTH = 30
SPINBOX_WIDTH = 15
ENTRY_WIDTH = 30


# =============================================================================
# Layout components
# =============================================================================


class _PackGridMixin:
    """Delegates pack/grid to self.frame."""

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    def __str__(self):
        return str(self.frame)


class ScrollableFrame(_PackGridMixin):
    """Canvas + scrollbar wrapper that provides a scrollable content frame."""

    def __init__(self, parent, **label_frame_kwargs):
        self.frame = ttk.LabelFrame(parent, **label_frame_kwargs)

        self._canvas = tk.Canvas(self.frame)
        self._scrollbar = ttk.Scrollbar(
            self.frame, orient=tk.VERTICAL, command=self._canvas.yview
        )
        self.content = ttk.Frame(self._canvas)

        self.content.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )

        self._canvas.create_window((0, 0), window=self.content, anchor=tk.NW)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)


class SelectorBar(_PackGridMixin):
    """Labeled combobox with a button, for selecting from a list of values."""

    def __init__(
        self,
        parent,
        label: str,
        values: list,
        on_select: Callable[[str], None],
        button_text: str = "Load",
    ):
        self.frame = ttk.Frame(parent, padding=10)
        self._on_select = on_select

        ttk.Label(self.frame, text=f"{label}:").pack(side=tk.LEFT, padx=5)

        self._var = tk.StringVar(value=values[0])
        self._combo = ttk.Combobox(
            self.frame, textvariable=self._var, width=40, state="readonly"
        )
        self._combo["values"] = values
        self._combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(self.frame, text=button_text, command=self._trigger).pack(
            side=tk.LEFT, padx=5
        )

    def _trigger(self):
        self._on_select(self._var.get())

    def set_values(self, values: list):
        self._combo["values"] = values
        if values:
            self._var.set(values[0])

    def get(self) -> str:
        return self._var.get()


class ValueSlider(_PackGridMixin):
    """Labeled slider with a synced numeric entry field.

    The slider and entry stay in sync: dragging the slider updates the entry,
    editing the entry updates the slider. Calls on_change with the new int value.

    Usage:
        slider = ValueSlider(parent, label="Step", from_=0, to=100,
                             on_change=my_callback)
        slider.set_range(0, 50)
        slider.set(25)
        current = slider.get()
    """

    def __init__(
        self,
        parent,
        label: str,
        from_: int = 0,
        to: int = 100,
        on_change: Callable[[int], None] = lambda v: None,
    ):
        self.frame = ttk.Frame(parent)
        self._on_change = on_change
        self._var = tk.IntVar(value=from_)

        ttk.Label(self.frame, text=f"{label}:").pack(side=tk.LEFT, padx=5)

        self._scale = ttk.Scale(
            self.frame,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=self._var,
            command=self._on_scale,
        )
        self._scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self._entry = ttk.Entry(self.frame, textvariable=self._var, width=8)
        self._entry.pack(side=tk.LEFT, padx=5)
        self._entry.bind("<Return>", self._on_entry)
        self._entry.bind("<FocusOut>", self._on_entry)

    def _on_scale(self, raw_value):
        value = int(float(raw_value))
        self._var.set(value)
        self._on_change(value)

    def _on_entry(self, event=None):
        value = int(float(self._var.get()))
        low = int(float(self._scale.cget("from")))
        high = int(float(self._scale.cget("to")))
        value = max(low, min(high, value))
        self._var.set(value)
        self._on_change(value)

    def set_range(self, from_: int, to: int):
        """Update slider range."""
        self._scale.configure(from_=from_, to=to)

    def set(self, value: int):
        """Set current value."""
        self._var.set(value)

    def get(self) -> int:
        """Get current value."""
        return self._var.get()


class ActionBar(_PackGridMixin):
    """Bottom action bar with optional text inputs and buttons.

    Usage:
        bar = ActionBar(parent,
            inputs=[("Save As", "default.yaml")],
            buttons=[("Save", save_cb), ("Run", run_cb), ("Cancel", close_cb)],
        )
        bar.get_input("Save As")  # returns current text value
    """

    def __init__(
        self,
        parent,
        buttons: list[tuple[str, Callable]],
        inputs: Optional[list[tuple[str, str]]] = None,
    ):
        self.frame = ttk.Frame(parent, padding=10)
        self._inputs = {}

        if inputs:
            for label, default in inputs:
                ttk.Label(self.frame, text=f"{label}:").pack(side=tk.LEFT, padx=5)
                var = tk.StringVar(value=default)
                ttk.Entry(self.frame, textvariable=var, width=40).pack(
                    side=tk.LEFT, padx=5
                )
                self._inputs[label] = var

        for label, callback in buttons:
            ttk.Button(self.frame, text=label, command=callback).pack(
                side=tk.LEFT, padx=5
            )

    def get_input(self, label: str) -> str:
        return self._inputs[label].get()


class StatusBar:
    """Sunken label at the bottom of a window."""

    def __init__(self, parent, initial_text: str = "Ready."):
        self._var = tk.StringVar(value=initial_text)
        self._label = ttk.Label(
            parent, textvariable=self._var, relief=tk.SUNKEN, anchor=tk.W
        )

    def set(self, text: str):
        self._var.set(text)

    def pack(self, **kwargs):
        self._label.pack(**kwargs)

    def grid(self, **kwargs):
        self._label.grid(**kwargs)


# =============================================================================
# Window launcher
# =============================================================================


def launch_window(window_class, *args, **kwargs):
    """Create a Tk root, instantiate window_class(root, *args, **kwargs),
    run mainloop, destroy, and return window.result."""
    root = tk.Tk()
    root.tk.call("tk", "scaling", 1.0)
    window = window_class(root, *args, **kwargs)
    root.mainloop()
    root.destroy()
    return window.result


# =============================================================================
# Widget creation (config parameter widgets)
# =============================================================================


def create_widget(
    parent,
    key: str,
    value: Any,
    spec: Optional[list],
    on_change: Callable[[Any], None],
) -> Tuple[Any, Callable[[Any], None]]:
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

    # Locked -> disabled Entry (no @ui annotation in config)
    if value_type == "locked":
        var = tk.StringVar(value=str(value))
        widget = tk.Entry(
            parent,
            textvariable=var,
            width=ENTRY_WIDTH,
            font=font,
            state="disabled",
            disabledforeground="gray",
        )
        return widget, lambda v: var.set(str(v))

    # Boolean -> Checkbutton with custom indicator
    if value_type == "bool":
        var = tk.BooleanVar(value=bool(value))

        frame = tk.Frame(parent)

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

        frame = tk.Frame(parent)

        entry = tk.Entry(frame, textvariable=var, width=SPINBOX_WIDTH, font=font)
        entry.pack(side=tk.LEFT)

        button_frame = tk.Frame(frame)
        button_frame.pack(side=tk.LEFT)

        def _set_clamped(val):
            clamped = max(min_val, min(max_val, val))
            if value_type == "int":
                clamped = int(clamped)
            var.set(clamped)
            on_change(clamped)

        def validate_and_clamp():
            try:
                _set_clamped(float(entry.get()))
            except ValueError:
                var.set(value)

        tk.Button(
            button_frame,
            text="▲",
            command=lambda: _set_clamped(float(entry.get()) + increment),
            width=2,
            font=(FONT_FAMILY, FONT_SIZE),
        ).pack(side=tk.TOP)

        tk.Button(
            button_frame,
            text="▼",
            command=lambda: _set_clamped(float(entry.get()) - increment),
            width=2,
            font=(FONT_FAMILY, FONT_SIZE),
        ).pack(side=tk.TOP)

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


def ask_yes_no(title, message):
    """Prompt user with a yes/no dialog."""
    return messagebox.askyesno(title, message)


"""Shared GUI building blocks for all viewer windows.

This is the only module in the project that imports tkinter.
Viewers import all GUI primitives from here.
"""


# =============================================================================
# Re-exports for viewers (so they never import tkinter directly)
# =============================================================================
Frame = ttk.Frame
Text = tk.Text
Label = ttk.Label
Separator = ttk.Separator
Notebook = ttk.Notebook
Combobox = ttk.Combobox
Button = ttk.Button
Entry = ttk.Entry
Scale = ttk.Scale
StringVar = tk.StringVar
IntVar = tk.IntVar

# Layout constants
W = tk.W
X = tk.X
Y = tk.Y
LEFT = tk.LEFT
RIGHT = tk.RIGHT
TOP = tk.TOP
BOTH = tk.BOTH
HORIZONTAL = tk.HORIZONTAL
VERTICAL = tk.VERTICAL


# =============================================================================
# Size configuration
# =============================================================================
FONT_FAMILY = "TkDefaultFont"
FONT_SIZE = 12

CHECKBOX_SCALE = 2
COMBOBOX_WIDTH = 30
SPINBOX_WIDTH = 15
ENTRY_WIDTH = 30


# =============================================================================
# Layout components
# =============================================================================


class _PackGridMixin:
    """Delegates pack/grid to self.frame."""

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    def __str__(self):
        return str(self.frame)


class ScrollableFrame(_PackGridMixin):
    """Canvas + scrollbar wrapper that provides a scrollable content frame."""

    def __init__(self, parent, **label_frame_kwargs):
        self.frame = ttk.LabelFrame(parent, **label_frame_kwargs)

        self._canvas = tk.Canvas(self.frame)
        self._scrollbar = ttk.Scrollbar(
            self.frame, orient=tk.VERTICAL, command=self._canvas.yview
        )
        self.content = ttk.Frame(self._canvas)

        self.content.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )

        self._canvas.create_window((0, 0), window=self.content, anchor=tk.NW)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)


class SelectorBar(_PackGridMixin):
    """Labeled combobox with a button, for selecting from a list of values."""

    def __init__(
        self,
        parent,
        label: str,
        values: list,
        on_select: Callable[[str], None],
        button_text: str = "Load",
    ):
        self.frame = ttk.Frame(parent, padding=10)
        self._on_select = on_select

        ttk.Label(self.frame, text=f"{label}:").pack(side=tk.LEFT, padx=5)

        self._var = tk.StringVar(value=values[0])
        self._combo = ttk.Combobox(
            self.frame, textvariable=self._var, width=40, state="readonly"
        )
        self._combo["values"] = values
        self._combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(self.frame, text=button_text, command=self._trigger).pack(
            side=tk.LEFT, padx=5
        )

    def _trigger(self):
        self._on_select(self._var.get())

    def set_values(self, values: list):
        self._combo["values"] = values
        if values:
            self._var.set(values[0])

    def get(self) -> str:
        return self._var.get()


class ValueSlider(_PackGridMixin):
    """Labeled slider with a synced numeric entry field.

    The slider and entry stay in sync: dragging the slider updates the entry,
    editing the entry updates the slider. Calls on_change with the new int value.

    Usage:
        slider = ValueSlider(parent, label="Step", from_=0, to=100,
                             on_change=my_callback)
        slider.set_range(0, 50)
        slider.set(25)
        current = slider.get()
    """

    def __init__(
        self,
        parent,
        label: str,
        from_: int = 0,
        to: int = 100,
        on_change: Callable[[int], None] = lambda v: None,
    ):
        self.frame = ttk.Frame(parent)
        self._on_change = on_change
        self._var = tk.IntVar(value=from_)

        ttk.Label(self.frame, text=f"{label}:").pack(side=tk.LEFT, padx=5)

        self._scale = ttk.Scale(
            self.frame,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=self._var,
            command=self._on_scale,
        )
        self._scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self._entry = ttk.Entry(self.frame, textvariable=self._var, width=8)
        self._entry.pack(side=tk.LEFT, padx=5)
        self._entry.bind("<Return>", self._on_entry)
        self._entry.bind("<FocusOut>", self._on_entry)

    def _on_scale(self, raw_value):
        value = int(float(raw_value))
        self._var.set(value)
        self._on_change(value)

    def _on_entry(self, event=None):
        value = int(float(self._var.get()))
        low = int(float(self._scale.cget("from")))
        high = int(float(self._scale.cget("to")))
        value = max(low, min(high, value))
        self._var.set(value)
        self._on_change(value)

    def set_range(self, from_: int, to: int):
        """Update slider range."""
        self._scale.configure(from_=from_, to=to)

    def set(self, value: int):
        """Set current value."""
        self._var.set(value)

    def get(self) -> int:
        """Get current value."""
        return self._var.get()


class ActionBar(_PackGridMixin):
    """Bottom action bar with optional text inputs and buttons.

    Usage:
        bar = ActionBar(parent,
            inputs=[("Save As", "default.yaml")],
            buttons=[("Save", save_cb), ("Run", run_cb), ("Cancel", close_cb)],
        )
        bar.get_input("Save As")  # returns current text value
    """

    def __init__(
        self,
        parent,
        buttons: list[tuple[str, Callable]],
        inputs: Optional[list[tuple[str, str]]] = None,
    ):
        self.frame = ttk.Frame(parent, padding=10)
        self._inputs = {}

        if inputs:
            for label, default in inputs:
                ttk.Label(self.frame, text=f"{label}:").pack(side=tk.LEFT, padx=5)
                var = tk.StringVar(value=default)
                ttk.Entry(self.frame, textvariable=var, width=40).pack(
                    side=tk.LEFT, padx=5
                )
                self._inputs[label] = var

        for label, callback in buttons:
            ttk.Button(self.frame, text=label, command=callback).pack(
                side=tk.LEFT, padx=5
            )

    def get_input(self, label: str) -> str:
        return self._inputs[label].get()


class StatusBar:
    """Sunken label at the bottom of a window."""

    def __init__(self, parent, initial_text: str = "Ready."):
        self._var = tk.StringVar(value=initial_text)
        self._label = ttk.Label(
            parent, textvariable=self._var, relief=tk.SUNKEN, anchor=tk.W
        )

    def set(self, text: str):
        self._var.set(text)

    def pack(self, **kwargs):
        self._label.pack(**kwargs)

    def grid(self, **kwargs):
        self._label.grid(**kwargs)


# =============================================================================
# Window launcher
# =============================================================================


def launch_window(window_class, *args, **kwargs):
    """Create a Tk root, instantiate window_class(root, *args, **kwargs),
    run mainloop, destroy, and return window.result."""
    root = tk.Tk()
    root.tk.call("tk", "scaling", 1.0)
    window = window_class(root, *args, **kwargs)
    root.mainloop()
    root.destroy()
    return window.result


# =============================================================================
# Widget creation (config parameter widgets)
# =============================================================================


def create_widget(
    parent,
    key: str,
    value: Any,
    spec: Optional[list],
    on_change: Callable[[Any], None],
) -> Tuple[Any, Callable[[Any], None]]:
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

    # Locked -> disabled Entry (no @ui annotation in config)
    if value_type == "locked":
        var = tk.StringVar(value=str(value))
        widget = tk.Entry(
            parent,
            textvariable=var,
            width=ENTRY_WIDTH,
            font=font,
            state="disabled",
            disabledforeground="gray",
        )
        return widget, lambda v: var.set(str(v))

    # Boolean -> Checkbutton with custom indicator
    if value_type == "bool":
        var = tk.BooleanVar(value=bool(value))

        frame = tk.Frame(parent)

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

        frame = tk.Frame(parent)

        entry = tk.Entry(frame, textvariable=var, width=SPINBOX_WIDTH, font=font)
        entry.pack(side=tk.LEFT)

        button_frame = tk.Frame(frame)
        button_frame.pack(side=tk.LEFT)

        def _set_clamped(val):
            clamped = max(min_val, min(max_val, val))
            if value_type == "int":
                clamped = int(clamped)
            var.set(clamped)
            on_change(clamped)

        def validate_and_clamp():
            try:
                _set_clamped(float(entry.get()))
            except ValueError:
                var.set(value)

        tk.Button(
            button_frame,
            text="▲",
            command=lambda: _set_clamped(float(entry.get()) + increment),
            width=2,
            font=(FONT_FAMILY, FONT_SIZE),
        ).pack(side=tk.TOP)

        tk.Button(
            button_frame,
            text="▼",
            command=lambda: _set_clamped(float(entry.get()) - increment),
            width=2,
            font=(FONT_FAMILY, FONT_SIZE),
        ).pack(side=tk.TOP)

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


def ask_yes_no(title, message):
    """Prompt user with a yes/no dialog."""
    return messagebox.askyesno(title, message)


class PlaybackControl(_PackGridMixin):
    """ValueSlider with a play/pause button for stepped playback.

    Calls on_change with each new value, same as ValueSlider.
    get_interval returns milliseconds between ticks (caller controls speed).

    Usage:
        ctrl = PlaybackControl(parent, label="Step", from_=0, to=100,
                               on_change=render, get_interval=lambda: 700)
        ctrl.set_range(0, 50)
        ctrl.set(0)
    """

    def __init__(
        self,
        parent,
        label: str,
        from_: int = 0,
        to: int = 100,
        on_change: Callable[[int], None] = lambda v: None,
        get_interval: Callable[[], int] = lambda: 500,
    ):
        self.frame = ttk.Frame(parent)
        self._get_interval = get_interval
        self._after_id = None

        self._btn = ttk.Button(self.frame, text="Play", width=5, command=self._toggle)
        self._btn.pack(side=tk.LEFT, padx=(0, 5))

        self._slider = ValueSlider(
            self.frame, label=label, from_=from_, to=to, on_change=on_change
        )
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _toggle(self):
        if self._after_id is not None:
            self._stop()
        else:
            self._play()

    def _play(self):
        self._btn.configure(text="Stop")
        self._tick()

    def _stop(self):
        self.frame.after_cancel(self._after_id)
        self._after_id = None
        self._btn.configure(text="Play")

    def _tick(self):
        value = self._slider.get() + 1
        high = int(float(self._slider._scale.cget("to")))
        if value > high:
            self._stop()
            return
        self._slider.set(value)
        self._slider._on_change(value)
        self._after_id = self.frame.after(self._get_interval(), self._tick)

    def set_range(self, from_: int, to: int):
        self._slider.set_range(from_, to)

    def set(self, value: int):
        self._slider.set(value)

    def get(self) -> int:
        return self._slider.get()
