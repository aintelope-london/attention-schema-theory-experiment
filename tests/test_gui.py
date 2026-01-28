# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

"""Unit tests for GUI components: ui_schema_manager and widgets."""

import os
import tkinter as tk

import pytest

from aintelope.gui.ui_schema_manager import load_ui_schema, get_field_spec
from aintelope.gui.widgets import create_widget, get_range_display


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def ui_schema():
    """Load UI schema once for all tests."""
    return load_ui_schema()


@pytest.fixture(scope="module")
def tk_root():
    """Create hidden Tk root for widget tests."""
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


# =============================================================================
# Field Spec Resolution Tests
# =============================================================================


class TestGetFieldSpec:
    """Tests for get_field_spec path navigation."""

    def test_top_level_key(self, ui_schema):
        """Top-level key returns correct spec."""
        spec = get_field_spec(ui_schema, "experiment_name")
        assert spec == [None, "str"]

    def test_nested_key(self, ui_schema):
        """Nested key (one level) returns correct spec."""
        spec = get_field_spec(ui_schema, "trainer_params.num_workers")
        assert spec == [[1, 16], "int"]

    def test_deep_nested_key(self, ui_schema):
        """Deeply nested key returns correct spec."""
        spec = get_field_spec(ui_schema, "hparams.env_params.scores.GOLD_SCORE")
        assert spec == [None, "str"]

    def test_missing_key_returns_none(self, ui_schema):
        """Non-existent path returns None."""
        spec = get_field_spec(ui_schema, "nonexistent.path.here")
        assert spec is None

    def test_intermediate_node_returns_none(self, ui_schema):
        """Intermediate node (dict, not leaf) returns None."""
        spec = get_field_spec(ui_schema, "hparams.env_params")
        assert spec is None


# =============================================================================
# Range Display Formatting Tests
# =============================================================================


class TestGetRangeDisplay:
    """Tests for get_range_display formatting."""

    def test_numeric_range(self):
        """Numeric range formats as 'Range: min - max'."""
        spec = [[0, 100], "int"]
        assert get_range_display(spec) == "Range: 0 - 100"

    def test_string_choices(self):
        """String choices format as 'Choices: a, b, c'."""
        spec = [["alpha", "beta", "gamma"], "str"]
        assert get_range_display(spec) == "Choices: alpha, beta, gamma"

    def test_null_range_returns_empty(self):
        """Null range (bool/free text) returns empty string."""
        spec = [None, "bool"]
        assert get_range_display(spec) == ""

    def test_no_spec_returns_empty(self):
        """None spec returns empty string."""
        assert get_range_display(None) == ""


# =============================================================================
# Widget Type Selection Tests
# =============================================================================


class TestCreateWidget:
    """Tests for create_widget factory returning correct widget types."""

    def test_bool_spec_returns_frame(self, tk_root):
        """Bool spec returns Frame (custom checkbox container)."""
        parent = tk.Frame(tk_root)
        spec = [None, "bool"]
        widget, refresh = create_widget(parent, "test_bool", True, spec, lambda v: None)
        assert isinstance(widget, tk.Frame)

    def test_string_choices_returns_combobox(self, tk_root):
        """String choices spec returns Combobox."""
        from tkinter import ttk

        parent = tk.Frame(tk_root)
        spec = [["option_a", "option_b"], "str"]
        widget, refresh = create_widget(
            parent, "test_choices", "option_a", spec, lambda v: None
        )
        assert isinstance(widget, ttk.Combobox)

    def test_int_range_returns_frame(self, tk_root):
        """Int range spec returns Frame (custom spinbox container)."""
        parent = tk.Frame(tk_root)
        spec = [[0, 100], "int"]
        widget, refresh = create_widget(parent, "test_int", 50, spec, lambda v: None)
        assert isinstance(widget, tk.Frame)

    def test_float_range_returns_frame(self, tk_root):
        """Float range spec returns Frame (custom spinbox container)."""
        parent = tk.Frame(tk_root)
        spec = [[0.0, 1.0], "float"]
        widget, refresh = create_widget(parent, "test_float", 0.5, spec, lambda v: None)
        assert isinstance(widget, tk.Frame)

    def test_no_spec_returns_entry(self, tk_root):
        """No spec (None) returns Entry as default."""
        parent = tk.Frame(tk_root)
        widget, refresh = create_widget(
            parent, "test_default", "some text", None, lambda v: None
        )
        assert isinstance(widget, tk.Entry)


if __name__ == "__main__":  # and os.name == "nt":
    pytest.main([__file__])
