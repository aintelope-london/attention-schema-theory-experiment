# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""AST tripwire — forbids hash-randomized numerics from entering the codebase.

Without PYTHONHASHSEED set, Python randomizes hashes of str/bytes/frozenset per
run. Iteration over set/frozenset reflects that randomization, and hash() of
those types returns a different value each run. If numeric output depends on
either, runs are not bit-reproducible.

The codebase is held free of such patterns. This test walks each .py file under
`aintelope/` and `tests/` and fails if it finds any. On failure, the author is
asked to lock determinism down properly (see DOCUMENTATION.md § Determinism
invariants) — the tripwire exists so that the day it is needed is the day it
fires, not silently later.

Patterns caught:
    hash(...)                           - any call to the built-in hash
    for x in {...}                      - iteration over set literal
    for x in (set|frozenset)(...)       - iteration over set/frozenset constructor
    for x in name   where name = set(...) or {...}  at module/function top level

Known limitation: iteration over attribute-held sets (`self._s = set(); for x in self._s`)
is not caught statically. The smoke test catches resulting numeric drift.
"""

import ast
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCAN_ROOTS = ("aintelope", "tests")

_MESSAGE = (
    "breaks PYTHONHASHSEED determinism, it is time.\n"
    "See DOCUMENTATION.md § Determinism invariants."
)


class _Finder(ast.NodeVisitor):
    """Collects (lineno, kind) for every forbidden pattern found."""

    def __init__(self):
        self.hits = []
        self._set_names = []  # list, not set — avoids self-trigger

    def _track_assign(self, node):
        value = node.value
        if not (
            isinstance(value, (ast.Set, ast.SetComp))
            or (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in ("set", "frozenset")
            )
        ):
            return
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                self._set_names.append(target.id)

    def visit_Assign(self, node):
        self._track_assign(node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._track_assign(node)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "hash":
            self.hits.append((node.lineno, "hash() call"))
        self.generic_visit(node)

    def visit_For(self, node):
        kind = self._classify_iter(node.iter)
        if kind:
            self.hits.append((node.lineno, kind))
        self.generic_visit(node)

    def _classify_iter(self, iter_node):
        if isinstance(iter_node, (ast.Set, ast.SetComp)):
            return "iteration over set literal"
        if isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name):
            if iter_node.func.id in ("set", "frozenset"):
                return f"iteration over {iter_node.func.id}() constructor"
        if isinstance(iter_node, ast.Name) and iter_node.id in self._set_names:
            return f"iteration over set-valued name '{iter_node.id}'"
        return None


def _scan(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    finder = _Finder()
    finder.visit(tree)
    return finder.hits


def _files():
    for root in _SCAN_ROOTS:
        yield from (_REPO_ROOT / root).rglob("*.py")


def test_no_hash_dependent_numerics():
    violations = []
    for path in _files():
        for lineno, kind in _scan(path):
            violations.append(f"  {path.relative_to(_REPO_ROOT)}:{lineno}  {kind}")
    if violations:
        pytest.fail(f"{_MESSAGE}\n\n" + "\n".join(violations), pytrace=False)
