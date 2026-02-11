"""Progress reporting for experiment execution."""

from typing import Callable, Optional


class ProgressReporter:
    def __init__(
        self,
        levels: list[str],
        on_update: Optional[Callable[["ProgressReporter"], None]] = None,
    ):
        self.levels = levels
        self.state = {level: {"current": 0, "total": 0} for level in levels}
        self._on_update = on_update

    def set_total(self, level: str, total: int):
        self.state[level]["total"] = total
        self._notify()

    def update(self, level: str, current: int):
        self.state[level]["current"] = current
        # Reset all levels below
        reset = False
        for lev in self.levels:
            if reset:
                self.state[lev]["current"] = 0
            if lev == level:
                reset = True
        self._notify()

    def _notify(self):
        if self._on_update:
            self._on_update(self)
