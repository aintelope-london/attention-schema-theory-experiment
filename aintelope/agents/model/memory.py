# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import random
from collections import deque


class ReplayMemory:
    """Generic replay memory that accepts arbitrary dictionary entries."""

    def __init__(self, capacity, fields):
        self.memory = deque([], maxlen=capacity)
        self.fields = fields

    def push(self, **kwargs):
        """Save a transition as dictionary"""
        entry = {field: kwargs.get(field) for field in self.fields}
        if any(v is None for v in entry.values()):
            raise ValueError(
                f"Missing fields in memory: {[k for k, v in entry.items() if v is None]}"
            )
        self.memory.append(entry)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def latest(self):
        return self.memory[-1]

    def __len__(self):
        return len(self.memory)
