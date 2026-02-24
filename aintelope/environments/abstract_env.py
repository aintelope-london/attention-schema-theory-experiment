"""Abstract environment contract.

Minimal multi-agent MDP interface. The orchestration layer
(experiments.py) depends only on this contract, never on concrete
environment internals.

All methods exchange dicts keyed by agent_id. The environment
does not dictate how many agents exist — that comes from config.
"""

from abc import ABC, abstractmethod


class AbstractEnv(ABC):
    @abstractmethod
    def reset(self, **kwargs):
        """Reset the environment.

        Returns:
            observations: {agent_id: observation}
            infos:        {agent_id: dict}
        """

    @abstractmethod
    def step_parallel(self, actions):
        """All agents act simultaneously.

        Args:
            actions: {agent_id: action}

        Returns:
            observations, rewards, terminateds, truncateds, infos
        """

    @abstractmethod
    def step_sequential(self, actions):
        """Agents act one at a time, observing intermediate effects.

        Args:
            actions: {agent_id: action}

        Returns:
            observations, rewards, terminateds, truncateds, infos
        """

    @property
    @abstractmethod
    def manifesto(self):
        """Environment manifesto. Built on reset().

        Returns:
            dict with at minimum:
                observation_shapes: {field_name: shape_tuple}
                action_space: list of action indices
        """

    @abstractmethod
    def board_state(self):
        """Global board state for logging.

        Returns:
            (board_array, layer_order)
        """

    @property
    @abstractmethod
    def score_dimensions(self):
        """List of score dimension names for event logging."""
