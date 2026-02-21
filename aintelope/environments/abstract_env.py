"""Abstract environment contract.

All environments expose this interface. The orchestration layer
(experiments.py) depends only on this contract, never on concrete
environment internals.
"""

from abc import ABC, abstractmethod


class AbstractEnv(ABC):
    @abstractmethod
    def reset(self, **kwargs):
        """Reset the environment.

        Returns:
            observations: {agent_id: ndarray}
            infos:        {agent_id: {"position": (r,c), "direction": (dr,dc), ...}}

        Manifesto is built on reset and accessible via the manifesto property.
        """

    @abstractmethod
    def step_parallel(self, actions):
        """All agents act simultaneously.

        Args:
            actions: {agent_id: action}

        Returns:
            observations, scores, terminateds, truncateds, infos
            All dicts keyed by agent_id. scores values are dicts.
        """

    @abstractmethod
    def step_sequential(self, actions):
        """Agents act one at a time, observing intermediate effects.

        Args:
            actions: {agent_id: action}

        Returns:
            observations, scores, terminateds, truncateds, infos
            Same format as step_parallel.
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
        """List of score dimension names."""

    @abstractmethod
    def observation_space(self, agent_id):
        """Gymnasium Space for the given agent."""

    @abstractmethod
    def action_space(self, agent_id):
        """Gymnasium Space for the given agent."""

    @property
    @abstractmethod
    def max_num_agents(self):
        """Maximum number of agents."""

    @property
    @abstractmethod
    def agents(self):
        """List of current agent ids."""
