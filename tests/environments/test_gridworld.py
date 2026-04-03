"""Unit tests for GridworldEnv.

Covers: orientation, movement, wall/agent blocking,
food consumption, predator contact, predator persistence,
interoception reset, manifesto correctness.
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

from aintelope.environments.gridworld import (
    GridworldEnv,
    FLOOR,
    WALL,
    PREDATOR,
    FOOD,
    _N_BASE,
    _N,
    _E,
    _S,
    _W,
    _cw,
    _ccw,
    _flip,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _cfg(map_size=9, objects=None, obs_radius=3):
    objects = objects or {"food": {"count": 0}, "predator": {"count": 0}}
    return OmegaConf.create(
        {
            "run": {"seed": 0},
            "env_params": {
                "map_size": map_size,
                "objects": objects,
                "actions": ["forward", "left", "right", "backward", "wait"],
                "observation_radius": obs_radius,
                "observation_format": "boolean_cube",
            },
            "agent_params": {
                "agents": {
                    "agent_0": {"agent_class": "main_agent"},
                }
            },
        }
    )


def _two_agent_cfg(map_size=9):
    return OmegaConf.create(
        {
            "run": {"seed": 42},
            "env_params": {
                "map_size": map_size,
                "objects": {"food": {"count": 0}, "predator": {"count": 0}},
                "actions": ["forward", "left", "right", "backward", "wait"],
                "observation_radius": 3,
                "observation_format": "boolean_cube",
            },
            "agent_params": {
                "agents": {
                    "agent_0": {"agent_class": "main_agent"},
                    "agent_1": {"agent_class": "main_agent"},
                }
            },
        }
    )


def _make(map_size=9, objects=None, obs_radius=3):
    env = GridworldEnv(_cfg(map_size=map_size, objects=objects, obs_radius=obs_radius))
    obs, state = env.reset()
    return env, obs, state


def _act(env, name):
    idx = env.manifesto["action_space"].index(name)
    return {"agent_0": {"action": idx}}


def _place(env, pos, facing=_N):
    """Place agent_0 at pos with given facing. Board must be cleared first."""
    old = env._positions["agent_0"]
    env._board[old] = FLOOR
    env._positions["agent_0"] = pos
    env._board[pos] = _N_BASE + env.agents.index("agent_0")
    env._facing["agent_0"] = facing


def _pos(env):
    return env._positions["agent_0"]


def _face(env):
    return env._facing["agent_0"]


# ── Orientation ────────────────────────────────────────────────────────────────


class TestOrientation:
    def test_default_facing_north(self):
        env, _, _ = _make()
        assert _face(env) == _N

    def test_left_rotates_ccw(self):
        env, _, _ = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env.step_parallel(_act(env, "left"))
        assert _face(env) == _ccw(_N)

    def test_right_rotates_cw(self):
        env, _, _ = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env.step_parallel(_act(env, "right"))
        assert _face(env) == _cw(_N)

    def test_backward_flips_facing(self):
        env, _, _ = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env.step_parallel(_act(env, "backward"))
        assert _face(env) == _flip(_N)

    def test_four_lefts_full_circle(self):
        env, _, _ = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        for _ in range(4):
            env.step_parallel(_act(env, "left"))
        assert _face(env) == _N

    def test_four_rights_full_circle(self):
        env, _, _ = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        for _ in range(4):
            env.step_parallel(_act(env, "right"))
        assert _face(env) == _N


# ── Movement ───────────────────────────────────────────────────────────────────


class TestMovement:
    def _center(self, facing=_N):
        env, _, _ = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), facing)
        return env

    def test_forward_north(self):
        env = self._center(_N)
        env.step_parallel(_act(env, "forward"))
        assert _pos(env) == (4, 5)

    def test_forward_east(self):
        env = self._center(_E)
        env.step_parallel(_act(env, "forward"))
        assert _pos(env) == (5, 6)

    def test_backward_flips_and_moves(self):
        env = self._center(_N)
        env.step_parallel(_act(env, "backward"))
        assert _pos(env) == (6, 5)
        assert _face(env) == _S

    def test_left_turns_and_moves_west(self):
        env = self._center(_N)
        env.step_parallel(_act(env, "left"))
        assert _pos(env) == (5, 4)
        assert _face(env) == _W

    def test_right_turns_and_moves_east(self):
        env = self._center(_N)
        env.step_parallel(_act(env, "right"))
        assert _pos(env) == (5, 6)
        assert _face(env) == _E

    def test_wait_does_not_move(self):
        env = self._center(_N)
        before = _pos(env)
        env.step_parallel(_act(env, "wait"))
        assert _pos(env) == before

    def test_vacated_cell_becomes_floor(self):
        env = self._center(_N)
        r, c = _pos(env)
        env.step_parallel(_act(env, "forward"))
        assert env._board[r, c] == FLOOR

    def test_new_cell_shows_agent_tile(self):
        env = self._center(_N)
        env.step_parallel(_act(env, "forward"))
        nr, nc = _pos(env)
        assert env._board[nr, nc] == _N_BASE + env.agents.index("agent_0")


# ── Blocking ───────────────────────────────────────────────────────────────────


class TestBlocking:
    def test_wall_blocks_movement(self):
        env, _, _ = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (1, 5), _N)  # adjacent to north wall
        before = _pos(env)
        env.step_parallel(_act(env, "forward"))
        assert _pos(env) == before

    def test_agent_blocks_agent(self):
        env = GridworldEnv(_two_agent_cfg())
        env.reset()
        env._board[1:-1, 1:-1] = FLOOR
        env._positions["agent_0"] = (5, 5)
        env._positions["agent_1"] = (4, 5)
        env._board[5, 5] = _N_BASE + 0
        env._board[4, 5] = _N_BASE + 1
        env._facing["agent_0"] = _N
        env._facing["agent_1"] = _S
        fwd = env.manifesto["action_space"].index("forward")
        wait = env.manifesto["action_space"].index("wait")
        env.step_parallel(
            {
                "agent_0": {"action": fwd},
                "agent_1": {"action": wait},
            }
        )
        assert env._positions["agent_0"] == (5, 5)


# ── Food ───────────────────────────────────────────────────────────────────────


class TestFood:
    def _with_food(self, food_pos):
        env, _, _ = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env._board[food_pos] = FOOD
        return env

    def test_food_interoception_fires(self):
        env = self._with_food((4, 5))
        obs, state = env.step_parallel(_act(env, "forward"))
        assert obs["agent_0"]["interoception"][0] == 1.0

    def test_food_consumed(self):
        env = self._with_food((4, 5))
        env.step_parallel(_act(env, "forward"))
        assert env._board[4, 5] != FOOD

    def test_no_food_interoception_on_empty(self):
        env, _, _ = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        obs, _ = env.step_parallel(_act(env, "forward"))
        assert obs["agent_0"]["interoception"][0] == 0.0

    def test_food_position_in_state(self):
        env = self._with_food((4, 5))
        # state is already live after _with_food sets the board tile;
        # step with wait to get a fresh state without consuming food
        _, state = env.step_parallel(_act(env, "wait"))
        assert state["food_position"] == (4, 5)

    def test_food_gone_from_state_after_consumed(self):
        env = self._with_food((4, 5))
        env.step_parallel(_act(env, "forward"))
        assert env.state["food_position"] is None


# ── Predator ───────────────────────────────────────────────────────────────────


class TestPredator:
    def _with_predator(self, pred_pos):
        env, _, _ = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env._board[pred_pos] = PREDATOR
        env._predator_cells.add(pred_pos)
        return env

    def test_predator_interoception_fires(self):
        env = self._with_predator((4, 5))
        obs, _ = env.step_parallel(_act(env, "forward"))
        assert obs["agent_0"]["interoception"][1] == 1.0

    def test_predator_persists_after_contact(self):
        env = self._with_predator((4, 5))
        env.step_parallel(_act(env, "forward"))  # step onto predator
        env.step_parallel(_act(env, "forward"))  # move away
        assert env._board[4, 5] == PREDATOR

    def test_predator_no_food_interoception(self):
        env = self._with_predator((4, 5))
        obs, _ = env.step_parallel(_act(env, "forward"))
        assert obs["agent_0"]["interoception"][0] == 0.0


# ── Termination ───────────────────────────────────────────────────────────────


class TestTermination:
    def _cfg_with_termination(self, termination):
        return OmegaConf.create(
            {
                "run": {"seed": 0},
                "env_params": {
                    "map_size": 9,
                    "objects": {"food": {"count": 0}, "predator": {"count": 0}},
                    "actions": ["forward", "left", "right", "backward", "wait"],
                    "observation_radius": 3,
                    "observation_format": "boolean_cube",
                    "termination": termination,
                },
                "agent_params": {"agents": {"agent_0": {"agent_class": "main_agent"}}},
            }
        )

    def test_done_on_food_when_termination_food(self):
        env = GridworldEnv(self._cfg_with_termination("food"))
        env.reset()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env._board[4, 5] = FOOD
        _, state = env.step_parallel(_act(env, "forward"))
        assert state["dones"]["agent_0"] is True

    def test_not_done_on_empty_move_when_termination_food(self):
        env = GridworldEnv(self._cfg_with_termination("food"))
        env.reset()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        _, state = env.step_parallel(_act(env, "forward"))
        assert state["dones"]["agent_0"] is False

    def test_not_done_on_food_when_no_termination(self):
        env = GridworldEnv(self._cfg_with_termination(None))
        env.reset()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env._board[4, 5] = FOOD
        _, state = env.step_parallel(_act(env, "forward"))
        assert state["dones"]["agent_0"] is False

    def test_not_done_on_predator_when_termination_food(self):
        env = GridworldEnv(self._cfg_with_termination("food"))
        env.reset()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env._board[4, 5] = PREDATOR
        env._predator_cells.add((4, 5))
        _, state = env.step_parallel(_act(env, "forward"))
        assert state["dones"]["agent_0"] is False


class TestInteroceptionReset:
    def test_resets_to_zero_next_step(self):
        env, _, _ = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env._board[4, 5] = FOOD
        obs, _ = env.step_parallel(_act(env, "forward"))
        assert obs["agent_0"]["interoception"][0] == 1.0
        obs, _ = env.step_parallel(_act(env, "forward"))
        assert obs["agent_0"]["interoception"][0] == 0.0


# ── State ──────────────────────────────────────────────────────────────────────


class TestState:
    def test_agent_positions_in_state(self):
        env, _, state = _make()
        assert "agent_0" in state["agent_positions"]

    def test_board_shape(self):
        env, _, state = _make(map_size=9)
        assert state["board"].shape == (len(env.layers), 11, 11)

    def test_layers_match_board_channels(self):
        env, _, state = _make()
        assert len(state["layers"]) == state["board"].shape[0]

    def test_dones_false_at_start(self):
        env, _, state = _make()
        assert all(not v for v in state["dones"].values())


# ── Manifesto ─────────────────────────────────────────────────────────────────


class TestManifesto:
    def test_food_ind_matches_layers(self):
        env, _, _ = _make()
        assert env.manifesto["food_ind"] == env.layers.index("food")

    def test_action_names(self):
        env, _, _ = _make()
        assert list(env.manifesto["action_names"].values()) == [
            "forward",
            "left",
            "right",
            "backward",
            "wait",
        ]

    def test_observation_shapes(self):
        env, obs, _ = _make(obs_radius=3)
        v = 2 * 3 + 1
        assert env.manifesto["observation_shapes"]["vision"] == (len(env.layers), v, v)
        assert (
            obs["agent_0"]["vision"].shape
            == env.manifesto["observation_shapes"]["vision"]
        )

    def test_agent_layers_in_layers(self):
        env, _, _ = _make()
        assert "agent_0" in env.layers

    def test_per_agent_layers_two_agents(self):
        env = GridworldEnv(_two_agent_cfg())
        env.reset()
        assert "agent_0" in env.layers
        assert "agent_1" in env.layers
