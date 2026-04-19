# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import copy
import os
import torch
from aintelope.agents.model.memory import ReplayMemory
from aintelope.agents.model.dl_components import NeuralNet, DQN, ModelBased
from aintelope.agents.model.reward_inference import RewardInference
from aintelope.agents.model.roi import ROI


def expand_keywords(entries, obs_fields):
    """Expand 'observation'/'next_observation' keywords to modality names from manifesto."""
    expanded = []
    for entry in entries:
        if entry == "observation":
            expanded.extend(obs_fields)
        elif entry == "next_observation":
            expanded.extend("next_" + f for f in obs_fields)
        else:
            expanded.append(entry)
    return expanded


def fill_plans(obs_shapes, action_space, plans, obs_fields, internal_actions=0):
    """Expand string size references in architecture to actual values from dynamic fields.
    Normalize formats before adding them into the plans.
    """
    lookup = dict(obs_shapes)
    lookup.update({k: v[0] for k, v in lookup.items()})
    lookup["observation"] = "+".join(obs_fields)
    lookup["next_observation"] = "+".join("next_" + f for f in obs_fields)

    plans["n_env_actions"] = len(action_space)
    plans["n_actions"] = len(action_space) + internal_actions
    lookup["action"] = plans["n_actions"]

    if "vision" in obs_shapes:
        plans["vision_size"] = obs_shapes["vision"][1:]

    if "target" in plans.get("metadata", {}):
        plans["metadata"]["target"] = expand_keywords(
            plans["metadata"]["target"], obs_fields
        )

    if "architecture" in plans.keys():
        if "vision_net" in plans["architecture"] and "vision" in obs_shapes:
            channels, height, width = obs_shapes["vision"]
            for layer in plans["architecture"]["vision_net"]:
                if layer["type"] == "conv":
                    kernel = layer["kernel"]
                    height = height - kernel + 1
                    width = width - kernel + 1
                    channels = layer["size"]
            plans["vision_encoded_shape"] = [height, width, channels]
            lookup["vision_encoded"] = height * width * channels

        for plan_name, plan in plans["architecture"].items():
            for layer in plan:
                if "source" in layer.keys():
                    source = layer["source"]
                    if "+" in source:
                        layer["size"] = sum(lookup[src] for src in source.split("+"))
                    else:
                        layer["size"] = lookup[source]
                if layer["type"] == "unflatten":
                    shape = plans["vision_encoded_shape"]
                    layer["size"] = [shape[0], shape[1], shape[2]]
                if "size" in layer.keys():
                    last_size = layer["size"]
            lookup[plan_name] = last_size


def instantiate_components(
    architecture, cfg, env_manifesto, agent_id, activations, components, memory, device
):
    """Populate the `components` dict with instantiated entries from `architecture`.

    Shared between Model (full training connectome) and DummyAgent (probe-driven
    components with pre-trained weights). Factory-based: each entry's `type`
    maps to a class via the -NN suffix or direct name lookup, and its plans
    come from cfg.models if a library card exists, else from the entry itself.

    The caller owns `activations`, `components`, and `memory`; they are shared
    by reference with each component so siblings can reach each other.
    """
    obs_shapes = env_manifesto["observation_shapes"]
    action_space = env_manifesto["action_space"]
    obs_fields = list(obs_shapes.keys())

    # internal_actions: sum of non-env action slots declared by architecture entries.
    # Each entry may declare n_internal_actions in config (e.g. roi: n_internal_actions: 3).
    # This is read generically here — no component names are mentioned.
    internal_actions = sum(
        entry.get("n_internal_actions", 0) for entry in architecture.values()
    )

    context = {
        "cfg": cfg,
        "device": device,
        "components": components,
        "memory": memory,
        "env_manifesto": env_manifesto,
        "agent_id": agent_id,
        "activations": activations,
    }

    for component_id, entry in architecture.items():
        module_type = "NeuralNet" if "-NN" in entry.type else entry.type
        module_cls = globals()[module_type]

        # Components with a library card in cfg.models get fill_plans.
        # Components without one receive the architecture entry as plans so
        # they can self-parameterise directly from their own config fields.
        if entry.type in cfg.models:
            plans = copy.deepcopy(cfg.models[entry.type])
            fill_plans(obs_shapes, action_space, plans, obs_fields, internal_actions)
        else:
            plans = entry

        context["plans"] = plans
        context["component_id"] = component_id
        context["inputs"] = expand_keywords(list(entry.inputs), obs_fields)

        components[component_id] = module_cls(context)


class Model:
    """
    Brains of a single agent. Component-based connectome.
    Interfaces to the get_action and update functionalities.
    Uses a factory pattern to instantiate components from config.
    """

    def __init__(self, agent_id, env_manifesto, cfg, checkpoint=None):
        self.cfg = cfg
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_shapes = env_manifesto["observation_shapes"]
        self.components = {}
        self.activations = {}
        self.resets = 0

        architecture = cfg.agent_params.agents[agent_id].get("architecture") or {}
        obs_fields = list(obs_shapes.keys())

        # Inputs that are not component ids and not observation keywords are
        # extra activation keys written by strategy components (e.g. "internal_action").
        # These must be tracked in memory for training.
        all_inputs_flat = {inp for e in architecture.values() for inp in e.inputs}
        expand_kw = {"observation", "next_observation"}
        internal_activation_keys = list(
            all_inputs_flat - set(architecture.keys()) - set(obs_fields) - expand_kw
        )

        root_component_ids = [cid for cid in architecture if cid not in all_inputs_flat]

        memory_field_list = (
            ["done"]
            + obs_fields
            + ["next_" + f for f in obs_fields]
            + root_component_ids
            + internal_activation_keys
        )
        self.memory = ReplayMemory(
            cfg.agent_params.replay_buffer_size, memory_field_list
        )

        # Output keys returned from get_action:
        #   - root component ids (not consumed by any sibling)
        #   - explicitly declared outputs (consumed by siblings but also needed externally)
        output_keys = {
            k for e in architecture.values() for k in getattr(e, "outputs", [])
        }
        self._output_keys = (
            set(root_component_ids) | output_keys | set(internal_activation_keys)
        )

        instantiate_components(
            architecture=architecture,
            cfg=cfg,
            env_manifesto=env_manifesto,
            agent_id=agent_id,
            activations=self.activations,
            components=self.components,
            memory=self.memory,
            device=self.device,
        )

        if checkpoint:
            self._load_checkpoint(checkpoint)

    def reset(self):
        """Episode boundary reset -- clears all activations and propagates to components."""
        self.activations.clear()
        for component in self.components.values():
            component.reset()
        self.resets += 1

    def _load_checkpoint(self, path):
        """Load a bundled checkpoint and distribute to components."""
        data = torch.load(path, map_location=self.device)
        for component_id, state_data in data.items():
            if component_id in self.components and hasattr(
                self.components[component_id], "load_checkpoint_data"
            ):
                self.components[component_id].load_checkpoint_data(state_data)

    def get_action(self, observation):
        self.activations.update(observation)
        self.activations["internal_episode"] = self.resets
        self.components["action"].activate(self.activations)
        return {
            k: self.activations[k] for k in self._output_keys if k in self.activations
        }

    def update(self, next_observation, done=False):
        for field, value in next_observation.items():
            self.activations[f"next_{field}"] = value
        self.activations["done"] = float(done)
        self.components["reward"].activate(self.activations)
        self.memory.push(**self.activations)
        signals = {}
        report = self.components["action"].update(signals)
        # Partial clear: remove transient fields only. Action outputs (e.g.
        # "internal_action") persist so stateful components can read them next step.
        for key in [
            k for k in self.activations if k.startswith("next_") or k == "done"
        ]:
            del self.activations[key]
        report["step_reward"] = self.activations.get("reward", 0)
        return report

    def save(self, path):
        """Save all components into a single bundled file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            cid: c.checkpoint_data()
            for cid, c in self.components.items()
            if hasattr(c, "checkpoint_data")
        }
        torch.save(data, path)