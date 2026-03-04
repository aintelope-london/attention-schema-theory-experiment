import copy
import os
import torch
from aintelope.agents.model.memory import ReplayMemory
from aintelope.agents.model.dl_components import NeuralNet, DQN, ModelBased
from aintelope.agents.model.reward_inference import RewardInference


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
        action_space = env_manifesto["action_space"]
        self.components = {}
        self.activations = {}

        architecture = cfg.agent_params[agent_id].architecture
        obs_fields = list(obs_shapes.keys())

        all_inputs = {inp for entry in architecture.values() for inp in entry.inputs}
        memory_field_list = (
            ["done"]
            + obs_fields
            + ["next_" + field for field in obs_fields]
            + [cid for cid in architecture.keys() if cid not in all_inputs]
        )
        self.memory = ReplayMemory(cfg.agent_params.batch_size, memory_field_list)

        context = {
            "cfg": self.cfg,
            "device": self.device,
            "components": self.components,
            "memory": self.memory,
            "env_manifesto": env_manifesto,
            "agent_id": agent_id,
            "activations": self.activations,
        }

        for component_id, entry in architecture.items():
            plans = copy.deepcopy(cfg.models[entry.type])
            self.fill_plans(obs_shapes, action_space, plans, obs_fields)
            context["plans"] = plans
            context["component_id"] = component_id
            context["inputs"] = expand_keywords(list(entry.inputs), obs_fields)

            module_type = "NeuralNet" if "-NN" in entry.type else entry.type
            module_cls = globals()[module_type]
            self.components[component_id] = module_cls(context)

        if checkpoint:
            self._load_checkpoint(checkpoint)

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
        self.components["action"].activate(self.activations)
        return self.activations["action"]

    def update(self, next_observation, done=False):
        for field, value in next_observation.items():
            self.activations[f"next_{field}"] = value
        self.activations["done"] = float(done)
        self.components["reward"].activate(self.activations)
        self.memory.push(**self.activations)
        signals = {}
        report = self.components["action"].update(signals)
        self.activations.clear()
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

    @staticmethod
    def fill_plans(obs_shapes, action_space, plans, obs_fields):
        """
        Expand string size references in architecture to actual values from dynamic fields.
        Normalize formats before adding them into the plans.
        """
        lookup = dict(obs_shapes)
        lookup.update({k: v[0] for k, v in lookup.items()})
        lookup["action"] = len(action_space)
        lookup["observation"] = "+".join(obs_fields)
        lookup["next_observation"] = "+".join("next_" + f for f in obs_fields)
        plans["n_actions"] = len(action_space)
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
                            layer["size"] = sum(
                                lookup[src] for src in source.split("+")
                            )
                        else:
                            layer["size"] = lookup[source]
                    if layer["type"] == "unflatten":
                        shape = plans["vision_encoded_shape"]
                        layer["size"] = [shape[0], shape[1], shape[2]]
                    if "size" in layer.keys():
                        last_size = layer["size"]
                lookup[plan_name] = last_size
