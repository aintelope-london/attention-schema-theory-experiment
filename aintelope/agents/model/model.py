import copy
import os
import torch
from aintelope.agents.model.memory import ReplayMemory
from aintelope.agents.model.dl_components import NeuralNet, DQN, ModelBased
from aintelope.agents.model.reward_inference import RewardInference


class Model:
    """
    Brains of a single agent. Handles PyTorch, affective and other ML components.
    Interfaces to the get_action and update -functionalities.
    Uses a factory pattern to instantiate the components from config.
    """

    def __init__(self, agent_id, env_manifesto, cfg, checkpoint=None):
        self.cfg = cfg
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_shapes = env_manifesto["observation_shapes"]
        action_space = env_manifesto["action_space"]
        self.components = {}
        self.state = {}

        assert (
            "action" in cfg.agent_params[agent_id].architecture
        ), "Architecture must include 'action' role"
        assert (
            "reward" in cfg.agent_params[agent_id].architecture
        ), "Architecture must include 'reward' role"

        memory_field_list = (
            list(obs_shapes.keys())
            + ["next_" + field for field in obs_shapes.keys()]
            + [role for role, _ in cfg.agent_params[agent_id].architecture.items()]
        )
        self.memory = ReplayMemory(cfg.agent_params.batch_size, memory_field_list)

        context = {
            "cfg": self.cfg,
            "device": self.device,
            "components": self.components,
            "memory": self.memory,
            "env_manifesto": env_manifesto,
            "agent_id": agent_id,
        }

        for module_role, module_name in cfg.agent_params[agent_id].architecture.items():
            plans = copy.deepcopy(cfg.models[module_name])
            self.fill_plans(obs_shapes, action_space, plans)
            context["plans"] = plans
            context["role"] = module_role
            module_type = "NeuralNet" if "-NN" in module_name else module_name
            module_cls = globals()[module_type]

            context["checkpoint"] = None
            self.components[module_role] = module_cls(context)

        # Load checkpoint after all components are built (bundled format)
        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path):
        """Load a bundled checkpoint and distribute to components by role."""
        data = torch.load(path, map_location=self.device)
        for role, state_data in data.items():
            if role in self.components and hasattr(
                self.components[role], "load_checkpoint_data"
            ):
                self.components[role].load_checkpoint_data(state_data)

    def get_action(self, observation):
        self.state = dict(observation)
        for role in self.components:
            if "post_activate" not in self.components[role].metadata:
                self.state[role] = self.components[role].activate(observation)[
                    0
                ]  # TODO: handle confidence later, [1]

        return self.state["action"]

    def update(self, next_observation):
        for field in next_observation.keys():
            self.state[f"next_{field}"] = next_observation[field]
        for role in self.components:
            if "post_activate" in self.components[role].metadata:
                self.state[role] = self.components[role].activate(self.state)[
                    0
                ]  # TODO: handle confidence later, [1]

        self.memory.push(**self.state)
        reports = []
        for role in self.components:
            report = self.components[role].update()
            reports.append(report)
        return reports

    def save(self, path):
        """Save all components with checkpoint data into a single bundled file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {}
        for role, component in self.components.items():
            if hasattr(component, "checkpoint_data"):
                data[role] = component.checkpoint_data()
        torch.save(data, path)

    @staticmethod
    def fill_plans(obs_shapes, action_space, plans):
        """
        Expand string size references in architecture to actual values from dynamic fields.
        Normalize formats before adding them into the plans.
        """
        obs_fields = list(obs_shapes.keys())
        lookup = dict(obs_shapes)
        lookup.update({k: v[0] for k, v in lookup.items()})
        lookup["action"] = len(action_space)
        lookup["observation"] = "+".join(obs_fields)
        lookup["next_observation"] = "+".join(["next_" + f for f in obs_fields])
        plans["n_actions"] = len(action_space)
        if "vision" in obs_shapes:
            plans["vision_size"] = obs_shapes["vision"][1:]

        if "target" in plans.get("metadata", {}):
            expanded_targets = []
            for target_spec in plans["metadata"]["target"]:
                if target_spec == "observation":
                    expanded_targets.extend(obs_fields)
                elif target_spec == "next_observation":
                    expanded_targets.extend(["next_" + f for f in obs_fields])
                else:
                    expanded_targets.append(target_spec)
            plans["metadata"]["target"] = expanded_targets

        if "architecture" in plans.keys():
            # Pre-calculate vision encoding path if it exists
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


# ------- Everything below going to different classes later on
