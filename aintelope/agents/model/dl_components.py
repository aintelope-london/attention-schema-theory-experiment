# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import torch
from torch import nn
from aintelope.agents.model.component import Component


class NeuralNet(nn.Module, Component):
    """PyTorch component."""

    def __init__(self, context):
        super().__init__()
        self.cfg = context["cfg"]
        self.device = context["device"]
        self.component_id = context["component_id"]
        self.inputs = context["inputs"]
        self.components = context["components"]
        self.activations = context["activations"]
        self.memory = context["memory"]
        self.plans = context["plans"]
        self.metadata = context["plans"]["metadata"]

        self.component_inputs = [i for i in self.inputs if i in self.components]

        self.net = Network(self.plans).to(self.device)

        if self.metadata.get("use_target_net", False):
            self.target_net = Network(self.plans).to(self.device)
            self.target_net.load_state_dict(self.net.state_dict())
        else:
            self.target_net = None

        optimizer_cls = getattr(torch.optim, self.metadata["optimizer"])
        self.optimizer = optimizer_cls(
            self.net.parameters(), **self.metadata["optimizer_params"]
        )
        self.loss_fn = getattr(nn, self.metadata["loss_function"])()
        self.latest_loss = None

    def activate(self, activations):
        for inp in self.component_inputs:
            self.components[inp].activate(activations)

        input_dict = {
            field: np.expand_dims(activations[field], axis=0)
            for field in self.net.inputs
        }
        output = self.net(input_dict)
        activations[self.component_id] = {
            k: v[0].detach().cpu().numpy() for k, v in output.items()
        }

    def update(self, signals=None):
        return self.optimize(signals or {})

    def optimize(self, signals):
        if len(self.memory) < self.cfg.agent_params.batch_size:
            return {}

        batch = self.memory.sample(self.cfg.agent_params.batch_size)
        custom_loss_fn = signals.get(self.component_id)

        if custom_loss_fn is not None:
            tensors = {
                field: torch.tensor(
                    np.stack(
                        [
                            np.array([entry[field]])
                            if np.isscalar(entry[field])
                            else np.atleast_1d(entry[field])
                            for entry in batch
                        ]
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                for field in self.memory.fields
            }
            loss = custom_loss_fn(self.net, self.target_net, tensors)
        else:
            tensors = {
                field: np.stack([entry[field] for entry in batch])
                for field in self.net.inputs
            }
            output = self.net(tensors)
            loss = torch.tensor(0.0, device=self.device)
            for (_, output_tensor), target_field in zip(
                output.items(), self.metadata["target"]
            ):
                target_data = np.stack([entry[target_field] for entry in batch])
                target = torch.tensor(
                    target_data, dtype=torch.float32, device=self.device
                )
                loss = loss + self.loss_fn(output_tensor, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.latest_loss = loss.item()
        return {"loss": loss.item()}

    def update_target_net(self):
        target_dict = self.target_net.state_dict()
        policy_dict = self.net.state_dict()
        for key in policy_dict:
            target_dict[key] = policy_dict[key] * self.metadata["tau"] + target_dict[
                key
            ] * (1 - self.metadata["tau"])
        self.target_net.load_state_dict(target_dict)

    def checkpoint_data(self):
        return {
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.latest_loss or 1.0,
        }

    def load_checkpoint_data(self, data):
        self.net.load_state_dict(data["model_state_dict"])
        self.net.eval()


class Network(nn.Module):
    """Pytorch wrapper and more.
    Compose all of the subnetworks into a single network."""

    def __init__(self, plans):
        super().__init__()
        self.architecture = nn.ModuleDict()
        self.plans = plans
        self.vision_size = plans.get("vision_size", None)

        self.inputs = list(
            {
                field
                for plan in plans["architecture"].values()
                for layer in plan
                if layer["type"] == "input"
                for field in layer["source"].split("+")
                if field not in plans["architecture"]
            }
        )

        for plan_name, plan in self.plans["architecture"].items():
            self.compose_network(plan, plan_name)

    def compose_network(self, layers, plan_name):
        """Compose a network based on the layer list."""
        subnetwork = nn.ModuleList()
        input_size = None
        vision_size = None

        for layer_config in layers:
            if layer_config["type"] in ["input"]:
                input_size = layer_config["size"]
                if (
                    layer_config["type"] == "input"
                    and layer_config.get("source") == "vision"
                ):
                    vision_size = self.vision_size
                continue

            elif layer_config["type"] == "conv":
                output_channels = layer_config["size"]
                kernel_size = layer_config["kernel"]
                layer = nn.Conv2d(input_size, output_channels, kernel_size, stride=1)

                if vision_size is not None:
                    vision_size = self.calc_conv_shape(
                        vision_size, layer, transpose=False
                    )
                input_size = output_channels

            elif layer_config["type"] == "linear":
                output_size = layer_config["size"]

                if vision_size is not None:
                    actual_input_size = input_size * int(np.prod(vision_size))
                    subnetwork.append(nn.Flatten(start_dim=1))
                    vision_size = None
                else:
                    actual_input_size = input_size

                layer = nn.Linear(actual_input_size, output_size)
                input_size = output_size

            elif layer_config["type"] == "relu":
                layer = nn.ReLU()

            elif layer_config["type"] == "conv_transpose":
                output_channels = layer_config["size"]
                kernel_size = layer_config["kernel"]
                layer = nn.ConvTranspose2d(
                    input_size, output_channels, kernel_size, stride=1
                )

                if vision_size is not None:
                    vision_size = self.calc_conv_shape(
                        vision_size, layer, transpose=True
                    )
                input_size = output_channels

            elif layer_config["type"] == "unflatten":
                target_shape = layer_config["size"]
                unflatten_shape = (target_shape[2], target_shape[0], target_shape[1])
                layer = nn.Unflatten(1, unflatten_shape)
                vision_size = target_shape[:2]
                input_size = target_shape[2]

            elif layer_config["type"] == "output":
                continue
            else:
                continue

            subnetwork.append(layer)
        self.architecture[plan_name] = subnetwork

    def calc_conv_shape(self, input_size, layer, transpose=False):
        input_size = np.array(input_size)
        kernel = np.array(layer.kernel_size)
        padding = (
            np.array(layer.padding) if hasattr(layer, "padding") else np.array([0, 0])
        )
        stride = (
            np.array(layer.stride) if hasattr(layer, "stride") else np.array([1, 1])
        )

        if transpose:
            output_size = (input_size - 1) * stride - 2 * padding + kernel
        else:
            output_size = (
                np.floor((input_size - kernel + 2 * padding) / stride).astype(int) + 1
            )

        return output_size.tolist()

    def forward(self, input_batch):
        """Process input through all network components following config order."""
        device = next(self.parameters()).device
        activations = {
            k: (
                torch.tensor(v, dtype=torch.float32)
                if not isinstance(v, torch.Tensor)
                else v
            ).to(device)
            for k, v in input_batch.items()
        }
        outputs = {}

        for component_name, plan in self.plans["architecture"].items():
            input_layer = next(layer for layer in plan if layer["type"] == "input")
            sources = input_layer["source"].split("+")

            inputs = []
            for src in sources:
                if src == "action":
                    action_tensor = activations["action"].long().squeeze()
                    one_hot = torch.zeros(
                        activations["action"].size(0),
                        self.plans["n_actions"],
                        device=activations["action"].device,
                    )
                    one_hot.scatter_(1, action_tensor.view(-1, 1), 1)
                    inputs.append(one_hot)
                else:
                    inputs.append(activations[src])

            x = torch.cat(inputs, dim=-1)

            for layer in self.architecture[component_name]:
                x = layer(x)

            activations[component_name] = x

            if any(layer["type"] == "output" for layer in plan):
                outputs[component_name] = x

        return outputs


# ------------- Strategy components


class DQN(Component):
    """DQN -- epsilon-greedy action selection with Bellman Q-learning.

    When any component in the architecture declares n_internal_actions > 0,
    n_actions = n_env_actions + internal_actions. The q_net output is
    split at n_env_actions:
      activations["action"]          <- env action (int), eps-greedy over [:n_env_actions]
      activations["internal_action"] <- internal action (int), eps-greedy over [n_env_actions:]

    Both slices train against the same reward signal via independent Bellman
    targets. internal_action persists in activations between steps so that
    components (e.g. ROI) can read it on the next get_action call.

    When no internal actions exist (n_env_actions == n_actions),
    activations["internal_action"] is not written.

    Passes a loss_fn closure into signals for q_net to execute.
    """

    def __init__(self, context):
        self.component_id = context["component_id"]
        self.inputs = context["inputs"]
        self.components = context["components"]
        self.activations = context["activations"]
        self.cfg = context["cfg"]
        self.metadata = context["plans"]["metadata"]
        self.n_env_actions = context["plans"]["n_env_actions"]
        self.n_actions = context["plans"]["n_actions"]

        self._q_net_id = self.inputs[0]
        self.update_count = 0

    def activate(self, activations):
        self.components[self._q_net_id].activate(activations)
        q_values = list(activations[self._q_net_id].values())[0]
        n = self.n_env_actions
        activations["epsilon"] = epsilon(
            self.cfg.run.experiment.episodes,
            self.metadata["greedy_until"],
            activations["internal_episode"],
        )

        if np.random.random() < activations["epsilon"]:
            activations[self.component_id] = np.random.randint(len(q_values[:n]))
            if n < self.n_actions:
                activations["internal_action"] = np.random.randint(len(q_values[n:]))
        else:
            activations[self.component_id] = int(np.argmax(q_values[:n]))
            if n < self.n_actions:
                activations["internal_action"] = int(np.argmax(q_values[n:]))

    def update(self, signals={}):
        q_net = self.components[self._q_net_id]
        gamma = self.cfg.agent_params.gamma
        loss_fn = q_net.loss_fn
        n = self.n_env_actions
        n_actions = self.n_actions
        self.update_count += 1

        def bellman_loss(net, target_net, tensors):
            state_inputs = {f: tensors[f] for f in net.inputs}
            next_state_inputs = {f: tensors[f"next_{f}"] for f in net.inputs}

            q_values = list(net(state_inputs).values())[0]  # (batch, n_actions)
            with torch.no_grad():
                q_next = list(target_net(next_state_inputs).values())[0]

            reward = tensors["reward"].squeeze(1)
            mask = 1.0 - tensors["done"].squeeze(1)

            env_taken = (
                q_values[:, :n]
                .gather(1, tensors["action"].long().view(-1, 1))
                .squeeze(1)
            )
            total_loss = loss_fn(
                env_taken, reward + gamma * q_next[:, :n].max(dim=1).values * mask
            )

            if n < n_actions:
                internal_taken = (
                    q_values[:, n:]
                    .gather(1, tensors["internal_action"].long().view(-1, 1))
                    .squeeze(1)
                )
                total_loss = total_loss + loss_fn(
                    internal_taken,
                    reward + gamma * q_next[:, n:].max(dim=1).values * mask,
                )

            return total_loss

        report = {}
        if self.update_count % self.metadata["update_frequency"] == 0:
            signals[self._q_net_id] = bellman_loss
            report = q_net.update(signals)
            if self.update_count % q_net.metadata["target_net_update_frequency"] == 0:
                q_net.update_target_net()

        report["episode"] = self.activations.get("episode", 0)
        report["reward"] = float(self.activations.get("reward", 0.0))
        report["explore_episodes"] = int(
            self.cfg.run.experiment.episodes * self.metadata["greedy_until"]
        )
        report["epsilon"] = self.activations.get("epsilon", 0.0)
        report["greedy_until"] = self.metadata["greedy_until"]
        return report


def epsilon(max_episodes, greedy_until, episode):
    """Epsilon-greedy epsilon, decaying linearly to greedy.

    Explores from episode 0 up to greedy_until * total_episodes, then
    exploits fully. greedy_until=0.0 means always greedy (pure exploitation).
    """
    explore_episodes = max(int(max_episodes * greedy_until), 1)
    epsilon = max(1.0 - episode / explore_episodes, 0.0)
    return epsilon


class ModelBased(Component):
    """Model-based RL -- MCTS over dynamics and value components.

    TODO: ROI joint action search -- MCTS must search the joint
    (env_action x internal_action) space. Deferred until ROI baseline is stable.
    """

    def __init__(self, context):
        self.component_id = context["component_id"]
        self.inputs = context["inputs"]
        self.components = context["components"]
        self.activations = context["activations"]
        self.metadata = context["plans"]["metadata"]

        self.dynamics_id = self.inputs[0]
        self.value_id = self.inputs[1]
        self.mcts = MCTS(context["plans"])

    def activate(self, activations):
        def value_fn(state):
            temp = dict(state)
            self.components[self.value_id].activate(temp)
            return float(list(temp[self.value_id].values())[0])

        def dynamics_fn(state, action_idx):
            temp = {**state, "action": np.array([action_idx])}
            self.components[self.dynamics_id].activate(temp)
            output = temp[self.dynamics_id]
            return {
                target.replace("next_", ""): output[name]
                for name, target in zip(
                    output.keys(),
                    self.components[self.dynamics_id].metadata["target"],
                )
            }

        best_action = self.mcts.search(activations, value_fn, dynamics_fn)
        activations[self.component_id] = best_action

    def reset(self):
        pass


class MCTS:
    def __init__(self, plans):
        self.c_puct = plans["metadata"]["c_puct"]
        self.num_simulations = plans["metadata"]["num_simulations"]
        self.max_depth = plans["metadata"]["max_depth"]
        self.n_actions = plans["n_actions"]

    def search(self, root_state, value_fn, dynamics_fn):
        root = MCTSNode(root_state)
        for _ in range(self.num_simulations):
            node = root
            path = [node]
            while node.is_expanded() and len(path) < self.max_depth:
                action_idx = max(
                    node.children.keys(),
                    key=lambda a: node.children[a].ucb_score(self.c_puct),
                )
                node = node.children[action_idx]
                path.append(node)

            if len(path) < self.max_depth:
                self._expand(node, dynamics_fn)
                if node.children:
                    action_idx = np.random.choice(list(node.children.keys()))
                    node = node.children[action_idx]
                    path.append(node)

            value = self._evaluate(node, value_fn)
            for node in path:
                node.visits += 1
                node.value_sum += value

        return max(root.children.keys(), key=lambda a: root.children[a].visits)

    def _expand(self, node, dynamics_fn):
        for action_idx in range(self.n_actions):
            next_state = dynamics_fn(node.state, action_idx)
            node.children[action_idx] = MCTSNode(
                next_state, parent=node, action=action_idx
            )

    def _evaluate(self, node, value_fn):
        return value_fn(node.state)


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, c_puct):
        if self.visits == 0:
            return float("inf")
        exploration = c_puct * np.sqrt(np.log(self.parent.visits) / self.visits)
        return self.value() + exploration
