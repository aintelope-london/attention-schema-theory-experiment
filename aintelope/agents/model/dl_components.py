import numpy as np
import torch
from torch import nn
from aintelope.agents.model.component import Component


class NeuralNet(nn.Module, Component):
    '''PyTorch component.'''
    
    def __init__(self, context):
        #plans, cfg, components, device, checkpoint, role, memory):
        super().__init__()
        self.cfg = context["cfg"]
        self.device = context["device"]
        self.role = context["role"]
        self.memory = context["memory"]
        self.plans = context["plans"]
        self.checkpoint = context["checkpoint"]
        self.metadata = context["plans"]["metadata"]
        
        self.step_count = 0
        self.last_action = None

        self.net = Network(self.plans).to(self.device)
        if self.checkpoint:
            self.load_checkpoint(self.checkpoint)
        
        if self.metadata.get("use_target_net", False):
            self.target_net = Network(self.plans).to(self.device)
            self.target_net.load_state_dict(self.net.state_dict())
        else:
            self.target_net = None
        
        optimizer_cls = getattr(torch.optim, self.metadata["optimizer"])
        self.optimizer = optimizer_cls(self.net.parameters(), **self.metadata["optimizer_params"]) 
        self.loss_fn = getattr(nn, self.metadata["loss_function"])()
        self.latest_loss = None

    def activate(self, observation):
        input_dict = {field: np.expand_dims(observation[field], axis=0) 
                  for field in self.net.inputs if field in observation}
        output = self.net(input_dict)
        confidence = 1.0 #WIP
        return {k: v[0].detach().cpu().numpy() for k, v in output.items()}, confidence

    def update(self):
        # No-op here, needed for MCTS
        return self.optimize()

    def optimize(self):
        if len(self.memory) < self.cfg.dl_params.batch_size:
            return None
        
        batch_data = self.memory.sample(self.cfg.dl_params.batch_size)
        tensors = {field: np.stack([entry[field] for entry in batch_data]) 
           for field in self.net.inputs}
        
        output = self.net(tensors)
        total_loss = 0 
        for (output_name, output_tensor), target_field in zip(output.items(), self.metadata["target"]):
            target_data = np.stack([entry[target_field] for entry in batch_data])
            target = torch.tensor(target_data, dtype=torch.float32, device=self.device)
            
            loss = self.loss_fn(output_tensor, target)
            total_loss += loss
        # NEW END, wrt commented
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}

    def update_target_net(self):
        target_dict = self.target_net.state_dict()
        policy_dict = self.net.state_dict()
        for key in policy_dict:
            target_dict[key] = policy_dict[key] * self.metadata["tau"] + target_dict[key] * (1 - self.metadata["tau"])
        self.target_net.load_state_dict(target_dict)

    def checkpoint_data(self):
        return {
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.latest_loss or 1.0,
        }

    def load_checkpoint(self, path):
        """Assumes that the network has been initialized with correct sizes."""
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval()
    

class Network(nn.Module):
    """Pytorch wrapper and more.
    Compose all of the subnetworks into a single network."""
    
    def __init__(self, plans):
        super().__init__()
        self.architecture = nn.ModuleDict()
        self.plans = plans
        self.vision_size = plans.get("vision_size", None)

        self.inputs = list({field for plan in plans["architecture"].values() 
                        for layer in plan if layer["type"] == "input"
                        for field in layer["source"].split('+')
                        if field not in plans["architecture"]})   # ← T
        
        # Build each component network
        for plan_name, plan in self.plans["architecture"].items():
            self.compose_network(plan, plan_name)

    def compose_network(self, layers, plan_name):
        """Compose a network based on the layer list."""
        subnetwork = nn.ModuleList()
        input_size = None
        vision_size = None  # Local variable for tracking spatial dimensions

        for layer_config in layers:
            if layer_config["type"] in ["input"]:
                input_size = layer_config["size"]
                if layer_config["type"] == "input" and layer_config.get("source") == "vision":
                    vision_size = self.vision_size 
                continue
                
            elif layer_config["type"] == "conv":
                output_channels = layer_config["size"]
                kernel_size = layer_config["kernel"]
                layer = nn.Conv2d(input_size, output_channels, kernel_size, stride=1)
                
                # Update spatial dims only if we're tracking them
                if vision_size is not None:
                    vision_size = self.calc_conv_shape(vision_size, layer, transpose=False)
                input_size = output_channels
                
            elif layer_config["type"] == "linear":
                output_size = layer_config["size"]
                
                # Check if we need to flatten from conv - look for any Conv2d in the subnetwork
                has_conv = any(isinstance(layer, nn.Conv2d) for layer in subnetwork[-2:-1])
                
                if has_conv and vision_size is not None:
                    actual_input_size = input_size * np.prod(vision_size)
                    subnetwork.append(nn.Flatten())
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
                layer = nn.ConvTranspose2d(input_size, output_channels, kernel_size, stride=1)
                
                # Update spatial dims if we're tracking them
                if vision_size is not None:
                    vision_size = self.calc_conv_shape(vision_size, layer, transpose=True)
                input_size = output_channels
                
            elif layer_config["type"] == "unflatten":
                target_shape = layer_config["size"]  
                unflatten_shape = (target_shape[2], target_shape[0], target_shape[1])
                layer = nn.Unflatten(1, unflatten_shape)
                # Resume spatial tracking after unflatten
                vision_size = target_shape[:2]  # [H, W]
                input_size = target_shape[2]  # Channels
                
            elif layer_config["type"] == "output":
                continue
            else:
                continue
                
            subnetwork.append(layer)
        self.architecture[plan_name] = subnetwork

    def calc_conv_shape(self, input_size, layer, transpose=False):
        """
        Calculate output spatial dimensions for conv2d or conv_transpose2d.
        
        Args:
            input_size: Current spatial dimensions [H, W]
            layer: The conv or conv_transpose layer
            transpose: True for conv_transpose, False for regular conv
        """
        input_size = np.array(input_size)
        kernel = np.array(layer.kernel_size)
        padding = np.array(layer.padding) if hasattr(layer, 'padding') else np.array([0, 0])
        stride = np.array(layer.stride) if hasattr(layer, 'stride') else np.array([1, 1])
        
        if transpose:
            # ConvTranspose2d: output = (input - 1) * stride - 2*padding + kernel
            output_size = (input_size - 1) * stride - 2 * padding + kernel
        else:
            # Conv2d: output = floor((input - kernel + 2*padding) / stride) + 1
            output_size = np.floor((input_size - kernel + 2 * padding) / stride).astype(int) + 1
        
        return output_size.tolist()
                
    def forward(self, input_batch):
        """Process input through all network components following config order."""
        activations = {k: torch.tensor(v, dtype=torch.float32) 
                for k, v in input_batch.items()}
        outputs = {}
   
        for component_name, plan in self.plans["architecture"].items():
            input_layer = next(layer for layer in plan if layer["type"] == "input")
            sources = input_layer["source"].split('+')
            
            # Handle action one-hot encoding inline
            inputs = []
            for src in sources:
                if src == "action":
                    action_tensor = activations["action"].long().squeeze()
                    one_hot = torch.zeros(activations["action"].size(0), self.plans["n_actions"], 
                                        device=activations["action"].device)
                    one_hot.scatter_(1, action_tensor.view(-1, 1), 1)
                    inputs.append(one_hot)
                else:
                    inputs.append(activations[src])
        
            x = torch.cat(inputs, dim=-1)

            for layer in self.architecture[component_name]:
                x = layer(x)
            
            activations[component_name] = x
            
            # Check if this component has an output declaration
            if any(layer["type"] == "output" for layer in plan):
                outputs[component_name] = x
        
        return outputs
    
# ------------- Other

class DQN(Component):
    """DQN wrapper - selects actions using epsilon-greedy from Q-network."""
    
    def __init__(self, context):
        #cfg, components, device, checkpoint, role, memory):
        self.role = context["role"]
        self.memory = context["memory"]
        self.cfg = context["cfg"]
        self.metadata = context["plans"]["metadata"]
        
        # Reference the Q-network from components 
        self.q_network = context["components"]["q_network"]
        
        # Epsilon parameters
        self.eps_start = self.metadata.get("eps_start", self.cfg.rl_params.eps_start)
        self.eps_end = self.metadata.get("eps_end", self.cfg.rl_params.eps_end)
        self.eps_decay_steps = self.metadata.get("eps_decay_steps", self.cfg.rl_params.eps_last_step)
        self.step_count = 0
        self.last_action = None

    def activate(self, observation):
        # Get Q-values from network
        output_dict, _ = self.q_network.activate(observation)
        q_values = list(output_dict.values())[0]
        
        # Epsilon-greedy selection
        epsilon = self._compute_epsilon()
        if np.random.random() < epsilon:
            action = np.random.randint(len(q_values))
        else:
            action = int(np.argmax(q_values))
        
        self.step_count += 1
        self.last_action = action
        return {"action": action}, 1.0
    
    def update(self):
        # Delegate to underlying Q-network
        return self.q_network.update()
    
    def _compute_epsilon(self):
        decay_rate = (self.eps_start - self.eps_end) / self.eps_decay_steps
        return max(self.eps_end, self.eps_start - self.step_count * decay_rate) 

class ModelBased(Component):
    """Model-based RL implementation (V(S), SAE, MCTS)."""
    
    def __init__(self, context):
        #cfg, components, device, checkpoint, role, memory):
        self.role = context["role"]
        self.memory = context["memory"]
        self.dynamics_fn = context["components"]["dynamic"]
        self.value_fn = context["components"]["value"]
        self.mcts = MCTS(context["plans"], self.dynamics_fn, self.value_fn)
        self.last_action = None
        self.metadata = context["plans"]["metadata"]

    def activate(self, observation):
        def value_fn(observation):
            output_dict, _ = self.value_fn.activate(observation)
            return float(list(output_dict.values())[0])

        def dynamics_fn(observation, action_idx):
            input_dict = {**observation, "action": np.array([action_idx])}
            output_dict, _ = self.dynamics_fn.activate(input_dict)
            # NEW, wrt commented below
            result = {} 
            for output_name, target_field in zip(output_dict.keys(), self.dynamics_fn.metadata["target"]):
                obs_field = target_field.replace("next_", "")  # next_vision -> vision
                result[obs_field] = output_dict[output_name]
            
            return result
            '''return {
                'vision': output_dict['vision_denet'],
                'audio': output_dict['audio_denet'],
                'interoception': output_dict['interoception_denet']
            }'''
        
        best_action_idx = self.mcts.search(observation, value_fn, dynamics_fn)
        self.last_action = best_action_idx
        confidence = 1.0
        return {"action": best_action_idx}, confidence

    def update(self):
        reports = {}
        reports["dynamic"] = self.dynamics_fn.update()
        reports["value"] = self.value_fn.update()
        return {"mcts_reports": reports}


class MCTS:
    def __init__(self, plans, dynamics_component, value_component):
        self.c_puct = plans["metadata"]["c_puct"]
        self.num_simulations = plans["metadata"]["num_simulations"] 
        self.max_depth = plans["metadata"]["max_depth"]
        self.dynamics_component = dynamics_component
        self.value_component = value_component
        self.n_actions = plans["n_actions"]
        
    def search(self, root_state, value_fn, dynamics_fn):
        root = MCTSNode(root_state)
        for _ in range(self.num_simulations):
            node = root
            path = [node]
            # Selection
            while node.is_expanded() and len(path) < self.max_depth:
                action_idx = max(node.children.keys(), key=lambda a: node.children[a].ucb_score(self.c_puct))
                node = node.children[action_idx]
                path.append(node)
            # Expansion & Evaluation
            if len(path) < self.max_depth:
                self._expand(node, dynamics_fn)
                if node.children:
                    action_idx = np.random.choice(list(node.children.keys()))
                    node = node.children[action_idx]
                    path.append(node)
            # Simulation
            value = self._evaluate(node, value_fn)
            # Backpropagation
            for node in path:
                node.visits += 1
                node.value_sum += value
        # Select best action
        best_action_idx = max(root.children.keys(), key=lambda a: root.children[a].visits)
        return best_action_idx

    def _expand(self, node, dynamics_fn):
        for action_idx in range(self.n_actions):
            # Compute state immediately during expansion
            next_state = dynamics_fn(node.state, action_idx)
            node.children[action_idx] = MCTSNode(next_state, parent=node, action=action_idx)

    def _evaluate(self, node, value_fn):
        # State always exists, just evaluate
        return value_fn(node.state) if node.state is not None else 0.0


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
            return float('inf')
        exploration = c_puct * np.sqrt(np.log(self.parent.visits) / self.visits)
        return self.value() + exploration